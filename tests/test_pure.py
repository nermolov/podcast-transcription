"""Tests for pure functions. Run with `python tests/test_pure.py` (or pytest, if available)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from transcriber.pipeline import EmbedMapper, OverlapMapper, Stitcher, align, fixed_plan, plan_chunks, progressive_plan, split_turn, srt_ts, stitch


def test_plan_chunks():
    assert plan_chunks(100, 3000, 60) == [(0.0, 100)]
    p = plan_chunks(7000, 3000, 60)
    assert len(p) == 3 and p[0][0] == 0 and abs(p[-1][0] + p[-1][1] - 7000) < 0.1
    assert all(b[0] < a[0] + a[1] for a, b in zip(p, p[1:]))  # consecutive chunks overlap


def seg(s, e, spk, text): return dict(start=s, end=e, speaker_id=spk, text=text)


def test_stitch_offsets_dedups_and_maps_speakers():
    plan = [(0.0, 100.0), (60.0, 100.0)]
    c0 = [seg(0, 10, 0, "hello there"), seg(70, 80, 1, "overlap line")]
    c1 = [seg(10, 20, 0, "overlap line"), seg(50, 60, 1, "later words")]  # raw labels flipped; times relative to chunk start
    final, flips = stitch([c0, c1], plan, 60)
    assert [s["text"] for s in final] == ["hello there", "overlap line", "later words"]
    assert [s["speaker"] for s in final] == [0, 1, 0] and flips[0]["raw_labels_flipped"]
    assert final[-1]["start"] == 110.0


def test_align_matches_and_interpolates():
    vv = [dict(start=0.0, end=3.0, speaker=0, text="hello big world")]
    pw = [dict(w="Hello", start=0.1, end=0.5), dict(w="world", start=2.0, end=2.5)]
    turns, tot = align(vv, pw)
    w = turns[0]["words"]
    assert [x["matched"] for x in w] == [True, False, True] and tot["matched"] == 2 and tot["interpolated"] == 1
    assert w[0]["start"] == 0.1 and w[2]["end"] == 2.5 and w[0]["end"] <= w[1]["start"] <= w[2]["start"]


def test_srt_and_split():
    assert srt_ts(3661.5) == "01:01:01,500"
    ws = [dict(w=f"w{i}", start=i * 0.2, end=i * 0.2 + 0.1) for i in range(30)]
    assert [len(c) for c in split_turn(ws)] == [14, 14, 2]


def test_progressive_plan():
    p = progressive_plan(6622.5, 2400, 30)
    assert [round(e - n) for _, n, e in p] == [120, 360, 1200, 2400, 2542]  # short tail folded into the last chunk
    assert all(a == n - 30 for a, n, _ in p[1:]) and p[0] == (0.0, 0.0, 120.0) and p[-1][2] == 6622.5
    assert progressive_plan(100, 2400, 30) == [(0.0, 0.0, 100)]
    assert progressive_plan(600, 2400, 30)[-1] == (450.0, 480.0, 600)
    assert all(e - a <= 59 * 60 for a, _, e in progressive_plan(3 * 3540 + 10, 3500, 30))  # folding never exceeds the model limit
    assert fixed_plan(7000, 3000, 60)[1][1] == fixed_plan(7000, 3000, 60)[0][2]


def S(s, e, spk, text): return dict(start=s, end=e, speaker_id=spk, text=text)


def test_stitcher_streams_finals_and_waits_for_seam():
    plan = [(0.0, 0.0, 100.0), (70.0, 100.0, 200.0)]
    st = Stitcher(plan, OverlapMapper())
    assert [f["text"] for f in st.add(0, S(0, 30, 0, "a b c"))] == ["a b c"] and st.ready_until == 30  # final at once
    assert [f["text"] for f in st.add(0, S(30, 60, 1, "d e f"))] == ["d e f"]
    assert st.add(0, S(72, 90, 0, "g h i")) == []  # inside the next overlap: waits for the seam
    assert st.add(0, S(90, 100, 1, "j k")) == [] and st.chunk_done(0) == []
    assert st.add(1, S(0, 2, 1, "x y")) == []  # chunk 1, still inside the overlap: no mapping yet
    out = st.add(1, S(2, 20, 0, "g h i"))  # ends at 90 < new_start: still undecided
    out += st.add(1, S(20, 35, 1, "j k l m"))  # ends at 105 >= new_start: mapping + seam decided
    # cut at 90, where chunk 1 starts a segment and chunk 0 ended one: chunk 0's "g h i", chunk 1's "j k l m"
    assert [(f["text"], f["speaker"], f["chunk"]) for f in out] == [("g h i", 0, 0), ("j k l m", 1, 1)]
    assert st.boundaries[0]["seam"] == "start" and st.boundaries[0]["cut"] == 90
    out = st.add(1, S(35, 50, 0, "n o")) + st.chunk_done(1) + st.finish()
    assert [f["text"] for f in out] == ["n o"] and st.ready_until == 200.0
    assert [f["start"] for f in st.final] == sorted(f["start"] for f in st.final)


def test_stitcher_dedupes_overlap_spanning_segment():
    # chunk 1 opens with one segment spanning its whole overlap: no loss, no duplicated words
    plan = [(0.0, 0.0, 100.0), (70.0, 100.0, 200.0)]
    st = Stitcher(plan, OverlapMapper())
    st.add(0, S(0, 70.3, 0, "intro words here")); st.add(0, S(70.3, 100, 1, "so this is like writing tests for the"))
    st.chunk_done(0)
    st.add(1, S(0, 32.3, 0, "tests for the models where the model writes")); st.chunk_done(1); st.finish()
    assert [(f["text"], f["speaker"]) for f in st.final] == [("intro words here", 0), ("so this is like writing tests for the", 1),
                                                            ("models where the model writes", 1)]
    assert st.boundaries[0]["seam"] == "dedupe" and st.boundaries[0]["dedupe_words"] == 3


class FakeEmbed:
    """Speaker identity from a timeline of (start, end, true speaker) -> one-hot-ish unit vectors."""
    def __init__(self, truth): self.truth = truth

    def __call__(self, st, en):
        import numpy as np
        m = (st + en) / 2
        who = next(w for a, b, w in self.truth if a <= m < b)
        v = np.full(8, 0.05); v[who] = 1.0; return v / np.linalg.norm(v)


def test_embed_mapper_three_speakers_single_speaker_overlap():
    # true speakers A=0, B=1, C=2. The overlap [90,100] holds only C, so overlap matching maps C and must guess the others by
    # elimination; chunk 1 labels them 0=C, 1=B, 2=A, and elimination picks the wrong way round.
    truth = [(0, 30, 0), (30, 60, 1), (60, 100, 2), (100, 130, 1), (130, 160, 0), (160, 200, 2)]
    plan = [(0.0, 0.0, 100.0), (90.0, 100.0, 200.0)]
    c0 = [S(0, 30, 0, "a"), S(30, 60, 1, "b"), S(60, 100, 2, "see you there friend")]
    c1 = [S(0, 10, 0, "there friend"), S(10, 40, 1, "b2"), S(40, 70, 2, "a2"), S(70, 110, 0, "c2")]
    res = {}
    for name, mapper in (("overlap", OverlapMapper()), ("embed", EmbedMapper(FakeEmbed(truth)))):
        st = Stitcher(plan, mapper)
        for s in c0: st.add(0, s)
        st.chunk_done(0)
        for s in c1: st.add(1, s)
        st.chunk_done(1); st.finish()
        res[name] = [(f["text"], f["speaker"]) for f in st.final]
    assert res["embed"] == [("a", 0), ("b", 1), ("see you there friend", 2), ("b2", 1), ("a2", 0), ("c2", 2)]
    assert [x[1] for x in res["overlap"]] == [0, 1, 2, 0, 1, 2]  # elimination swaps A and B


def test_embed_mapper_new_speaker_and_short_labels_wait():
    truth = [(0, 50, 0), (50, 100, 1), (100, 101, 1), (101, 140, 3), (140, 200, 0)]
    plan = [(0.0, 0.0, 100.0), (90.0, 100.0, 200.0)]
    st = Stitcher(plan, EmbedMapper(FakeEmbed(truth)))
    for s in (S(0, 50, 0, "a"), S(50, 100, 1, "b")): st.add(0, s)
    st.chunk_done(0)
    st.add(1, S(0, 11, 0, "b2")); assert st.maps[1] == {0: 1}  # B continues past the overlap -> chunk 1 mapped
    assert st.add(1, S(11, 11.3, 1, "yeah")) == []  # label 1 seen only on a 0.3 s backchannel: wait instead of guessing
    out = st.add(1, S(11.3, 50, 1, "new voice talks for a while"))
    assert [f["speaker"] for f in out] == [2, 2]  # >= 3 s that matches nobody -> new speaker 2 (and the backchannel with it)
    st.add(1, S(50, 110, 2, "a again")); st.chunk_done(1); st.finish()
    assert [f["speaker"] for f in st.final] == [0, 1, 1, 2, 2, 0]


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"): f(); print("ok", k)
