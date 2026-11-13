"""Tests for pure functions. Run with `python tests/test_pure.py` (or pytest, if available)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from transcriber.pipeline import align, plan_chunks, split_turn, srt_ts, stitch


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


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"): f(); print("ok", k)
