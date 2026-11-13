#!/bin/bash
# Build the server, pin it with a GC root, install a launchd LaunchAgent and start it.
# Config via env (all optional): LABEL STATE_DIR HOST PORT API_TOKEN HF_HOME LOG_DIR
set -euo pipefail
shopt -u patsub_replacement 2>/dev/null || true   # bash 5.2: keep '&' literal in ${//}
export PATH=/nix/var/nix/profiles/default/bin:$PATH
cd "$(dirname "$0")/.."

LABEL=${LABEL:-com.local.transcribe-service}
STATE_DIR=${STATE_DIR:-$HOME/.local/state/transcribe-service}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8000}
API_TOKEN=${API_TOKEN:-}
HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
LOG_DIR=${LOG_DIR:-$HOME/Library/Logs/transcribe-service}
GCROOT_DIR=$HOME/.local/share/$LABEL
PLIST=$HOME/Library/LaunchAgents/$LABEL.plist
DOMAIN=gui/$(id -u)

mkdir -p "$STATE_DIR" "$LOG_DIR" "$GCROOT_DIR" "$HOME/Library/LaunchAgents"
# --out-link creates a GC root, so the store path survives `nix-collect-garbage`. Flake files must be git-tracked.
OUT=$(nix build .#server --out-link "$GCROOT_DIR/server" --print-out-paths | tail -1)
BIN=$OUT/bin/transcribe-server
[ -x "$BIN" ] || { echo "missing $BIN" >&2; exit 1; }

xml() { local v=$1; v=${v//&/&amp;}; v=${v//</&lt;}; v=${v//>/&gt;}; printf '%s' "$v"; }
content=$(<deploy/com.local.transcribe-service.plist)
for kv in LABEL:"$LABEL" BIN:"$BIN" STATE_DIR:"$STATE_DIR" LOG_DIR:"$LOG_DIR" HOME:"$HOME" HOST:"$HOST" PORT:"$PORT" API_TOKEN:"$API_TOKEN" HF_HOME:"$HF_HOME"; do
  content=${content//@${kv%%:*}@/$(xml "${kv#*:}")}
done
umask 077   # the plist may contain API_TOKEN
printf '%s\n' "$content" > "$PLIST"
plutil -lint "$PLIST"

launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null || true   # idempotent re-install
launchctl bootstrap "$DOMAIN" "$PLIST"
launchctl kickstart "$DOMAIN/$LABEL" 2>/dev/null || true

echo "installed $LABEL -> $BIN"
echo "logs: $LOG_DIR/server.log   state: $STATE_DIR   url: http://$HOST:$PORT/healthz"
for _ in $(seq 1 30); do
  curl -fsS "http://$HOST:$PORT/healthz" 2>/dev/null && { echo; exit 0; }
  sleep 1
done
echo "warning: no /healthz response after 30 s; see $LOG_DIR/server.log" >&2
exit 1
