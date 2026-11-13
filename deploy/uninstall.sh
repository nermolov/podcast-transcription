#!/bin/bash
# Stop and remove the LaunchAgent, its plist and the GC root. State and logs are kept unless PURGE=1.
# Uses the same LABEL / STATE_DIR / LOG_DIR env as install.sh.
set -euo pipefail
LABEL=${LABEL:-com.local.transcribe-service}
STATE_DIR=${STATE_DIR:-$HOME/.local/state/transcribe-service}
LOG_DIR=${LOG_DIR:-$HOME/Library/Logs/transcribe-service}
PLIST=$HOME/Library/LaunchAgents/$LABEL.plist

launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true   # sends SIGTERM, waits up to ExitTimeOut
rm -f "$PLIST"
rm -rf "$HOME/.local/share/$LABEL"
if [ "${PURGE:-0}" = 1 ]; then rm -rf "$STATE_DIR" "$LOG_DIR"; echo "purged $STATE_DIR and $LOG_DIR"; fi
echo "uninstalled $LABEL"
