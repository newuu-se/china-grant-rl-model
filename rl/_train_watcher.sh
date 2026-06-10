#!/usr/bin/env bash
# Waits for the PPO training run to reach a terminal state (success or crash),
# then writes results/training_summary_<ts>.txt and preserves a copy of the log.
# Runs detached in its own tmux session so the save happens even if the Claude
# Code session has closed. The phone push is handled separately by the monitor.
set -u
cd "$(dirname "$0")/.."

LOG="${1:?usage: _train_watcher.sh <train_log_path>}"
TERMINAL='Training complete|Traceback|Killed|Aborted|Segmentation fault|MemoryError|RuntimeError|Error:|stalled|terminated before returning'

# Poll until training finishes or crashes.
while true; do
    if grep -qE "$TERMINAL" "$LOG" 2>/dev/null; then
        break
    fi
    sleep 30
done

TS=$(date +%Y%m%d_%H%M%S)
mkdir -p results
SUMMARY="results/training_summary_${TS}.txt"
LOGCOPY="results/training_log_${TS}.log"

cp "$LOG" "$LOGCOPY" 2>/dev/null || true

if grep -q "Training complete" "$LOG"; then
    STATUS="COMPLETED"
else
    STATUS="FAILED / ABORTED"
fi

{
    echo "════════════════════════════════════════════════════════════"
    echo "  PPO Training Session Summary"
    echo "════════════════════════════════════════════════════════════"
    echo "  generated : $(date)"
    echo "  status    : $STATUS"
    echo "  source log: $LOG"
    echo "  log copy  : $LOGCOPY"
    echo ""
    echo "── final metrics ────────────────────────────────────────────"
    grep -E "Training complete|Best reward|Final model" "$LOG" | tail -5
    echo ""
    echo "── best_reward per test epoch (last 30) ─────────────────────"
    grep "best_reward:" "$LOG" | tail -30
    echo ""
    echo "── last 8 episode outcomes ──────────────────────────────────"
    grep -E "ARRIVED|TIMEOUT" "$LOG" | tail -8
    echo ""
    echo "── checkpoints saved ────────────────────────────────────────"
    ls -1 checkpoints/*.pth 2>/dev/null | sed 's,.*/,  ,'
    echo ""
    echo "── tail of log (last 30 lines) ──────────────────────────────"
    tail -30 "$LOG"
} > "$SUMMARY"

echo "$SUMMARY" > results/.last_training_summary
echo "[watcher] training reached terminal state ($STATUS); wrote $SUMMARY"
