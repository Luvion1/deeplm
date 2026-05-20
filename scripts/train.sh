#!/usr/bin/env bash
set -uo pipefail

export MODAL_LOGS_TIMEOUT=3600

modal run --detach scripts/train_modal.py "$@"
echo ""
echo "═══════════════════════════════════════════"
echo " Training started on Modal (detached)"
echo " Following logs — Ctrl+C stops following"
echo "═══════════════════════════════════════════"
sleep 5
modal logs -f scripts/train_modal.py
