"""
Training Monitor — real-time dashboard via Modal logs.

Usage:
    # Terminal 1: start training
    modal run scripts/train_modal.py

    # Terminal 2: watch live
    python scripts/monitor.py

    # Or use Modal's built-in:
    modal logs deeplm-train --follow
"""
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path


def watch_logs(log_file: str, refresh: float = 2.0):
    """Tail the JSONL log file and display a live dashboard."""
    if not os.path.exists(log_file):
        print(f"⏳ Waiting for log file: {log_file}")
        while not os.path.exists(log_file):
            time.sleep(1)

    print(f"📊 Monitoring: {log_file}")
    print(f"{'='*60}")
    
    last_step = 0
    while True:
        try:
            entries = []
            with open(log_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entries.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue

            # Show latest entries
            for e in entries:
                if e.get("event") == "step" and e.get("step", 0) > last_step:
                    _print_step(e)
                    last_step = e["step"]
                elif e.get("event") == "checkpoint":
                    _print_checkpoint(e)
                elif e.get("event") == "complete":
                    _print_summary(e)
                    return
                elif e.get("event") == "start":
                    pass  # header already shown

            time.sleep(refresh)

        except KeyboardInterrupt:
            print("\n👋 Monitoring stopped")
            break
        except Exception as ex:
            print(f"⚠ Error: {ex}")
            time.sleep(refresh)


def _print_step(e: dict):
    status = {"ok": "✓", "spike": "⚡", "diverging": "📈",
              "nan": "💀", "grad_explosion": "💥", "plateau": "⏸"}.get(e.get("status", ""), "?")
    
    eta = e.get("eta_sec", 0)
    eta_s = f"{eta//3600:.0f}h{(eta%3600)//60:.0f}m" if eta > 60 else f"{eta:.0f}s"
    mem = f" | VRAM {e.get('gpu_mem_gb', 0):.1f}GB" if e.get("gpu_mem_gb", 0) > 0 else ""

    print(
        f"  {e['step']:>6,} | "
        f"L {e['loss']:>8.4f} | "
        f"μ {e['avg_loss']:>8.4f} | "
        f"LR {e['lr']:.2e} | "
        f"G {e['grad_norm']:>6.2f} | "
        f"{e['tokens_per_sec']:>7,} tok/s | "
        f"ETA {eta_s}{mem} | {status}"
    )


def _print_checkpoint(e: dict):
    star = "★" if e.get("is_best") else "✓"
    print(f"  {star} CP step-{e['step']} | loss {e['loss']:.4f}")


def _print_summary(e: dict):
    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE")
    print(f"  Steps: {e['total_steps']:,}")
    print(f"  Time: {e['total_time_hours']:.2f}h")
    print(f"  Best loss: {e['best_loss']:.4f}")
    print(f"  Final loss: {e['final_loss']:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    # Default log file path (Modal volume mount)
    log_path = sys.argv[1] if len(sys.argv) > 1 else "/vol/output/training.jsonl"
    
    if not os.path.exists(log_path):
        # Try local path
        log_path = sys.argv[1] if len(sys.argv) > 1 else "./deeplm_output/training.jsonl"
    
    watch_logs(log_path)
