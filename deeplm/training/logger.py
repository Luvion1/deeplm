"""
Enterprise training logger — real-time, persistent, smart monitoring.

Features:
- Real-time log streaming to console + file
- Smart anomaly detection (loss spikes, NaN, gradient explosion)
- ETA prediction with adaptive smoothing
- Memory usage tracking
- Checkpoint management with best-loss tracking
- JSONL format for easy parsing
"""
import json
import math
import os
import time
from collections import deque
from datetime import datetime
from typing import Dict, Optional, Tuple


class SmartLogger:
    """Real-time training logger with anomaly detection and ETA prediction."""

    def __init__(self, log_dir: str, total_steps: int, model_params: int,
                 batch_size: int, grad_accum: int, seq_length: int,
                 lr: float, vocab_size: int, gpu_name: str):
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, "training.jsonl")
        self.summary_file = os.path.join(log_dir, "summary.json")
        self.total_steps = total_steps

        # ETA tracking
        self.step_times = deque(maxlen=100)
        self.start_time = time.time()
        self.last_step_time = None

        # Loss smoothing
        self.loss_history = deque(maxlen=500)
        self.recent_losses = deque(maxlen=20)
        self._ema_loss = None

        # Anomaly detection
        self.loss_baseline = None
        self.anomaly_count = 0
        self.last_anomaly_step = 0

        # Throughput tracking
        self.total_tokens = 0
        self.tokens_per_sec_history = deque(maxlen=100)

        # Initial log entry
        self._write({
            "event": "start",
            "timestamp": datetime.now().isoformat(),
            "model_params": model_params,
            "vocab_size": vocab_size,
            "batch_size": batch_size,
            "grad_accum": grad_accum,
            "effective_batch": batch_size * grad_accum,
            "seq_length": seq_length,
            "learning_rate": lr,
            "total_steps": total_steps,
            "gpu": gpu_name,
        })

        self._print_header(model_params, batch_size, grad_accum, seq_length, lr, total_steps)

    def _print_header(self, params, bs, ga, sl, lr, total):
        eff = bs * ga
        print(f"\n{'='*65}")
        print(f"  Deeplm Training — {params/1e6:.0f}M params — {total:,} steps")
        print(f"{'='*65}")
        print(f"  Batch: {bs} x {ga} = {eff} eff | Seq: {sl} | LR: {lr}")
        print(f"  Log: {self.log_file}")
        print(f"{'='*65}")
        print(f"  {'Step':>6} | {'Loss':>8} | {'Avg':>8} | {'LR':>10} | "
              f"{'Grad':>6} | {'Tok/s':>8} | {'Time':>7} | {'Status'}")
        print(f"{'-'*65}")

    def log_step(self, step: int, loss: float, grad_norm: float, lr: float,
                 tokens_per_sec: float, gpu_mem: float = 0, mtp_loss: float = None):
        """Log a single training step."""
        now = time.time()
        elapsed = now - self.start_time

        if self.last_step_time:
            self.step_times.append(now - self.last_step_time)
        self.last_step_time = now

        self.loss_history.append(loss)
        self.recent_losses.append(loss)
        self.tokens_per_sec_history.append(tokens_per_sec)
        self.total_tokens += int(tokens_per_sec * (self.step_times[-1] if self.step_times else 1))

        # Compute averages
        avg_loss = sum(self.recent_losses) / len(self.recent_losses)
        smooth_loss = self._smooth_loss(loss)

        # ETA prediction
        eta = self._predict_eta(step)

        # Anomaly detection
        status, details = self._detect_anomalies(step, loss, grad_norm, avg_loss)

        # Build log entry
        entry = {
            "event": "step",
            "step": step,
            "progress_pct": round(step / self.total_steps * 100, 1),
            "loss": round(loss, 4),
            "avg_loss": round(avg_loss, 4),
            "smooth_loss": round(smooth_loss, 4),
            "lr": round(lr, 8),
            "grad_norm": round(grad_norm, 4),
            "tokens_per_sec": round(tokens_per_sec),
            "elapsed_sec": round(elapsed),
            "eta_sec": round(eta),
            "gpu_mem_gb": round(gpu_mem, 2),
            "status": status,
            "timestamp": datetime.now().isoformat(),
        }
        if mtp_loss is not None:
            entry["mtp_loss"] = round(mtp_loss, 4)

        # Write to persistent file
        self._write(entry)

        # Console output
        eta_str = f"{eta//3600:.0f}h{(eta%3600)//60:.0f}m" if eta > 60 else f"{eta:.0f}s"
        status_flag = self._status_flag(status)

        print(f"  {step:>6,} | {loss:>8.4f} | {avg_loss:>8.4f} | {lr:>10.2e} | "
              f"{grad_norm:>6.2f} | {tokens_per_sec:>8,.0f} | {elapsed//60:>3.0f}m{elapsed%60:>02.0f}s | {status_flag}")

        # If anomaly, print details
        if status != "ok" and details:
            print(f"  WARNING: {details}")

    def log_checkpoint(self, step: int, path: str, loss: float, is_best: bool = False):
        """Log checkpoint save event."""
        entry = {
            "event": "checkpoint",
            "step": step,
            "path": path,
            "loss": round(loss, 4),
            "is_best": is_best,
            "timestamp": datetime.now().isoformat(),
        }
        self._write(entry)
        badge = "BEST" if is_best else "OK"
        print(f"  [{badge}] Checkpoint saved: {path}")

    def log_summary(self, step: int, final_loss: float, best_loss: float):
        """Write final training summary."""
        total_time = time.time() - self.start_time
        avg_tok_s = 0
        if self.tokens_per_sec_history:
            avg_tok_s = sum(self.tokens_per_sec_history) / len(self.tokens_per_sec_history)

        summary = {
            "event": "complete",
            "total_steps": step,
            "best_loss": round(best_loss, 4),
            "final_loss": round(final_loss, 4),
            "total_time_sec": round(total_time),
            "total_time_hours": round(total_time / 3600, 2),
            "avg_tokens_per_sec": round(avg_tok_s),
            "completed_at": datetime.now().isoformat(),
        }
        with open(self.summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        self._write(summary)

        print(f"\n{'='*65}")
        print(f"  TRAINING COMPLETE!")
        print(f"  Steps: {step:,} | Time: {total_time/3600:.1f}h")
        print(f"  Best loss: {best_loss:.4f} | Final loss: {final_loss:.4f}")
        print(f"  Avg tok/s: {avg_tok_s:,.0f}")
        print(f"  Summary: {self.summary_file}")
        print(f"{'='*65}")

    # -- Private helpers --

    def _write(self, entry: Dict):
        """Write to persistent JSONL file."""
        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _smooth_loss(self, loss: float) -> float:
        """Exponential moving average for smoother loss display."""
        if self._ema_loss is None:
            self._ema_loss = loss
        else:
            self._ema_loss = 0.95 * self._ema_loss + 0.05 * loss
        return self._ema_loss

    def _predict_eta(self, step: int) -> float:
        """Predict remaining time using median of recent step times."""
        if len(self.step_times) < 5 or step == 0:
            return 0
        sorted_times = sorted(self.step_times)
        median_time = sorted_times[len(sorted_times) // 2]
        remaining = self.total_steps - step
        return remaining * median_time

    def _detect_anomalies(self, step: int, loss: float, grad_norm: float, avg_loss: float) -> Tuple[str, str]:
        """Detect training anomalies. Returns (status, detail_message)."""
        # NaN detection
        if math.isnan(loss) or math.isinf(loss):
            self.anomaly_count += 1
            self.last_anomaly_step = step
            return "nan", "NaN detected in loss! Training will likely fail."

        # Gradient explosion
        if grad_norm > 50:
            return "grad_explosion", f"Gradient explosion: {grad_norm:.2f} (threshold: 50)"

        # Loss spike
        if self.loss_baseline is not None and loss > self.loss_baseline * 3:
            self.anomaly_count += 1
            self.last_anomaly_step = step
            if self.anomaly_count > 3:
                return "spike", f"Repeated loss spikes ({self.anomaly_count}x). LR might be too high."
            return "spike", f"Loss spike: {loss:.4f} vs baseline {self.loss_baseline:.4f}"

        # Diverging
        if len(self.recent_losses) >= 10:
            recent_avg = sum(self.recent_losses) / len(self.recent_losses)
            if self.loss_baseline is not None and recent_avg > self.loss_baseline * 1.5:
                return "diverging", f"Loss diverging: avg {recent_avg:.4f} vs baseline {self.loss_baseline:.4f}"

        # Plateau detection
        if len(self.recent_losses) >= 20:
            first_half = sum(list(self.recent_losses)[:10]) / 10
            second_half = sum(list(self.recent_losses)[10:]) / 10
            if abs(first_half - second_half) / max(abs(first_half), 1e-8) < 0.01:
                return "plateau", f"Loss plateau: no improvement in last 20 steps"

        # Update baseline
        if self.loss_baseline is None or loss < self.loss_baseline:
            self.loss_baseline = loss
            self.anomaly_count = 0

        return "ok", ""

    def _status_flag(self, status: str) -> str:
        flags = {
            "ok": "OK",
            "spike": "SPIKE",
            "diverging": "DIVERGE",
            "nan": "NaN",
            "grad_explosion": "GRAD_EXP",
            "plateau": "PLATEAU",
        }
        return flags.get(status, "?")


class MetricsTracker:
    """Tracks running metrics for monitoring dashboard."""

    def __init__(self, window: int = 500):
        self.window = window
        self.data = {}

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if k not in self.data:
                self.data[k] = deque(maxlen=self.window)
            self.data[k].append(v)

    def recent(self, key: str, n: int = 10):
        if key not in self.data or len(self.data[key]) == 0:
            return 0
        recent = list(self.data[key])[-n:]
        return sum(recent) / len(recent)

    def best(self, key: str):
        if key not in self.data or len(self.data[key]) == 0:
            return 0
        return min(self.data[key])

    def latest(self, key: str):
        if key not in self.data or len(self.data[key]) == 0:
            return 0
        return self.data[key][-1]
