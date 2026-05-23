"""
AutoTuner — Adaptive Training Controller.

Self-tuning controller for ALL training aspects:
- Hyperparameters (LR, gradient clipping, weight decay, momentum)
- Dataset routing (category weights, active sources, difficulty)
- Batch & throughput (batch size, grad accum, sequence length)
- Precision (dtype, loss scaling, grad scaling)
- Model architecture (expert top-k, dropout, layer skip)
- Regularization (dropout, label smoothing, gradient penalty)
- Learning schedule (warmup, decay type, restarts)
- MoE specific (routing temperature, load balance, capacity)
- MTP/Auxiliary (MTP depth, auxiliary loss weights)
- Optimizer (betas, eps, weight decay base)

Works standalone (no Modal dependency).
"""
import json
import math
from collections import deque

import torch

from .control import TrainingControl


class AutoTuner:
    # ── Class-level constants ──
    LR_FLOOR = 0.05
    LR_CEIL = 2.0
    GN_FLOOR = 0.1
    GN_CEIL = 2.0
    WD_FLOOR = 0.1
    WD_CEIL = 3.0
    MAX_REDUCTIONS = 5
    MAX_REVIVES = 3
    REVIVE_THR_DIVISOR = 20
    PLATEAU_DIVISOR = 50
    COOLDOWN_DEFAULT = 500
    CONFIDENCE_WARMUP = 3000
    TRAJ_DECAY = 0.995
    HIST_MAXLEN = 1000
    GNHIST_MAXLEN = 200
    ADJMEM_MAXLEN = 100
    LOG_MAXLEN = 1000
    HISTORY_MAXLEN = 2000
    ADJSIGNS_MAXLEN = 8
    ADJWIN_MAXLEN = 20
    EVALLOSS_MAXLEN = 50
    LAYERGN_MAXLEN = 20
    PLANHIST_MAXLEN = 50
    AUTO_LAG_MAXLEN = 6
    AUTO_CORR_SAMPLE = 1_000_000
    HESS_EVAL_EVERY = 50
    KURT_ALPHA = 0.05
    AUTO_LAGS = (1, 3, 5)

    def __init__(self, lr, warmup, total, gn=1.0):
        self.base_lr = lr
        self.warmup = warmup
        self.total = total
        self.base_gn = gn

        # Loss tracking
        self.hist = deque(maxlen=self.HIST_MAXLEN)
        self.ema_short = 0.0
        self.ema_med = 0.0
        self.ema_long = 0.0
        self.short_init = False
        self.med_init = False
        self.long_init = False
        self.best = float("inf")
        self.best_step = 0
        self.plateau = 0
        self.plateau_thr = max(200, total // self.PLATEAU_DIVISOR)
        self.init_best = True

        # Grad norm tracking
        self.gn_hist = deque(maxlen=self.GNHIST_MAXLEN)
        self.gn_ema = 0.0
        self.gn_var = 0.0
        self.gn_var_init = False
        self.last_valid_step = 0
        self.gn_gap_steps = 0

        # Gradient direction (cosine sim)
        self.flat_grad = None
        self.cos_sim_ema = 0.0
        self.cos_sim_init = False

        # Gradient histogram per component (with EMA smoothing)
        self.grad_hist = {}
        self.grad_hist_ema = {}
        self.prev_params = None
        self.update_ratio_ema = 0.0
        self.ur_init = False
        self.ur_alpha = 0.05

        # Gradient noise scale
        self.gn_noise_ema = 0.0
        self.gn_noise_init = False
        self.gn_noise_alpha = 0.1

        # Adaptive multipliers
        self.lr_mult = 1.0
        self.gn_mult = 1.0
        self.mom_boost = 0.0
        self.mom_decay = 0.98

        # Phase & bounds
        self.phase = "warmup"
        self.eff_floor = self.LR_FLOOR
        self.eff_ceil = self.LR_CEIL
        self.cooldown = self.COOLDOWN_DEFAULT
        self.last_change = warmup
        self.last_gn_change = 0
        self.div_reductions = 0
        self.deg_reductions = 0
        self.max_red = self.MAX_REDUCTIONS

        # Adjustment memory
        self.adj_memory = deque(maxlen=self.ADJMEM_MAXLEN)
        self.last_adj_loss = None
        self._last_adj_step = 0
        self._last_adj_state = None
        self._eval_cd = 250

        # Logging
        self.log = deque(maxlen=self.LOG_MAXLEN)
        self.recovery = 0
        self.state = "init"
        self.history = deque(maxlen=self.HISTORY_MAXLEN)
        self.adjustments = 0
        self.anomalies = 0

        # NaN gap tracking
        self._prev_loss = None

        # Train/eval gap (overfitting detector)
        self.eval_losses = deque(maxlen=self.EVALLOSS_MAXLEN)
        self.gap_ema = 0.0
        self.gap_init = False
        self.overfit_signal = False
        self.last_eval_step = 0

        # Weight decay modulation
        self.wd_mult = 1.0
        self.wd_floor = self.WD_FLOOR
        self.wd_ceil = self.WD_CEIL

        # Layer health (per-layer gradient norm)
        self.layer_gn = {}
        self.sick_layers = set()
        self.layer_ratios = {}  # per-group gn ratios (NEW)

        # Confidence system
        self.confidence = 0.0
        self.conf_init = False

        # Revive mechanism
        self.stuck_steps = 0
        self.revive_thr = max(1500, total // self.REVIVE_THR_DIVISOR)
        self.revive_attempts = 0
        self.max_revives = self.MAX_REVIVES

        # Rollback state
        self.prev_lr_mult = 1.0
        self.prev_gn_mult = 1.0
        self.prev_wd_mult = 1.0
        self.rollback_ready = False

        # Oscillation dampener
        self.adj_signs = deque(maxlen=self.ADJSIGNS_MAXLEN)
        self.quiet_mode = False
        self.quiet_until = 0

        # Safety interlock
        self.adj_window = deque(maxlen=self.ADJWIN_MAXLEN)
        self.failed_adjs = 0
        self.frozen = False
        self.frozen_until = 0

        # Global training potential
        self.potential = float("inf")
        self.pot_init = False
        self.pot_alpha = 0.001
        self.dev_potential = 0.0

        # Credit attribution
        self.state_snapshot = {}

        # ── v5: Online dynamics model (Bayesian) ──
        self.dynamics = {
            "global": {
                "lr_sens_mean": 0.0,
                "lr_sens_m2": 0.0,
                "wd_sens_mean": 0.0,
                "wd_sens_m2": 0.0,
                "n": 0,
            },
            "per_phase": {},
        }

        # v5: Counterfactual prediction
        self.last_predicted = 0.0
        self.attribution = 0.5

        # Energy budget
        self.energy_budget = 1.0
        self.energy_spent = 0.0
        self.energy_window = 500
        self.last_reset_step = 0

        # Arbitration mode
        self.mode = "normal"
        self.mode_changed_at = 0
        self.mode_hold = 400

        # Pending dynamics evaluation
        self._pending_dynamics = {}

        # ── v8: Trajectory Predictor ──
        self.traj_fit = None
        self.traj_last_fit = 0
        self.traj_horizons = [1000, 5000, 10000]
        self.traj_predictions = {}
        self.traj_convergence_step = None
        self._traj_running = {}
        # Regime tracking (NEW)
        self._traj_regime_history = deque(maxlen=200)
        self._traj_short_running = {}  # short-window (faster decay)
        self._traj_score = 1.0  # trajectory quality score (0-1)

        # ── v8: Model Diagnosis ──
        self.diagnosis = "unknown"
        self.diagnosis_reasons = []
        self.diagnosis_last = 0
        self.model_needs = []

        # ── v8: Strategic Plan ──
        self.plan = {}
        self.plan_last_update = 0
        self.plan_last_applied_step = -1
        self.plan_accuracy = 0.5
        self.plan_history = deque(maxlen=self.PLANHIST_MAXLEN)
        self.plan_cache = {}

        # ── v8: Goal Tracker ──
        self.goals = {
            "milestones": [],
            "target_loss": None,
            "progress_velocity": 0.0,
            "velocity_ema": 0.0,
            "last_milestone_check": 0,
            "eta_to_target": None,
        }
        self._init_goals(lr, warmup, total)

        # ── v9: New Algorithms ──
        self.acc_mult = 1.0
        self.roughness_ema = 0.0
        self.roughness_init = False
        self._per_group_grad_norms = {}
        self._per_group_update_ratios = {}
        self._scheduled_mom = 0.9
        self._plateau_risk = False
        self._cycle_step = 0
        self._cycle_period = total // 4 if total > 0 else 5000
        self._cycle_amplitude = 0.05
        self._suggested_batch_mult = 1.0
        self._adaptive_alpha = 0.05

        # ── Dataset awareness ──
        self.dataset_info = {}  # category → {count, loss_ema, trend}
        self._cat_loss_ema = {}
        self._cat_loss_init = {}
        self._dataset_mode = "auto"  # auto | manual | explore
        self._dataset_weights = {}
        self._last_dataset_suggest = 0

        # ── v10: Mathematical Precision Signals ──

        # Gradient autocorrelation at multiple lags
        self._grad_flat_cache = deque(maxlen=max(self.AUTO_LAGS) + 1)  # flat grad vecs
        self._grad_autocorr = {lag: 0.0 for lag in self.AUTO_LAGS}
        self._grad_autocorr_init = {lag: False for lag in self.AUTO_LAGS}

        # Gradient higher moments (skewness, kurtosis)
        self._grad_skew_ema = 0.0
        self._grad_kurt_ema = 0.0
        self._grad_moments_init = False

        # Loss autocorrelation (ρ₁, ρ₂)
        self._loss_prev = None
        self._loss_lag1_corr = 0.0
        self._loss_lag2_corr = 0.0
        self._loss_corr_init = False

        # Hutchinson Hessian trace estimation
        self._hessian_trace_ema = 0.0
        self._hessian_trace_init = False
        self._hessian_step = 0
        self._hessian_noise_ratio = 0.0  # trace / grad_norm²

        # Lipschitz LR bound
        self._lip_gn_prev = None
        self._lip_bound = float("inf")
        self._lip_init = False
        self._lip_alpha = 0.1

        # Effective gradient rank (via spectral ratio)
        self._grad_spectral_ratio = 1.0  # λ₁ / Σλᵢ
        self._spectral_init = False

        # Kalman filter for loss
        self._kalman_x = 0.0    # estimated state
        self._kalman_P = 1.0    # estimate covariance
        self._kalman_Q = 1e-4   # process noise
        self._kalman_R = 1e-2   # measurement noise
        self._kalman_init = False
        self._kalman_gain = 0.0
        self._kalman_innov = 0.0

        # Loss change-point detection (CUSUM)
        self._cusum_S = 0.0
        self._cusum_thr = 3.0
        self._cusum_mean = 0.0
        self._cusum_n = 0

    # ── Gradient capture (call before optimizer.step) ──
    def capture_gradients(self, model, step=None):
        flat = []
        comp_stats = {}
        layer_norms = {}
        for name, p in model.named_parameters():
            if p.grad is None:
                continue
            flat.append(p.grad.detach().view(-1))
            parts = name.split(".")
            group = parts[0] if len(parts) == 1 else ".".join(parts[:2])
            if group not in layer_norms:
                layer_norms[group] = []
            layer_norms[group].append(p.grad.detach())
            comp = ".".join(parts[:2]) if len(parts) >= 2 else parts[0]
            if comp not in comp_stats:
                comp_stats[comp] = []
            g = p.grad.abs().detach().view(-1)
            n = min(g.numel(), 100)
            if g.numel() > 200:
                idx = torch.randint(0, g.numel(), (n,), device=g.device)
                comp_stats[comp].extend(g[idx].tolist())
            else:
                comp_stats[comp].extend(g[:n].tolist())
        if not flat:
            return
        cur = torch.cat(flat)
        if self.flat_grad is not None and self.flat_grad.numel() == cur.numel():
            cs = torch.nn.functional.cosine_similarity(
                cur.unsqueeze(0), self.flat_grad.unsqueeze(0)
            ).item()
            if not self.cos_sim_init:
                self.cos_sim_ema = cs
                self.cos_sim_init = True
            else:
                self.cos_sim_ema = 0.1 * cs + 0.9 * self.cos_sim_ema
        self.flat_grad = cur.detach()
        for group, grads in layer_norms.items():
            norms = torch.stack([g.norm() for g in grads])
            avg_norm = norms.mean().item()
            self._check_layer_health(group, avg_norm)
        self.sick_layers, self.layer_ratios = self._report_sick_layers()
        ema_decay = 0.9
        for comp, vals in comp_stats.items():
            if len(vals) >= 10:
                s = sorted(vals)
                fresh = {
                    "p5": s[int(len(s) * 0.05)],
                    "p50": s[len(s) // 2],
                    "p95": s[int(len(s) * 0.95)],
                    "sparsity": sum(1 for v in vals if abs(v) < 1e-8) / len(vals),
                }
                if comp in self.grad_hist_ema:
                    old = self.grad_hist_ema[comp]
                    self.grad_hist_ema[comp] = {
                        k: ema_decay * old[k] + (1 - ema_decay) * fresh[k]
                        for k in fresh
                    }
                else:
                    self.grad_hist_ema[comp] = fresh
        self.grad_hist = self.grad_hist_ema
        for _fn in (self._update_gradient_autocorr, self._update_gradient_moments, self._update_spectral_ratio):
            try:
                _fn(model)
            except Exception:
                pass

    # ── Update capture (call after optimizer.step, before zero_grad) ──
    def capture_update(self, model):
        cur = torch.cat([p.data.detach().view(-1) for p in model.parameters()])
        if self.prev_params is not None and self.prev_params.numel() == cur.numel():
            delta = cur - self.prev_params
            upd_norm = delta.norm().item()
            param_norm = self.prev_params.norm().item()
            if param_norm > 0:
                ur = upd_norm / param_norm
                if not self.ur_init:
                    self.update_ratio_ema = ur
                    self.ur_init = True
                else:
                    ad_alpha = min(0.3, max(0.02, ur * 5))
                    self.ur_alpha = 0.9 * self.ur_alpha + 0.1 * ad_alpha
                    self.update_ratio_ema = self.ur_alpha * ur + (1 - self.ur_alpha) * self.update_ratio_ema
        self.prev_params = cur.clone()

    # ── Record (call after step) ──
    def record(self, step, loss, gn):
        if math.isnan(loss) or math.isinf(loss) or math.isnan(gn) or math.isinf(gn):
            self.anomalies += 1
            if self._prev_loss is not None:
                self.gn_gap_steps += 1
            return
        self.gn_gap_steps = 0
        self.hist.append(loss)
        # Multi-timescale EMAs
        if not self.short_init:
            self.ema_short = loss
            self.short_init = True
        else:
            self.ema_short = 0.1 * loss + 0.9 * self.ema_short
        if not self.med_init:
            self.ema_med = loss
            self.med_init = True
        else:
            self.ema_med = 0.02 * loss + 0.98 * self.ema_med
        if not self.long_init:
            self.ema_long = loss
            self.long_init = True
        else:
            self.ema_long = 0.005 * loss + 0.995 * self.ema_long
        # Gradient noise scale
        if self.gn_var_init:
            self.gn_var = 0.1 * (gn - self.gn_ema) ** 2 + 0.9 * self.gn_var
        self.gn_hist.append(gn)
        if not self.gn_var_init:
            self.gn_ema = gn
            self.gn_var_init = True
        else:
            self.gn_ema = 0.1 * gn + 0.9 * self.gn_ema
        if self.gn_var_init and self.gn_ema > 0:
            noise = self.gn_var / max(self.gn_ema ** 2, 1e-12)
            if not self.gn_noise_init:
                self.gn_noise_ema = noise
                self.gn_noise_init = True
            else:
                self.gn_noise_ema = self.gn_noise_alpha * noise + (1 - self.gn_noise_alpha) * self.gn_noise_ema
        # Best / plateau
        if loss < self.best:
            self.best = loss
            self.best_step = step
            self.plateau = 0
            self.stuck_steps = 0
            self.init_best = False
        else:
            self.plateau += 1
            self.stuck_steps += 1
        self._prev_loss = loss
        # Phase detection
        self._update_phase(step)
        # Confidence
        self._update_confidence(step)
        # Global training potential
        if loss < self.potential or not self.pot_init:
            self.potential = loss
            self.pot_init = True
        else:
            self.potential = self.pot_alpha * loss + (1 - self.pot_alpha) * self.potential
        self.dev_potential = loss - self.potential
        # Mom decay
        if self.state != "oscillating":
            self.mom_boost *= self.mom_decay
            if abs(self.mom_boost) < 0.001:
                self.mom_boost = 0.0
        # Evaluate past adjustment
        if self.last_adj_loss is not None:
            if step - self._last_adj_step >= self._eval_cd:
                worked = loss < self.last_adj_loss * 0.98
                self.adj_memory.append({
                    "state": self._last_adj_state, "worked": worked,
                    "step": self._last_adj_step,
                })
                self.last_adj_loss = None
        # Update running trajectory fit (dual-window)
        self._update_running_fit(step, loss, decay=self.TRAJ_DECAY, store="_traj_running")
        self._update_running_fit(step, loss, decay=0.990, store="_traj_short_running")
        # Evaluate pending dynamics
        if self._pending_dynamics and step - self._pending_dynamics.get("step", 0) >= self._pending_dynamics["eval_cd"]:
            pd = self._pending_dynamics
            self._update_dynamics(pd["loss_before"], self.ema_short, pd["d_lr"], pd["d_wd"])
            self._pending_dynamics = {}

        # ── v10: Mathematical precision updates ──
        for _fn in (self._update_loss_autocorr, self._kalman_filter_step, self._update_cusum, self._estimate_lipschitz_bound):
            try:
                _fn(loss)
            except Exception:
                pass

    def _update_phase(self, step):
        if step < self.warmup:
            p = "warmup"
        elif step < self.total * 0.25:
            p = "exploration"
        elif step < self.total * 0.60:
            p = "balanced"
        else:
            p = "exploitation"
        if p != self.phase:
            self.phase = p
            if p == "exploration":
                self.eff_floor = 0.05
                self.eff_ceil = 2.0
                self.cooldown = 500
            elif p == "balanced":
                self.eff_floor = 0.15
                self.eff_ceil = 1.5
                self.cooldown = 750
            else:
                self.eff_floor = 0.4
                self.eff_ceil = 1.2
                self.cooldown = 1000
            self.lr_mult = max(self.eff_floor, min(self.eff_ceil, self.lr_mult))
            self.plateau_thr = max(200, self.total // 50)

    # ── Eval capture ──
    def capture_eval(self, step, eval_loss):
        self.eval_losses.append(eval_loss)
        self.last_eval_step = step
        if len(self.eval_losses) >= 3 and self.short_init:
            gap = eval_loss - self.ema_short
            if not self.gap_init:
                self.gap_ema = gap
                self.gap_init = True
            else:
                self.gap_ema = 0.1 * gap + 0.9 * self.gap_ema
            self.overfit_signal = self.gap_ema > max(0.3, 0.05 * eval_loss)

    # ── Layer health ──
    def _check_layer_health(self, name, gn):
        parts = name.split(".")
        group = parts[0] if len(parts) == 1 else ".".join(parts[:2])
        if group not in self.layer_gn:
            self.layer_gn[group] = deque(maxlen=self.LAYERGN_MAXLEN)
        self.layer_gn[group].append(gn)

    def _report_sick_layers(self):
        sick = set()
        ratios = {}
        if len(self.layer_gn) < 3:
            return sick, ratios
        meds = {}
        for g, vals in self.layer_gn.items():
            if len(vals) >= 5:
                meds[g] = sorted(vals)[len(vals) // 2]
        if not meds:
            return sick, ratios
        median_of_all = sorted(meds.values())[len(meds) // 2]
        for g, m in meds.items():
            r = m / max(median_of_all, 1e-12)
            ratios[g] = r
            if r > 4.0 or r < 0.1:
                sick.add(g)
        return sick, ratios

    # ── Confidence ──
    def _update_confidence(self, step):
        if step <= self.warmup:
            self.confidence = 0.0
            self.conf_init = False
            return
        steps_after = step - self.warmup
        time_factor = min(1.0, steps_after / self.CONFIDENCE_WARMUP)
        stability = 1.0
        if self.cos_sim_init:
            stability = 0.5 + 0.5 * max(0.0, self.cos_sim_ema)
        has_data = 0.5
        if self.gn_noise_init and self.ur_init:
            has_data = 1.0
        self.confidence = time_factor * stability * has_data
        self.conf_init = True

    # ── Bayesian dynamics model (Welford's online algorithm) ──
    def _update_dynamics(self, loss_before, loss_after, d_lr, d_wd):
        d = self.dynamics["global"]
        if abs(d_lr) > 1e-12:
            sens = (loss_after - loss_before) / d_lr
            d["n"] += 1
            n = d["n"]
            delta = sens - d["lr_sens_mean"]
            d["lr_sens_mean"] += delta / n
            d["lr_sens_m2"] += delta * (sens - d["lr_sens_mean"])
        if abs(d_wd) > 1e-12:
            sens_wd = (loss_after - loss_before) / d_wd
            if d["n"] == 0:
                d["n"] = 1
            n = d["n"]
            delta = sens_wd - d["wd_sens_mean"]
            d["wd_sens_mean"] += delta / n
            d["wd_sens_m2"] += delta * (sens_wd - d["wd_sens_mean"])

    def _dynamics_std(self, key_mean, key_m2):
        d = self.dynamics["global"]
        n = max(d["n"], 1)
        if n < 3:
            return None
        var = d.get(key_m2, 0.0) / (n - 1)
        return math.sqrt(max(var, 0.0))

    def _predict_improvement(self, d_lr, d_wd=0.0, return_ci=False):
        d = self.dynamics["global"]
        if d["n"] < 3:
            return None
        mean = d.get("lr_sens_mean", 0.0) * d_lr
        var = 0.0
        if d["n"] >= 3:
            lr_var = d.get("lr_sens_m2", 0.0) / (d["n"] - 1)
            var += lr_var * d_lr ** 2
        if abs(d_wd) > 1e-12:
            mean += d.get("wd_sens_mean", 0.0) * d_wd
            if d["n"] >= 3:
                wd_var = d.get("wd_sens_m2", 0.0) / (d["n"] - 1)
                var += wd_var * d_wd ** 2
        std = math.sqrt(max(var, 0.0)) if d["n"] >= 3 else 0.0
        if return_ci:
            return mean, std
        return mean

    # ── Formal energy cost ──
    def _energy_cost(self, new_lr, new_gn, new_wd):
        cost = abs(math.log(max(new_lr / max(self.lr_mult, 1e-12), 0.01)))
        cost += abs(math.log(max(new_gn / max(self.gn_mult, 1e-12), 0.1))) * 0.5
        cost += abs(math.log(max(new_wd / max(self.wd_mult, 1e-12), 0.1))) * 0.3
        return cost

    def _project_action(self, adj, pre_lr=None, pre_gn=None, pre_wd=None):
        if self.energy_spent >= self.energy_budget:
            adj.clear()
            return
        pl = adj.get("lr_mult", self.lr_mult)
        pg = adj.get("gn_mult", self.gn_mult)
        pw = adj.get("wd_mult", self.wd_mult)
        base_lr = pre_lr if pre_lr is not None else self.lr_mult
        base_gn = pre_gn if pre_gn is not None else self.gn_mult
        base_wd = pre_wd if pre_wd is not None else self.wd_mult
        cost = self._energy_cost(pl, pg, pw)
        remaining = self.energy_budget - self.energy_spent
        if cost <= remaining:
            self.energy_spent += cost
            return
        scale = remaining / max(cost, 1e-12)
        if "lr_mult" in adj:
            adj["lr_mult"] = base_lr + (pl - base_lr) * scale
        if "gn_mult" in adj:
            adj["gn_mult"] = base_gn + (pg - base_gn) * scale
        if "wd_mult" in adj:
            adj["wd_mult"] = base_wd + (pw - base_wd) * scale
        self.energy_spent += remaining

    # ── Goal initialization ──
    def _init_goals(self, lr, warmup, total):
        self.goals["milestones"] = [
            (15.0, None), (12.0, None), (10.0, None),
            (8.0, None), (6.0, None), (4.0, None), (3.0, None),
        ]
        self.goals["target_loss"] = 3.0
        self.goals["last_milestone_check"] = 0

    # ── Dual-window trajectory fitting ──
    def _update_running_fit(self, step, loss, decay=0.995, store="_traj_running"):
        tr = self.__dict__.setdefault(store, {})
        log_x = math.log(max(step, 1))
        log_y = math.log(max(loss, 1e-6))
        for key, val in [("sx", log_x), ("sy", log_y), ("sxx", log_x ** 2), ("syy", log_y ** 2), ("sxy", log_x * log_y)]:
            old = tr.get(key, 0.0)
            tr[key] = old * decay + val
        tr["n_eff"] = tr.get("n_eff", 0) * decay + 1.0

    def _fit_trajectory(self, step, store="_traj_running"):
        tr = self.__dict__.get(store, {})
        n = tr.get("n_eff", 0)
        if n < 20:
            return None
        sx, sy = tr.get("sx", 0), tr.get("sy", 0)
        sxx, syy, sxy = tr.get("sxx", 0), tr.get("syy", 0), tr.get("sxy", 0)
        x_mean = sx / n
        y_mean = sy / n
        den = sxx - n * x_mean * x_mean
        if den < 1e-12:
            return None
        a = (sxy - n * x_mean * y_mean) / den
        b = y_mean - a * x_mean
        ss_res = syy - 2 * a * sxy - 2 * b * sy + a ** 2 * sxx + 2 * a * b * sx + n * b ** 2
        ss_tot = syy - n * y_mean * y_mean
        r2 = max(0.0, min(1.0, 1.0 - ss_res / max(ss_tot, 1e-12)))
        return {"slope": a, "intercept": b, "r2": r2}

    def _update_traj_score(self, step):
        short = self._fit_trajectory(step, store="_traj_short_running")
        long_ = self._fit_trajectory(step, store="_traj_running")
        if short and long_ and long_["r2"] > 0.3:
            r2_gap = abs(short["r2"] - long_["r2"])
            slope_diff = abs(short["slope"] - long_["slope"])
            regime_shift = 1.0 - min(1.0, (r2_gap * 2 + slope_diff * 50) / 3)
            self._traj_score = 0.9 * self._traj_score + 0.1 * regime_shift
        self._traj_regime_history.append({
            "step": step,
            "short_slope": short["slope"] if short else None,
            "long_slope": long_["slope"] if long_ else None,
        })

    def _fit_and_predict(self, step):
        fit = self._fit_trajectory(step)
        if fit is None:
            self.traj_fit = None
            self.traj_predictions = {}
            self.traj_convergence_step = None
            return
        self.traj_fit = fit
        self.traj_last_fit = step
        # Use long-window fit for prediction
        for horizon in self.traj_horizons:
            future_step = step + horizon
            if future_step > 0:
                pred_log = fit["slope"] * math.log(future_step) + fit["intercept"]
                self.traj_predictions[horizon] = math.exp(min(pred_log, 709))
        target = self.goals.get("target_loss", 3.0)
        if fit["slope"] < 0 and target > 0:
            log_target = math.log(target)
            log_conv = (log_target - fit["intercept"]) / fit["slope"]
            if 0 < log_conv < 700:
                self.traj_convergence_step = int(math.exp(log_conv))
            else:
                self.traj_convergence_step = None
        else:
            self.traj_convergence_step = None

    # ── v8: Model diagnosis ──
    def _diagnose_model(self, step):
        if len(self.hist) < 30:
            self.diagnosis = "initializing"
            self.diagnosis_reasons = ["not enough data"]
            self.model_needs = ["wait for warmup"]
            return
        reasons = []
        needs = []
        diagnosis = "unknown"
        loss_now = self.ema_short if self.short_init else 0
        loss_med = self.ema_med if self.med_init else loss_now
        loss_long = self.ema_long if self.long_init else loss_now
        trend_short = (loss_now - loss_med) / max(loss_med, 1e-8)
        trend_long = (loss_med - loss_long) / max(loss_long, 1e-8)
        if self.state == "diverging" or (loss_now > loss_long * 1.2 and loss_now > 15):
            diagnosis = "diverging"
            reasons.append("loss increasing rapidly")
            needs.append("immediate LR reduction")
            needs.append("increase gradient clipping")
        elif self.overfit_signal:
            diagnosis = "overfitting"
            reasons.append(f"train-eval gap: {self.gap_ema:.3f}")
            needs.append("increase weight decay")
            needs.append("reduce learning rate slightly")
            if self.gap_ema > 1.0:
                needs.append("consider early stopping or regularization")
        elif (trend_long < -0.001 and trend_short < 0 and
              self.gn_noise_init and self.gn_noise_ema < 0.5 and
              self.cos_sim_init and self.cos_sim_ema > 0.3):
            diagnosis = "converging"
            reasons.append("smooth loss decrease, low noise")
            needs.append("maintain current trajectory")
            needs.append("gradually reduce LR for fine-tuning")
        elif self.state == "plateau" or self.plateau > self.plateau_thr * 0.5:
            diagnosis = "plateauing"
            reasons.append(f"no improvement for {self.plateau} steps")
            if self.ur_init and self.update_ratio_ema < 0.0001:
                reasons.append("parameters barely moving")
                needs.append("LR spike to escape local minimum")
                needs.append("increase momentum temporarily")
            else:
                reasons.append("learning stalled")
                needs.append("increase LR for exploration")
                needs.append("try different data mix")
        elif self.state == "oscillating":
            diagnosis = "oscillating"
            reasons.append("loss bouncing up and down")
            needs.append("reduce LR")
            needs.append("increase momentum damping")
        elif self.stuck_steps > self.revive_thr * 0.5:
            diagnosis = "stuck"
            reasons.append(f"stuck for {self.stuck_steps} steps")
            needs.append("major LR adjustment or revive")
            if self.revive_attempts < self.max_revives:
                needs.append("consider revive strategy")
        elif trend_long < -0.002 and trend_short < 0:
            diagnosis = "learning_well"
            reasons.append("steady improvement across timescales")
            needs.append("continue current strategy")
            if self.lr_mult < 1.0:
                needs.append("consider increasing LR to accelerate")
        elif trend_long < 0 and trend_long > -0.002:
            diagnosis = "learning_slow"
            reasons.append("improving but slowly")
            if self.lr_mult < 0.8:
                needs.append("LR may be too conservative, try increasing")
            elif self.lr_mult > 1.2:
                needs.append("LR may be too aggressive, try reducing slightly")
            else:
                needs.append("model may need more data diversity")
        elif abs(trend_short) < 0.01 and abs(trend_long) < 0.01:
            if self.gn_noise_init and self.gn_noise_ema > 1.0:
                diagnosis = "wandering"
                reasons.append("flat loss with high gradient noise")
                needs.append("reduce LR to stabilize")
                needs.append("increase batch size or gradient accumulation")
            else:
                diagnosis = "stable"
                reasons.append("loss is stable")
                needs.append("maintain or slightly increase LR")
        if self.sick_layers:
            reasons.append(f"sick layers: {self.sick_layers}")
            needs.append("reduce LR for affected layers")
        if self.conf_init and self.confidence < 0.3:
            reasons.append("low confidence in signals")
            needs.append("wait for more data before major changes")
        if self.traj_fit and self.traj_fit["r2"] > 0.7:
            if self.traj_fit["slope"] > 0:
                reasons.append("trajectory predicts divergence")
                needs.append("proactive LR reduction")
            elif self.traj_fit["slope"] > -0.1:
                reasons.append("trajectory predicts slow convergence")
                needs.append("may need LR boost to reach target")
        # ── v10: Mathematical diagnosis enhancements ──
        if self._grad_moments_init:
            if self._grad_kurt_ema > 3.0:
                reasons.append(f"heavy-tailed gradients (kurtosis={self._grad_kurt_ema:.1f})")
                if diagnosis in ("learning_well", "learning_slow", "stable"):
                    needs.append("reduce LR slightly for gradient stability")
            elif self._grad_kurt_ema < -1.0:
                reasons.append(f"abnormally light-tailed gradients")
                needs.append("check for gradient shrinkage")
            if abs(self._grad_skew_ema) > 2.0:
                reasons.append(f"skewed gradient distribution ({self._grad_skew_ema:.1f})")
                needs.append("consider gradient whitening")
        if self._grad_autocorr_init[1]:
            if all(self._grad_autocorr.get(lag, 0) < 0.1 for lag in self.AUTO_LAGS):
                reasons.append("gradient directions decorrelated (high noise)")
                if diagnosis in ("stable",):
                    needs.append("reduce LR to combat gradient noise")
            elif self._grad_autocorr.get(1, 0) > 0.8 and self._grad_autocorr.get(5, 0) > 0.5:
                reasons.append("gradient directions highly persistent")
                if self.lr_mult < 1.0:
                    needs.append("can safely increase LR")
            lag1 = self._grad_autocorr.get(1, 0)
            lag3 = self._grad_autocorr.get(3, 0)
            if lag1 > 0.3 and lag3 < -0.1:
                reasons.append("alternating gradient directions (periodic)")
                if diagnosis != "oscillating":
                    needs.append("add momentum damping")
        if self._loss_corr_init:
            if self._loss_lag1_corr < 0:
                reasons.append("negative loss autocorrelation (oscillation)")
                if diagnosis not in ("oscillating", "diverging"):
                    needs.append("reduce LR to dampen oscillation")
        if self._hessian_trace_init and self.gn_var_init:
            nr = self._hessian_noise_ratio
            if nr > 10.0:
                reasons.append(f"high Hessian noise ratio ({nr:.1f})")
                needs.append("reduce LR (curvature >> gradient scale)")
            elif nr < 0.1:
                reasons.append("low Hessian curvature")
                if diagnosis in ("plateauing", "stuck"):
                    needs.append("LR spike less effective (low curvature)")
        if self._lip_init and self._lip_bound < float("inf"):
            max_safe_lr = self._lip_bound
            current_lr = self.base_lr * self.lr_mult
            if current_lr > max_safe_lr * 0.5:
                reasons.append(f"LR near Lipschitz bound ({current_lr:.2e} / {max_safe_lr:.2e})")
                if diagnosis in ("oscillating", "diverging"):
                    needs.append("reduce LR below Lipschitz threshold")
        if self._spectral_init:
            if self._grad_spectral_ratio > 0.5:
                reasons.append(f"gradients low-rank (spectral ratio={self._grad_spectral_ratio:.2f})")
                if diagnosis in ("plateauing", "stuck"):
                    needs.append("increase stochasticity (higher LR or batch noise)")
        if self._cusum_n > 20 and self._cusum_S > self._cusum_thr:
            reasons.append(f"CUSUM change detected (S={self._cusum_S:.1f})")
            if diagnosis not in ("diverging", "oscillating"):
                needs.append("consider LR regime change")
        if self._kalman_init and self.short_init:
            kalman_gap = self._kalman_x - self.ema_short
            if abs(kalman_gap) > 0.1 * self.ema_short:
                reasons.append(f"Kalman-EMA divergence ({kalman_gap:.3f})")
                if diagnosis in ("stable",):
                    needs.append("verify loss signal quality")
        self.diagnosis = diagnosis
        self.diagnosis_reasons = reasons
        self.model_needs = needs
        self.diagnosis_last = step

    # ── v8: Strategic plan creation ──
    def _create_plan(self, step):
        diag = self.diagnosis
        phase = self.phase
        cache_key = (diag, phase)
        if cache_key in self.plan_cache:
            cached = self.plan_cache[cache_key]
            if cached.get("age", 0) < 2000:
                self.plan = dict(cached)
                self.plan["age"] = self.plan.get("age", 0) + (step - self.plan_last_update)
                self.plan_last_update = step
                return
        plan = {
            "created_at": step,
            "created_loss": self.ema_short if self.short_init else 0,
            "diagnosis": diag,
            "phase": phase,
            "horizon": 1000,
            "strategy": "maintain",
            "actions": [],
            "target_lr_mult": self.lr_mult,
            "target_gn_mult": self.gn_mult,
            "target_wd_mult": self.wd_mult,
            "confidence": 0.5,
            "age": 0,
        }
        if diag == "diverging":
            plan["strategy"] = "emergency_stabilize"
            plan["horizon"] = 500
            plan["actions"] = [
                {"step": 0, "action": "lr_mult", "value": max(self.eff_floor, self.lr_mult * 0.3)},
                {"step": 0, "action": "gn_mult", "value": 0.3},
                {"step": 200, "action": "lr_mult", "value": max(self.eff_floor, self.lr_mult * 0.5)},
                {"step": 400, "action": "lr_mult", "value": max(self.eff_floor, self.lr_mult * 0.7)},
            ]
            plan["target_lr_mult"] = max(self.eff_floor, self.lr_mult * 0.7)
            plan["confidence"] = 0.9
        elif diag == "overfitting":
            plan["strategy"] = "regularize"
            plan["horizon"] = 1000
            plan["actions"] = [
                {"step": 0, "action": "wd_mult", "value": min(self.wd_ceil, self.wd_mult * 1.2)},
                {"step": 0, "action": "lr_mult", "value": max(self.eff_floor, self.lr_mult * 0.9)},
                {"step": 500, "action": "wd_mult", "value": min(self.wd_ceil, self.wd_mult * 1.1)},
            ]
            plan["target_wd_mult"] = min(self.wd_ceil, self.wd_mult * 1.2)
            plan["confidence"] = 0.7
        elif diag == "plateauing":
            plan["strategy"] = "explore_escape"
            plan["horizon"] = 1500
            plan["actions"] = [
                {"step": 0, "action": "lr_mult", "value": min(self.eff_ceil, self.lr_mult * 1.5)},
                {"step": 0, "action": "mom", "value": 0.05},
                {"step": 300, "action": "lr_mult", "value": min(self.eff_ceil, self.lr_mult * 1.3)},
                {"step": 600, "action": "lr_mult", "value": self.lr_mult},
                {"step": 1000, "action": "lr_mult", "value": max(self.eff_floor, self.lr_mult * 0.85)},
            ]
            plan["target_lr_mult"] = max(self.eff_floor, self.lr_mult * 0.85)
            plan["confidence"] = 0.6
        elif diag == "converging":
            plan["strategy"] = "fine_tune"
            plan["horizon"] = 5000
            plan["actions"] = [
                {"step": 0, "action": "lr_mult", "value": max(self.eff_floor, self.lr_mult * 0.95)},
                {"step": 2000, "action": "lr_mult", "value": max(self.eff_floor, self.lr_mult * 0.85)},
                {"step": 4000, "action": "lr_mult", "value": max(self.eff_floor, self.lr_mult * 0.7)},
            ]
            plan["target_lr_mult"] = max(self.eff_floor, self.lr_mult * 0.7)
            plan["confidence"] = 0.8
        elif diag == "oscillating":
            plan["strategy"] = "dampen"
            plan["horizon"] = 800
            plan["actions"] = [
                {"step": 0, "action": "lr_mult", "value": max(self.eff_floor, self.lr_mult * 0.7)},
                {"step": 0, "action": "mom", "value": 0.0},
                {"step": 400, "action": "lr_mult", "value": max(self.eff_floor, self.lr_mult * 0.85)},
            ]
            plan["target_lr_mult"] = max(self.eff_floor, self.lr_mult * 0.85)
            plan["confidence"] = 0.7
        elif diag == "stuck":
            plan["strategy"] = "revive_and_reset"
            plan["horizon"] = 2000
            plan["actions"] = [
                {"step": 0, "action": "lr_mult", "value": min(self.eff_ceil, self.lr_mult * 2.0)},
                {"step": 0, "action": "mom", "value": -0.2},
                {"step": 500, "action": "lr_mult", "value": self.lr_mult},
                {"step": 1000, "action": "lr_mult", "value": max(self.eff_floor, self.lr_mult * 0.8)},
            ]
            plan["target_lr_mult"] = max(self.eff_floor, self.lr_mult * 0.8)
            plan["confidence"] = 0.5
        elif diag == "learning_well":
            plan["strategy"] = "accelerate"
            plan["horizon"] = 3000
            if self.lr_mult < 1.0:
                plan["actions"] = [
                    {"step": 0, "action": "lr_mult", "value": min(self.eff_ceil, self.lr_mult * 1.1)},
                    {"step": 1000, "action": "lr_mult", "value": min(self.eff_ceil, self.lr_mult * 1.15)},
                    {"step": 2000, "action": "lr_mult", "value": min(1.0, self.lr_mult * 1.2)},
                ]
                plan["target_lr_mult"] = min(1.0, self.lr_mult * 1.2)
            else:
                plan["actions"] = [
                    {"step": 0, "action": "maintain", "value": self.lr_mult},
                ]
                plan["target_lr_mult"] = self.lr_mult
            plan["confidence"] = 0.8
        elif diag == "learning_slow":
            plan["strategy"] = "optimize_pace"
            plan["horizon"] = 2000
            if self.lr_mult < 0.8:
                plan["actions"] = [
                    {"step": 0, "action": "lr_mult", "value": min(1.0, self.lr_mult * 1.15)},
                    {"step": 1000, "action": "lr_mult", "value": min(self.eff_ceil, self.lr_mult * 1.2)},
                ]
                plan["target_lr_mult"] = min(self.eff_ceil, self.lr_mult * 1.2)
            else:
                plan["actions"] = [
                    {"step": 0, "action": "lr_mult", "value": max(self.eff_floor, self.lr_mult * 0.9)},
                ]
                plan["target_lr_mult"] = max(self.eff_floor, self.lr_mult * 0.9)
            plan["confidence"] = 0.6
        elif diag == "wandering":
            plan["strategy"] = "stabilize"
            plan["horizon"] = 1000
            plan["actions"] = [
                {"step": 0, "action": "lr_mult", "value": max(self.eff_floor, self.lr_mult * 0.8)},
                {"step": 0, "action": "gn_mult", "value": max(0.3, self.gn_mult * 0.8)},
                {"step": 500, "action": "lr_mult", "value": max(self.eff_floor, self.lr_mult * 0.9)},
            ]
            plan["target_lr_mult"] = max(self.eff_floor, self.lr_mult * 0.9)
            plan["confidence"] = 0.6
        else:
            plan["strategy"] = "maintain"
            plan["horizon"] = 1000
            plan["actions"] = []
            plan["confidence"] = 0.5
        self.plan = plan
        self.plan_last_update = step
        self.plan_last_applied_step = -1
        if cache_key not in self.plan_cache:
            self.plan_cache[cache_key] = dict(plan)

    # ── Execute plan actions ──
    def _execute_plan(self, step, adj):
        if not self.plan or not self.plan.get("actions"):
            return
        plan_step = step - self.plan.get("created_at", step)
        max_action_step = max((a["step"] for a in self.plan["actions"] if a["step"] <= plan_step), default=-1)
        if max_action_step <= self.plan_last_applied_step:
            return
        for action in self.plan["actions"]:
            if action["step"] <= plan_step:
                act = action["action"]
                val = action["value"]
                if act == "lr_mult":
                    adj["lr_mult"] = val
                    adj["action"] = f"PLAN({self.plan['strategy']}): LR×{val:.2f}"
                elif act == "gn_mult":
                    adj["gn_mult"] = val
                    adj["action"] = f"PLAN({self.plan['strategy']}): clip×{val:.2f}"
                elif act == "wd_mult":
                    adj["wd_mult"] = val
                    adj["action"] = f"PLAN({self.plan['strategy']}): wd×{val:.2f}"
                elif act == "mom":
                    adj["mom"] = val
                    adj["action"] = f"PLAN({self.plan['strategy']}): mom={val:.2f}"
        self.plan_last_applied_step = max_action_step

    # ── Update goals and milestones ──
    def _update_goals(self, step):
        if not self.short_init or step - self.goals["last_milestone_check"] < 100:
            return
        loss_now = self.ema_short
        for i, (target, reached) in enumerate(self.goals["milestones"]):
            if reached is None and loss_now <= target:
                self.goals["milestones"][i] = (target, step)
        if len(self.hist) >= 200:
            h = list(self.hist)
            old_avg = sum(h[-200:-100]) / 100
            new_avg = sum(h[-100:]) / 100
            delta = old_avg - new_avg
            velocity = delta * 10
            if not self.goals["velocity_ema"]:
                self.goals["velocity_ema"] = velocity
            else:
                self.goals["velocity_ema"] = 0.1 * velocity + 0.9 * self.goals["velocity_ema"]
            self.goals["progress_velocity"] = self.goals["velocity_ema"]
        target = self.goals.get("target_loss", 3.0)
        if self.goals["progress_velocity"] > 0 and loss_now > target:
            remaining_loss = loss_now - target
            steps_needed = int(remaining_loss / self.goals["progress_velocity"] * 1000)
            self.goals["eta_to_target"] = step + steps_needed
        self.goals["last_milestone_check"] = step

    # ── Evaluate plan accuracy ──
    def _evaluate_plan(self, step):
        if not self.plan or step - self.plan.get("created_at", step) < 500:
            return
        plan = self.plan
        created_loss = plan.get("created_loss", self.ema_short)
        current_loss = self.ema_short if self.short_init else created_loss
        improved = current_loss < created_loss * 0.98
        self.plan_accuracy = 0.9 * self.plan_accuracy + 0.1 * (1.0 if improved else 0.0)
        self.plan_history.append({
            "strategy": plan.get("strategy", "unknown"),
            "improved": improved,
            "loss_before": created_loss,
            "loss_after": current_loss,
            "age": step - plan.get("created_at", step),
        })
        cache_key = (plan.get("diagnosis", "unknown"), plan.get("phase", "unknown"))
        if cache_key in self.plan_cache:
            cached = self.plan_cache[cache_key]
            if improved:
                cached["successes"] = cached.get("successes", 0) + 1
                cached["age"] = 0
            else:
                cached["failures"] = cached.get("failures", 0) + 1
                cached["age"] = cached.get("age", 0) + 1000

    def _reset_energy(self, step):
        if step - self.last_reset_step >= self.energy_window:
            self.energy_spent = 0.0
            self.last_reset_step = step

    # ── Arbitration mode ──
    def _set_mode(self, mode, step):
        if mode != self.mode:
            self.mode = mode
            self.mode_changed_at = step

    def _mode_done(self, step):
        return step - self.mode_changed_at >= self.mode_hold

    # ── Revive check ──
    def _revive(self, step, adj):
        if self.revive_attempts >= self.max_revives or self.mode != "normal":
            return False
        cost = self._energy_cost(min(self.eff_ceil, self.lr_mult * 2.5), 0.8, self.wd_mult)
        if self.energy_spent + cost > self.energy_budget:
            return False
        self.revive_attempts += 1
        self._set_mode("reviving", step)
        self.energy_spent += cost
        nm = min(self.eff_ceil, max(self.eff_floor, self.lr_mult * 2.5))
        adj["lr_mult"] = nm
        self.lr_mult = nm
        adj["gn_mult"] = 0.8
        self.gn_mult = 0.8
        adj["mom"] = -0.3
        adj["action"] = f"REVIVE: LR×{nm:.2f}, mom-0.3 ✦"
        self.log.append({"step": step, "type": "revive"})
        self.last_change = step
        self.last_gn_change = step
        self.adjustments += 1
        self.stuck_steps = 0
        self.plateau = 0
        return True

    # ── Multi-signal classify ──
    def classify(self, step):
        if len(self.hist) < 50:
            return "init"
        h = list(self.hist)
        if any(math.isnan(l) or math.isinf(l) for l in h[-20:]):
            return "diverging"
        last = h[-20:]
        inc = sum(1 for i in range(1, len(last)) if last[i] > last[i - 1])
        if inc >= 17 and (last[-1] - last[0]) / max(last[0], 1e-8) > 0.5:
            return "diverging"
        avg3 = sum(h[-3:]) / 3
        avg20 = sum(h[-23:-3]) / 20 if len(h) >= 23 else sum(h[:-3]) / max(len(h) - 3, 1)
        if avg3 > avg20 * 1.5 and not self.init_best and avg3 > self.best * 1.3:
            return "spike"
        if len(h) >= 40:
            diffs = [h[i] - h[i - 1] for i in range(len(h) - 40, len(h))]
            sc = sum(1 for i in range(1, len(diffs)) if diffs[i] * diffs[i - 1] < 0)
            if sc / max(len(diffs) - 1, 1) > 0.75 and h[-1] > h[-40] * 0.95:
                return "oscillating"
        if self.med_init and self.long_init:
            med = self.ema_med
            lng = self.ema_long
            short_trend = (self.ema_short - med) / max(med, 1e-8)
            long_trend = (med - lng) / max(lng, 1e-8)
            if long_trend < -0.005 and short_trend < -0.005:
                return "improving"
            if long_trend > 0.005 and short_trend > 0.005:
                return "degrading"
        r100 = h[-100:]
        o100 = h[-200:-100] if len(h) >= 200 else h[:max(len(h) - 100, 1)]
        ar = sum(r100) / len(r100)
        ao = sum(o100) / len(o100)
        cr = (ao - ar) / max(ao, 1e-8)
        if cr > 0.01:
            return "improving"
        if cr < -0.01:
            return "degrading"
        if self.plateau > self.plateau_thr:
            return "plateau"
        return "stable"

    # ── Apply LR ──
    def _apply_lr(self, adj, step, nm):
        max_delta = max(self.eff_floor * 2, abs(self.lr_mult) * 0.5)
        nm = min(max(self.lr_mult - max_delta, nm), self.lr_mult + max_delta)
        nm = max(self.eff_floor, min(self.eff_ceil, nm))
        adj["lr_mult"] = nm
        self.lr_mult = nm
        self.last_change = step
        self.adjustments += 1

    # ── Apply GN ──
    def _apply_gn(self, adj, step, nm):
        max_delta = max(0.1, abs(self.gn_mult) * 0.3)
        nm = max(self.gn_mult - max_delta, min(self.gn_mult + max_delta, nm))
        nm = max(self.GN_FLOOR, min(self.GN_CEIL, nm))
        adj["gn_mult"] = nm
        self.gn_mult = nm
        self.last_gn_change = step

    # ═══════════════════════════════════════════════════════════════
    # NEW: Signal gathering + decision handlers
    # ═══════════════════════════════════════════════════════════════

    def _gather_signals(self, step):
        return {
            "low_cos": self.cos_sim_init and self.cos_sim_ema < 0.2,
            "high_cos": self.cos_sim_init and self.cos_sim_ema > 0.7,
            "high_ur": self.ur_init and self.update_ratio_ema > 0.005,
            "low_ur": self.ur_init and self.update_ratio_ema < 0.0001,
            "high_noise": self.gn_noise_init and self.gn_noise_ema > 2.0,
            "noisy_gap": self.gn_gap_steps > 5,
            "overfit": self.gap_init and self.overfit_signal,
            "sick": len(self.sick_layers) > 0,
            "confident": self.conf_init and self.confidence > 0.6,
            "desperate": self.stuck_steps > self.revive_thr,
            "cd": (step - self.last_change) >= self.cooldown,
            "gn_cd": (step - self.last_gn_change) >= (self.cooldown // 2),
            "orig_lr": self.lr_mult,
            "orig_gn": self.gn_mult,
            "orig_wd": self.wd_mult,
            "cv_gn": math.sqrt(self.gn_var) / max(self.gn_ema, 1e-12) if self.gn_var > 0 and self.gn_ema > 0 else None,
            "traj_diverging": self.traj_fit and self.traj_fit["r2"] > 0.7 and self.traj_fit["slope"] > 0,
            # ── v10: Mathematical signals ──
            "high_kurt": self._grad_moments_init and self._grad_kurt_ema > 2.0,
            "low_kurt": self._grad_moments_init and self._grad_kurt_ema < -0.5,
            "grad_ac_low": self._grad_autocorr_init[1] and all(
                self._grad_autocorr.get(lag, 0) < 0.1 for lag in self.AUTO_LAGS),
            "grad_ac_high": self._grad_autocorr_init[1] and self._grad_autocorr.get(1, 0) > 0.7,
            "grad_ac_neg": self._grad_autocorr_init[1] and self._grad_autocorr.get(1, 0) < -0.1,
            "loss_ac_neg": self._loss_corr_init and self._loss_lag1_corr < 0,
            "hess_high": self._hessian_trace_init and self._hessian_noise_ratio > 5.0,
            "lip_risky": self._lip_init and (self.base_lr * self.lr_mult) > self._lip_bound * 0.5,
            "low_rank": self._spectral_init and self._grad_spectral_ratio > 0.5,
            "cusum_change": self._cusum_n > 20 and self._cusum_S > self._cusum_thr,
            "low_noise": self.gn_noise_init and self.gn_noise_ema < 0.5,
        }

    def _handle_state_transition(self, step, adj, sig):
        """Apply state-based LR/gn adjustments."""
        state = self.state
        cd = sig["cd"]
        if state == "diverging" and cd and self.div_reductions < self.max_red:
            nm = max(self.eff_floor, self.lr_mult * 0.2)
            self._apply_lr(adj, step, nm)
            self._apply_gn(adj, step, 0.3)
            adj["action"] = f"DIVERGE: LR×{nm:.2f}, clip×0.3"
            self.log.append({"step": step, "type": "divergence"})
            self.recovery += 1
            self.div_reductions += 1
            return
        if state == "spike" and cd:
            amt = 0.4 if sig["low_cos"] else 0.5
            nm = max(self.eff_floor, self.lr_mult * amt)
            self._apply_lr(adj, step, nm)
            if sig["high_noise"]:
                self._apply_gn(adj, step, max(0.3, self.gn_mult * 0.8))
            adj["action"] = f"SPIKE: LR×{nm:.2f}"
            self.log.append({"step": step, "type": "spike"})
            return
        if state == "oscillating" and cd:
            nm = max(self.eff_floor, self.lr_mult * 0.75)
            self._apply_lr(adj, step, nm)
            adj["mom"] = 0.08
            adj["action"] = f"OSCILLATE: LR×{nm:.2f}, mom+0.08"
            self.log.append({"step": step, "type": "oscillation"})
            return
        if state == "plateau" and cd:
            if sig["high_ur"] and sig["low_cos"]:
                adj["action"] = "HOLD: plateau ∇ inconsistent"
                self.log.append({"step": step, "type": "hold"})
                return
            nm = min(self.eff_ceil, self.lr_mult * 1.25)
            self._apply_lr(adj, step, nm)
            adj["mom"] = 0.03
            adj["action"] = f"PLATEAU: LR×{nm:.2f}"
            self.log.append({"step": step, "type": "plateau"})
            self.plateau = 0
            self.div_reductions = max(0, self.div_reductions - 1)
            self.deg_reductions = max(0, self.deg_reductions - 1)
            return
        if state == "improving":
            should_boost = sig["high_cos"] and not sig["high_ur"] and not sig["high_noise"] and not sig["noisy_gap"]
            boost = 1.15 if should_boost else 1.08
            if cd and self.lr_mult < 1.0:
                nm = min(1.0, self.lr_mult * boost)
                self._apply_lr(adj, step, nm)
                adj["action"] = f"IMPROVE: LR×{nm:.2f}"
                self.log.append({"step": step, "type": "improving"})
                self.div_reductions = max(0, self.div_reductions - 1)
                self.deg_reductions = max(0, self.deg_reductions - 1)
            elif cd and self.lr_mult >= 1.0 and self.lr_mult < self.eff_ceil:
                if step - self.last_change >= self.cooldown * 2:
                    nm = min(self.eff_ceil, self.lr_mult + 0.05)
                    self._apply_lr(adj, step, nm)
                    adj["action"] = f"STRONG: LR×{nm:.2f}"
                    self.log.append({"step": step, "type": "strong"})
            return
        if state == "degrading" and cd and self.deg_reductions < self.max_red:
            amt = 0.85 if (sig["high_noise"] and not sig["high_ur"]) else 0.70
            nm = max(self.eff_floor, self.lr_mult * amt)
            self._apply_lr(adj, step, nm)
            adj["mom"] = 0.05
            adj["action"] = f"DEGRADE: LR×{nm:.2f}"
            self.log.append({"step": step, "type": "degrading"})
            self.deg_reductions += 1
            return
        if state == "stable" and cd and self.lr_mult < 1.0:
            if sig["high_cos"] or not self.cos_sim_init:
                if step - self.last_change >= self.cooldown * 2:
                    nm = min(1.0, self.lr_mult * 1.05)
                    self._apply_lr(adj, step, nm)
                    adj["action"] = f"STABLE: LR×{nm:.2f}"
                    self.log.append({"step": step, "type": "stable"})
                    self.div_reductions = max(0, self.div_reductions - 1)
                    self.deg_reductions = max(0, self.deg_reductions - 1)

    def _handle_revive(self, step, adj, sig):
        if sig["desperate"] and not adj:
            self._revive(step, adj)

    def _handle_overfit(self, step, adj, sig):
        if sig["overfit"] and sig["cd"]:
            new_wd = min(self.wd_ceil, self.wd_mult * 1.15)
            if new_wd != self.wd_mult:
                self.wd_mult = new_wd
                adj["wd_mult"] = self.wd_mult
                adj["action"] = f"OVERFIT: wd×{self.wd_mult:.2f}"
                self.log.append({"step": step, "type": "overfit"})
                self.last_change = step
        elif not sig["overfit"] and self.gap_init and sig["cd"] and self.wd_mult > 1.0:
            new_wd = max(self.wd_floor, self.wd_mult * 0.92)
            if new_wd != self.wd_mult:
                self.wd_mult = new_wd
                adj["wd_mult"] = self.wd_mult
                adj["action"] = f"EASE_WD: wd×{self.wd_mult:.2f}"
                self.log.append({"step": step, "type": "ease_wd"})
                self.last_change = step

    def _handle_sick_layers(self, step, adj, sig):
        if sig["sick"] and not adj and sig["cd"] and sig["confident"]:
            per_group = {}
            for g, r in self.layer_ratios.items():
                if g in self.sick_layers:
                    if r > 4.0:
                        per_group[g] = max(self.eff_floor, self.lr_mult / (r * 0.25))
                    elif r < 0.1:
                        per_group[g] = max(self.eff_floor, self.lr_mult * r * 10.0)
                    else:
                        per_group[g] = self.lr_mult
                else:
                    per_group[g] = self.lr_mult
            adj["per_group_lr"] = per_group
            nm = max(self.eff_floor, self.lr_mult * 0.92)
            self._apply_lr(adj, step, nm)
            adj["action"] = f"SICK_LAYER: LR×{nm:.2f} ({len(self.sick_layers)} layers)"
            self.log.append({"step": step, "type": "sick_layer"})

    def _handle_grad_clipping(self, step, adj, sig):
        if len(self.gn_hist) >= 50 and sig["gn_cd"]:
            cv = sig["cv_gn"]
            if cv is None:
                return
            if sig["high_noise"] or cv > 0.7:
                self._apply_gn(adj, step, max(0.3, self.gn_mult * 0.85))
                if "action" not in adj:
                    adj["action"] = f"HIGH_VAR: clip×{adj['gn_mult']:.2f}"
            elif cv < 0.25 and not sig["high_noise"]:
                self._apply_gn(adj, step, min(1.0, self.gn_mult * 1.15))
                if "action" not in adj:
                    adj["action"] = f"STABLE_G: clip×{adj['gn_mult']:.2f}"

    def _handle_traj_divergence(self, step, adj, sig):
        """Proactive LR reduction when trajectory predicts divergence."""
        if sig["traj_diverging"] and not adj and sig["confident"] and sig["cd"]:
            nm = max(self.eff_floor, self.lr_mult * 0.75)
            self._apply_lr(adj, step, nm)
            adj["action"] = f"TRAJ_PROACTIVE: LR×{nm:.2f}"
            self.log.append({"step": step, "type": "traj_proactive"})

    def _handle_rollback(self, step, adj, sig):
        if self.rollback_ready and not adj and self.mode == "normal":
            rb_cd = self._eval_cd
            if step - self._last_adj_step >= rb_cd and self._last_adj_state is not None:
                if self.state == "degrading" and self._last_adj_state in ("improving", "stable", "init"):
                    cost = self._energy_cost(self.prev_lr_mult, self.prev_gn_mult, self.prev_wd_mult)
                    if self.energy_spent + cost <= self.energy_budget:
                        self.energy_spent += cost
                        self._set_mode("rolling_back", step)
                        self.lr_mult = self.prev_lr_mult
                        self.gn_mult = self.prev_gn_mult
                        self.wd_mult = self.prev_wd_mult
                        self.rollback_ready = False
                        adj["lr_mult"] = self.lr_mult
                        adj["gn_mult"] = self.gn_mult
                        adj["wd_mult"] = self.wd_mult
                        self.last_change = step
                        adj["action"] = f"ROLLBACK: LR×{self.lr_mult:.2f}"
                        self.log.append({"step": step, "type": "rollback"})

    def _handle_mathematical_signals(self, step, adj, sig):
        if adj or not sig["cd"]:
            return
        if sig["grad_ac_neg"]:
            nm = max(self.eff_floor, self.lr_mult * 0.85)
            self._apply_lr(adj, step, nm)
            adj["mom"] = 0.05
            adj["action"] = f"MATH_AC_NEG: LR×{nm:.2f} mom+0.05"
            self.log.append({"step": step, "type": "math_ac_neg"})
            return
        if sig["lip_risky"] and sig["high_kurt"]:
            nm = max(self.eff_floor, self.lr_mult * 0.8)
            self._apply_lr(adj, step, nm)
            adj["action"] = f"MATH_LIP_KURT: LR×{nm:.2f}"
            self.log.append({"step": step, "type": "math_lip_kurt"})
            return
        if sig["hess_high"] and self.state in ("oscillating", "diverging"):
            nm = max(self.eff_floor, self.lr_mult * 0.6)
            self._apply_lr(adj, step, nm)
            self._apply_gn(adj, step, max(0.3, self.gn_mult * 0.7))
            adj["action"] = f"MATH_HESS: LR×{nm:.2f} clip×{adj.get('gn_mult',0):.2f}"
            self.log.append({"step": step, "type": "math_hess"})
            return
        if sig["grad_ac_low"] and sig["high_noise"] and not sig["low_kurt"]:
            nm = max(self.eff_floor, self.lr_mult * 0.9)
            self._apply_lr(adj, step, nm)
            adj["action"] = f"MATH_NOISE: LR×{nm:.2f}"
            self.log.append({"step": step, "type": "math_noise"})
            return
        if sig["grad_ac_high"] and sig["low_noise"] and self.lr_mult < 1.0:
            nm = min(1.0, max(self.eff_floor, self.lr_mult * 1.1))
            self._apply_lr(adj, step, nm)
            adj["action"] = f"MATH_CONSISTENT: LR×{nm:.2f}"
            self.log.append({"step": step, "type": "math_consistent"})
            return
        if sig["cusum_change"] and self.state == "improving":
            nm = min(self.eff_ceil, self.lr_mult * 1.15)
            self._apply_lr(adj, step, nm)
            adj["action"] = f"MATH_CUSUM: LR×{nm:.2f}"
            self.log.append({"step": step, "type": "math_cusum"})
            return

    # ═══════════════════════════════════════════════════════════════
    # v10: Mathematical Precision Signals
    # ═══════════════════════════════════════════════════════════════

    def _update_gradient_autocorr(self, model):
        grads = [p.grad.detach().view(-1) for p in model.parameters() if p.grad is not None]
        if not grads:
            return
        flat = torch.cat(grads)
        n = flat.numel()
        if n > self.AUTO_CORR_SAMPLE:
            idx = torch.randperm(n, device=flat.device)[:self.AUTO_CORR_SAMPLE]
            flat = flat[idx]
        cn = flat / (flat.norm() + 1e-12)
        self._grad_flat_cache.append(cn)
        c = self._grad_flat_cache
        if len(c) <= max(self.AUTO_LAGS):
            return
        for lag in self.AUTO_LAGS:
            if len(c) > lag:
                prev = c[-(1 + lag)]
                if prev.numel() != cn.numel():
                    continue
                r = (cn @ prev).item()
                r = max(-1.0, min(1.0, r))
                if not self._grad_autocorr_init[lag]:
                    self._grad_autocorr[lag] = r
                    self._grad_autocorr_init[lag] = True
                else:
                    self._grad_autocorr[lag] = 0.05 * r + 0.95 * self._grad_autocorr[lag]

    def _update_gradient_moments(self, model):
        grads = [p.grad.detach().view(-1) for p in model.parameters() if p.grad is not None]
        if not grads:
            return
        all_g = torch.cat(grads)
        n = all_g.numel()
        if n < 10:
            return
        if n > self.AUTO_CORR_SAMPLE:
            idx = torch.randperm(n, device=all_g.device)[:self.AUTO_CORR_SAMPLE]
            all_g = all_g[idx]
        m1 = all_g.mean()
        m2 = ((all_g - m1) ** 2).mean()
        m3 = ((all_g - m1) ** 3).mean()
        m4 = ((all_g - m1) ** 4).mean()
        s2 = m2 + 1e-12
        skew = m3 / (s2 ** 1.5)
        kurt = m4 / (s2 ** 2) - 3.0
        skew = max(-10.0, min(10.0, skew.item()))
        kurt = max(-10.0, min(10.0, kurt.item()))
        if not self._grad_moments_init:
            self._grad_skew_ema = skew
            self._grad_kurt_ema = kurt
            self._grad_moments_init = True
        else:
            a = self.KURT_ALPHA
            self._grad_skew_ema = a * skew + (1 - a) * self._grad_skew_ema
            self._grad_kurt_ema = a * kurt + (1 - a) * self._grad_kurt_ema

    def _update_loss_autocorr(self, loss):
        if self._loss_prev is not None:
            if not self._loss_corr_init:
                self._loss_lag1_corr = loss * self._loss_prev
                self._loss_lag2_corr = self._loss_prev * self._loss_prev
                self._loss_corr_init = True
            else:
                d = 0.05
                self._loss_lag1_corr = (1 - d) * self._loss_lag1_corr + d * loss * self._loss_prev
        if self._loss_prev is not None:
            self._loss_lag2 = self._loss_prev
        self._loss_prev = loss

    def _estimate_hessian_trace(self, model, step):
        if step - self._hessian_step < self.HESS_EVAL_EVERY:
            return
        self._hessian_step = step
        model.eval()
        v = {n: torch.randint(0, 2, p.shape, device=p.device).float() * 2 - 1
             for n, p in model.named_parameters() if p.grad is not None}
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in v:
                    p._hess_v = p.grad.clone() if p.grad is not None else None
        outs = []
        for n, p in model.named_parameters():
            if n in v and p._hess_v is not None:
                out = (v[n] * p._hess_v).sum()
                outs.append(out)
        hv_norm = sum(outs).item() if outs else 0.0
        with torch.no_grad():
            for n, p in model.named_parameters():
                if hasattr(p, '_hess_v'):
                    del p._hess_v
        trace = (v_dot := hv_norm)
        gnorm2 = sum((p.grad.detach() ** 2).sum().item() for p in model.parameters() if p.grad is not None)
        if not self._hessian_trace_init:
            self._hessian_trace_ema = trace
            self._hessian_trace_init = True
        else:
            self._hessian_trace_ema = 0.05 * trace + 0.95 * self._hessian_trace_ema
        self._hessian_noise_ratio = trace / (math.sqrt(gnorm2) + 1e-12) if gnorm2 > 0 else 0.0
        model.train()

    def _estimate_lipschitz_bound(self):
        gn = self.gn_ema if self.gn_var_init else None
        if gn is None:
            return
        if self._lip_gn_prev is not None and self._lip_init:
            delta_gn = abs(gn - self._lip_gn_prev)
            if delta_gn > 1e-12 and gn > 0:
                local_L = delta_gn / (self.base_lr * self.lr_mult + 1e-12)
                bound_i = 2.0 / (local_L + 1e-12)
                if not self._lip_init:
                    self._lip_bound = bound_i
                    self._lip_init = True
                else:
                    self._lip_bound = (1 - self._lip_alpha) * self._lip_bound + self._lip_alpha * bound_i
        self._lip_gn_prev = gn
        self._lip_init = True

    def _kalman_filter_step(self, loss):
        if math.isnan(loss) or math.isinf(loss):
            return
        if not self._kalman_init:
            self._kalman_x = loss
            self._kalman_P = 1.0
            self._kalman_init = True
            return
        self._kalman_P = self._kalman_P + self._kalman_Q
        K = self._kalman_P / (self._kalman_P + self._kalman_R)
        innov = loss - self._kalman_x
        self._kalman_x = self._kalman_x + K * innov
        self._kalman_P = (1 - K) * self._kalman_P
        self._kalman_gain = K
        self._kalman_innov = innov

    def _update_spectral_ratio(self, model):
        grads = [p.grad.detach().view(-1) for p in model.parameters() if p.grad is not None]
        if not grads:
            return
        g = torch.cat(grads)
        n = g.numel()
        if n < 20:
            return
        if n > self.AUTO_CORR_SAMPLE:
            idx = torch.randperm(n, device=g.device)[:self.AUTO_CORR_SAMPLE]
            g = g[idx]
        n = g.numel()
        g_c = g - g.mean()
        g_c = g_c / (g_c.norm() + 1e-12)
        d = max(2, min(100, n // 10))
        idx = torch.randperm(n, device=g.device)[:d]
        sub = g_c[idx]
        cov_approx = sub.unsqueeze(1) @ sub.unsqueeze(0)
        try:
            s = torch.linalg.eigvalsh(cov_approx)
        except RuntimeError:
            return
        l1 = s[-1].item()
        tr = s.sum().item()
        ratio = l1 / (tr + 1e-12) if tr > 0 else 1.0
        if not self._spectral_init:
            self._grad_spectral_ratio = ratio
            self._spectral_init = True
        else:
            self._grad_spectral_ratio = 0.05 * ratio + 0.95 * self._grad_spectral_ratio

    def _update_cusum(self, loss):
        if math.isnan(loss) or math.isinf(loss):
            return
        a = 0.01
        if self._cusum_n == 0:
            self._cusum_mean = loss
            self._cusum_n = 1
            return
        self._cusum_mean = (1 - a) * self._cusum_mean + a * loss
        resid = loss - self._cusum_mean
        self._cusum_S = max(0, self._cusum_S + resid)
        self._cusum_n += 1

    def _compute_mathematical_stats(self, model, step, loss):
        if step % 5 == 0:
            self._update_gradient_autocorr(model)
            self._update_gradient_moments(model)
        if step % 10 == 0:
            self._update_spectral_ratio(model)
        self._update_loss_autocorr(loss)
        self._kalman_filter_step(loss)
        self._update_cusum(loss)
        self._estimate_lipschitz_bound()
        if step % self.HESS_EVAL_EVERY == 0:
            self._estimate_hessian_trace(model, step)

    # ═══════════════════════════════════════════════════════════════
    # REFACTORED: Smart adjustments
    # ═══════════════════════════════════════════════════════════════

    def get_adjustments(self, step):
        adj = {}
        state = self.classify(step)
        self.state = state

        if step % 200 == 0:
            self.history.append({"step": step, "state": state,
                                 "loss_short": f"{self.ema_short:.2f}" if self.short_init else "N/A",
                                 "cos_sim": f"{self.cos_sim_ema:.2f}" if self.cos_sim_init else "N/A",
                                 "update_ratio": f"{self.update_ratio_ema:.6f}" if self.ur_init else "N/A",
                                 "phase": self.phase})

        # ── Early exits ──
        if self.quiet_mode and step < self.quiet_until:
            return adj
        if self.frozen and step < self.frozen_until:
            return adj
        if self.mode != "normal" and not self._mode_done(step):
            return adj
        if self.mode != "normal" and self._mode_done(step):
            self._set_mode("normal", step)

        self._reset_energy(step)
        if step < self.warmup:
            return adj

        signals = self._gather_signals(step)
        orig_lr, orig_gn, orig_wd = signals["orig_lr"], signals["orig_gn"], signals["orig_wd"]

        # ── Decision handlers (first-wins priority) ──
        self._handle_state_transition(step, adj, signals)
        self._handle_revive(step, adj, signals)
        self._handle_overfit(step, adj, signals)
        self._handle_sick_layers(step, adj, signals)
        self._handle_grad_clipping(step, adj, signals)
        self._handle_traj_divergence(step, adj, signals)
        self._handle_mathematical_signals(step, adj, signals)
        self._handle_rollback(step, adj, signals)

        # ── Oscillation dampener ──
        if adj and "lr_mult" in adj:
            delta = adj["lr_mult"] - orig_lr
            sign = 1 if delta > 0 else (-1 if delta < 0 else 0)
            self.adj_signs.append(sign)
            if len(self.adj_signs) >= 4:
                alt = sum(1 for i in range(1, len(self.adj_signs)) if self.adj_signs[i] != self.adj_signs[i - 1])
                if alt >= 3:
                    self.quiet_mode = True
                    self.quiet_until = step + self.cooldown * 2
                    self.adj_signs.clear()
        elif not adj and self.quiet_mode and step > self.quiet_until:
            self.quiet_mode = False

        # ── Energy projection ──
        if adj and "lr_mult" in adj and self.mode == "normal":
            self._project_action(adj, pre_lr=orig_lr, pre_gn=orig_gn, pre_wd=orig_wd)
            if "lr_mult" in adj:
                self.lr_mult = adj["lr_mult"]
            if "gn_mult" in adj:
                self.gn_mult = adj["gn_mult"]
            if "wd_mult" in adj:
                self.wd_mult = adj["wd_mult"]

        # ── v8: Strategic Training Director ──
        self._update_goals(step)
        if step % 500 == 0:
            self._update_traj_score(step)
            self._fit_and_predict(step)
            self._evaluate_plan(step)
        if step % 200 == 0 or self.diagnosis_last == 0 or step - self.diagnosis_last >= 200:
            self._diagnose_model(step)
            if not self.plan or step - self.plan_last_update >= 1000 or self.plan.get("diagnosis") != self.diagnosis:
                self._create_plan(step)

        # Execute plan actions if no existing adjustment
        if not adj:
            self._execute_plan(step, adj)
            if adj and "lr_mult" in adj and self.mode == "normal":
                self._project_action(adj, pre_lr=orig_lr, pre_gn=orig_gn, pre_wd=orig_wd)
            if adj:
                if "lr_mult" in adj:
                    self.lr_mult = adj["lr_mult"]
                if "gn_mult" in adj:
                    self.gn_mult = adj["gn_mult"]
                if "wd_mult" in adj:
                    self.wd_mult = adj["wd_mult"]
                self.adjustments += 1
                self.last_change = step
                if "gn_mult" in adj:
                    self.last_gn_change = step

        # ── Track failed adjustments ──
        if adj and "lr_mult" in adj and self.last_adj_loss is not None:
            if self.ema_short >= self.last_adj_loss:
                self.failed_adjs += 1
        if adj and "lr_mult" in adj:
            self.prev_lr_mult = orig_lr
            self.prev_gn_mult = orig_gn
            self.prev_wd_mult = orig_wd
            self.last_adj_loss = self.ema_short
            self._last_adj_step = step
            self._eval_cd = max(250, self.cooldown // 2)
            self._last_adj_state = state
            self.rollback_ready = True
            if self.short_init:
                self._pending_dynamics = {
                    "loss_before": self.ema_short,
                    "d_lr": adj["lr_mult"] - orig_lr,
                    "d_wd": adj.get("wd_mult", self.wd_mult) - orig_wd,
                    "step": step,
                    "eval_cd": self._eval_cd,
                }

        # ── Safety interlock ──
        if adj:
            self.adj_window.append(step)
        if self.failed_adjs >= 3 and not self.frozen:
            self.frozen = True
            self.frozen_until = step + self.cooldown * 2
            adj["action"] = adj.get("action", "") + " [FROZEN]"
        if self.frozen and step > self.frozen_until:
            self.frozen = False
            self.failed_adjs = 0

        # ── Credit attribution ──
        if adj and "lr_mult" in adj:
            self.state_snapshot = {
                "lr": float(self.lr_mult), "wd": float(self.wd_mult),
                "gn": float(self.gn_mult), "mom": float(self.mom_boost),
                "loss": float(self.ema_short) if self.short_init else None,
                "step": step, "state": state,
            }
        if "mom" in adj:
            self.mom_boost = adj["mom"]
        return adj

    # ── Effective value accessors ──
    def effective_lr(self, sched_lr):
        return sched_lr * self.lr_mult

    def effective_gn(self, base_gn=None):
        return (self.base_gn if base_gn is None else base_gn) * self.gn_mult

    def effective_mom(self, base=0.9):
        return min(0.99, base + self.mom_boost)

    def effective_wd(self, base_wd):
        return base_wd * self.wd_mult

    # ── Stats ──
    def stats(self):
        s = {
            "phase": self.phase, "state": self.state,
            "lr_mult": f"{self.lr_mult:.3f}",
            "eff_lr": f"{self.base_lr * self.lr_mult:.2e}",
            "gn_mult": f"{self.gn_mult:.3f}",
            "wd_mult": f"{self.wd_mult:.2f}",
            "mom": f"{self.mom_boost:.3f}",
            "best": f"{self.best:.4f} (step {self.best_step})",
            "plateau": self.plateau,
            "stuck": self.stuck_steps,
            "conf": f"{self.confidence:.2f}" if self.conf_init else "N/A",
            "gap": f"{self.gap_ema:.3f}" if self.gap_init else "N/A",
            "sick": len(self.sick_layers),
            "quiet": self.quiet_mode,
            "frozen": self.frozen,
            "fail_adj": self.failed_adjs,
            "pot_dev": f"{self.dev_potential:.3f}" if self.pot_init else "N/A",
            "mode": self.mode,
            "energy": f"{self.energy_spent:.1f}/{self.energy_budget:.0f}",
            "lrsens_n": self.dynamics["global"]["n"],
            "phases_tracked": len(self.dynamics["per_phase"]),
            "cos_sim": f"{self.cos_sim_ema:.3f}" if self.cos_sim_init else "N/A",
            "upd_ratio": f"{self.update_ratio_ema:.6f}" if self.ur_init else "N/A",
            "gn_noise": f"{self.gn_noise_ema:.3f}" if self.gn_noise_init else "N/A",
            "adj": self.adjustments, "anomalies": self.anomalies,
            "div_red": self.div_reductions, "deg_red": self.deg_reductions,
            "revive": self.revive_attempts,
        }
        d = self.dynamics["global"]
        if d["n"] >= 3:
            lr_std = math.sqrt(max(d.get("lr_sens_m2", 0) / (d["n"] - 1), 0))
            s["lr_sens"] = f"{d['lr_sens_mean']:.3f}±{lr_std:.3f}"
        else:
            s["lr_sens"] = "N/A"
        s["diagnosis"] = self.diagnosis
        s["plan_strategy"] = self.plan.get("strategy", "none")
        s["plan_accuracy"] = f"{self.plan_accuracy:.2f}"
        s["traj_score"] = f"{self._traj_score:.2f}"
        if self._grad_moments_init:
            s["grad_skew"] = f"{self._grad_skew_ema:.3f}"
            s["grad_kurt"] = f"{self._grad_kurt_ema:.3f}"
        if self._hessian_trace_init:
            s["hess_tr"] = f"{self._hessian_trace_ema:.2e}"
            s["hess_nr"] = f"{self._hessian_noise_ratio:.3f}"
        if self._lip_init:
            s["lip_bound"] = f"{self._lip_bound:.2e}"
        if self._grad_autocorr_init[1]:
            ac_str = " ".join(f"ρ{τ}={self._grad_autocorr.get(τ,0):.2f}" for τ in self.AUTO_LAGS)
            s["grad_ac"] = ac_str
        if self._loss_corr_init:
            s["loss_ac1"] = f"{self._loss_lag1_corr:.4f}"
        if self._kalman_init:
            s["kalman"] = f"{self._kalman_x:.4f}"
            s["kalman_g"] = f"{self._kalman_gain:.3f}"
        if self._spectral_init:
            s["spec_ratio"] = f"{self._grad_spectral_ratio:.3f}"
        if self.traj_fit:
            s["traj_slope"] = f"{self.traj_fit['slope']:.4f}"
            s["traj_r2"] = f"{self.traj_fit['r2']:.3f}"
            if self.traj_convergence_step:
                s["convergence_at"] = f"step {self.traj_convergence_step:,}"
        if self.goals["velocity_ema"]:
            s["loss_velocity"] = f"{self.goals['velocity_ema']:.3f}/1k"
        if self.goals["eta_to_target"]:
            s["eta_target"] = f"step {self.goals['eta_to_target']:,}"
        milestones_reached = sum(1 for _, r in self.goals["milestones"] if r is not None)
        s["milestones"] = f"{milestones_reached}/{len(self.goals['milestones'])}"
        return s

    # ── State save ──
    def save_state(self):
        sd = {
            "lr_mult": self.lr_mult, "gn_mult": self.gn_mult,
            "mom_boost": self.mom_boost, "best": self.best,
            "best_step": self.best_step, "plateau": self.plateau,
            "div_reductions": self.div_reductions,
            "deg_reductions": self.deg_reductions,
            "adjustments": self.adjustments, "recovery": self.recovery,
            "init_best": self.init_best,
            "last_change": self.last_change,
            "last_gn_change": self.last_gn_change,
            "cooldown": self.cooldown,
            "phase": self.phase, "eff_floor": self.eff_floor,
            "eff_ceil": self.eff_ceil,
            "ema_short": self.ema_short, "short_init": self.short_init,
            "ema_med": self.ema_med, "med_init": self.med_init,
            "ema_long": self.ema_long, "long_init": self.long_init,
            "cos_sim_ema": self.cos_sim_ema,
            "cos_sim_init": self.cos_sim_init,
            "update_ratio_ema": self.update_ratio_ema,
            "ur_init": self.ur_init, "ur_alpha": self.ur_alpha,
            "gn_noise_ema": self.gn_noise_ema,
            "gn_noise_init": self.gn_noise_init,
            "gn_noise_alpha": self.gn_noise_alpha,
            "gn_ema": self.gn_ema, "gn_var": self.gn_var,
            "gn_var_init": self.gn_var_init,
            "gn_hist": list(self.gn_hist),
            "anomalies": self.anomalies,
            "gn_gap_steps": self.gn_gap_steps,
            "hist": list(self.hist),
            "wd_mult": self.wd_mult,
            "gap_ema": self.gap_ema,
            "gap_init": self.gap_init,
            "overfit_signal": self.overfit_signal,
            "stuck_steps": self.stuck_steps,
            "revive_attempts": self.revive_attempts,
            "confidence": self.confidence,
            "conf_init": self.conf_init,
            "rollback_ready": self.rollback_ready,
            "prev_lr_mult": self.prev_lr_mult,
            "prev_gn_mult": self.prev_gn_mult,
            "prev_wd_mult": self.prev_wd_mult,
            "quiet_mode": self.quiet_mode,
            "quiet_until": self.quiet_until,
            "frozen": self.frozen,
            "frozen_until": self.frozen_until,
            "failed_adjs": self.failed_adjs,
            "potential": self.potential,
            "pot_init": self.pot_init,
            "dev_potential": self.dev_potential,
            "energy_spent": self.energy_spent,
            "mode": self.mode,
            "mode_changed_at": self.mode_changed_at,
            "diagnosis": self.diagnosis,
            "diagnosis_reasons": self.diagnosis_reasons,
            "model_needs": self.model_needs,
            "plan": self.plan,
            "plan_accuracy": self.plan_accuracy,
            "plan_last_update": self.plan_last_update,
            "traj_fit": self.traj_fit,
            "traj_predictions": self.traj_predictions,
            "traj_convergence_step": self.traj_convergence_step,
            "traj_last_fit": self.traj_last_fit,
            "goals": self.goals,
            "_traj_score": self._traj_score,
            # v10: Mathematical signals
            "_grad_skew_ema": self._grad_skew_ema,
            "_grad_kurt_ema": self._grad_kurt_ema,
            "_grad_moments_init": self._grad_moments_init,
            "_loss_lag1_corr": self._loss_lag1_corr,
            "_loss_lag2_corr": self._loss_lag2_corr,
            "_loss_corr_init": self._loss_corr_init,
            "_hessian_trace_ema": self._hessian_trace_ema,
            "_hessian_trace_init": self._hessian_trace_init,
            "_hessian_noise_ratio": self._hessian_noise_ratio,
            "_lip_bound": self._lip_bound,
            "_lip_init": self._lip_init,
            "_grad_spectral_ratio": self._grad_spectral_ratio,
            "_spectral_init": self._spectral_init,
            "_kalman_x": self._kalman_x,
            "_kalman_P": self._kalman_P,
            "_kalman_init": self._kalman_init,
        }
        sd["dynamics_global"] = dict(self.dynamics["global"])
        sd["dynamics_per_phase"] = {k: dict(v) for k, v in self.dynamics["per_phase"].items()}
        if self.eval_losses:
            sd["eval_losses"] = list(self.eval_losses)
        if self.grad_hist_ema:
            sd["grad_hist_ema"] = self.grad_hist_ema
        if self.adj_memory:
            sd["adj_memory"] = list(self.adj_memory)
        if self.last_adj_loss is not None:
            sd["last_adj_loss"] = self.last_adj_loss
            sd["_last_adj_step"] = self._last_adj_step
            sd["_eval_cd"] = self._eval_cd
            sd["_last_adj_state"] = self._last_adj_state
        if self._prev_loss is not None:
            sd["_prev_loss"] = self._prev_loss
        if self.plan_cache:
            sd["plan_cache"] = {str(k): dict(v) for k, v in self.plan_cache.items()}
        if self.plan_history:
            sd["plan_history"] = list(self.plan_history)
        if self._pending_dynamics:
            sd["_pending_dynamics"] = dict(self._pending_dynamics)
        return sd

    # ── State restore ──
    def restore_state(self, state):
        if not state:
            return
        self.lr_mult = state.get("lr_mult", 1.0)
        self.gn_mult = state.get("gn_mult", 1.0)
        self.mom_boost = state.get("mom_boost", 0.0)
        self.best = state.get("best", float("inf"))
        self.best_step = state.get("best_step", 0)
        self.plateau = state.get("plateau", 0)
        self.div_reductions = state.get("div_reductions", state.get("reductions", 0))
        self.deg_reductions = state.get("deg_reductions", state.get("reductions", 0))
        self.adjustments = state.get("adjustments", 0)
        self.recovery = state.get("recovery", 0)
        self.init_best = state.get("init_best", True)
        self.last_change = state.get("last_change", self.warmup)
        self.last_gn_change = state.get("last_gn_change", 0)
        self.cooldown = state.get("cooldown", 500)
        self.phase = state.get("phase", "warmup")
        self.eff_floor = state.get("eff_floor", 0.05)
        self.eff_ceil = state.get("eff_ceil", 2.0)
        self.ema_short = state.get("ema_short", 0.0)
        self.short_init = state.get("short_init", False)
        self.ema_med = state.get("ema_med", 0.0)
        self.med_init = state.get("med_init", False)
        self.ema_long = state.get("ema_long", 0.0)
        self.long_init = state.get("long_init", False)
        self.cos_sim_ema = state.get("cos_sim_ema", 0.0)
        self.cos_sim_init = state.get("cos_sim_init", False)
        self.update_ratio_ema = state.get("update_ratio_ema", 0.0)
        self.ur_init = state.get("ur_init", False)
        self.ur_alpha = state.get("ur_alpha", 0.05)
        self.gn_noise_ema = state.get("gn_noise_ema", 0.0)
        self.gn_noise_init = state.get("gn_noise_init", False)
        self.gn_noise_alpha = state.get("gn_noise_alpha", 0.1)
        self.gn_ema = state.get("gn_ema", 0.0)
        self.gn_var = state.get("gn_var", 0.0)
        self.gn_var_init = state.get("gn_var_init", False)
        self.anomalies = state.get("anomalies", 0)
        self.gn_gap_steps = state.get("gn_gap_steps", 0)
        saved_hist = state.get("hist")
        if saved_hist:
            self.hist = deque(saved_hist, maxlen=self.HIST_MAXLEN)
        saved_gn = state.get("gn_hist")
        if saved_gn:
            self.gn_hist = deque(saved_gn, maxlen=self.GNHIST_MAXLEN)
        if "grad_hist_ema" in state:
            self.grad_hist_ema = state["grad_hist_ema"]
        self.grad_hist = self.grad_hist_ema
        self._update_gradient_autocorr(model)
        self._update_gradient_moments(model)
        self._update_spectral_ratio(model)
        if "adj_memory" in state:
            self.adj_memory = deque(state["adj_memory"], maxlen=self.ADJMEM_MAXLEN)
        if "last_adj_loss" in state:
            self.last_adj_loss = state["last_adj_loss"]
            self._last_adj_step = state.get("_last_adj_step", 0)
            self._eval_cd = state.get("_eval_cd", 250)
            self._last_adj_state = state.get("_last_adj_state", None)
        if "_prev_loss" in state:
            self._prev_loss = state["_prev_loss"]
        self.wd_mult = state.get("wd_mult", 1.0)
        self.gap_ema = state.get("gap_ema", 0.0)
        self.gap_init = state.get("gap_init", False)
        self.overfit_signal = state.get("overfit_signal", False)
        self.stuck_steps = state.get("stuck_steps", 0)
        self.revive_attempts = state.get("revive_attempts", 0)
        self.confidence = state.get("confidence", 0.0)
        self.conf_init = state.get("conf_init", False)
        self.rollback_ready = state.get("rollback_ready", False)
        self.prev_lr_mult = state.get("prev_lr_mult", 1.0)
        self.prev_gn_mult = state.get("prev_gn_mult", 1.0)
        self.prev_wd_mult = state.get("prev_wd_mult", 1.0)
        saved_eval = state.get("eval_losses")
        if saved_eval:
            self.eval_losses = deque(saved_eval, maxlen=self.EVALLOSS_MAXLEN)
        self.quiet_mode = state.get("quiet_mode", False)
        self.quiet_until = state.get("quiet_until", 0)
        self.frozen = state.get("frozen", False)
        self.frozen_until = state.get("frozen_until", 0)
        self.failed_adjs = state.get("failed_adjs", 0)
        self.potential = state.get("potential", float("inf"))
        self.pot_init = state.get("pot_init", False)
        self.dev_potential = state.get("dev_potential", 0.0)
        dyn_raw = state.get("dynamics_global", {})
        # Migrate old format (lr_sens/wd_sens/sens_init) → Bayesian (lr_sens_mean/lr_sens_m2)
        if "lr_sens" in dyn_raw and "lr_sens_mean" not in dyn_raw:
            dyn_raw = {
                "lr_sens_mean": dyn_raw.get("lr_sens", 0.0),
                "lr_sens_m2": 0.0,
                "wd_sens_mean": dyn_raw.get("wd_sens", 0.0),
                "wd_sens_m2": 0.0,
                "n": dyn_raw.get("n", 0),
            }
        self.dynamics = {
            "global": dyn_raw,
            "per_phase": state.get("dynamics_per_phase", {}),
        }
        self.energy_spent = state.get("energy_spent", 0.0)
        self.mode = state.get("mode", "normal")
        self.mode_changed_at = state.get("mode_changed_at", 0)
        self.diagnosis = state.get("diagnosis", "unknown")
        self.diagnosis_reasons = state.get("diagnosis_reasons", [])
        self.model_needs = state.get("model_needs", [])
        self.plan = state.get("plan", {})
        self.plan_accuracy = state.get("plan_accuracy", 0.5)
        self.plan_last_update = state.get("plan_last_update", 0)
        self.traj_fit = state.get("traj_fit")
        self.traj_predictions = state.get("traj_predictions", {})
        self.traj_convergence_step = state.get("traj_convergence_step")
        self.traj_last_fit = state.get("traj_last_fit", 0)
        self._traj_score = state.get("_traj_score", 1.0)
        saved_goals = state.get("goals")
        if saved_goals:
            self.goals.update(saved_goals)
        saved_plan_cache = state.get("plan_cache")
        if saved_plan_cache:
            self.plan_cache = {eval(k) if isinstance(k, str) else k: dict(v)
                               for k, v in saved_plan_cache.items()}
        saved_plan_hist = state.get("plan_history")
        if saved_plan_hist:
            self.plan_history = deque(saved_plan_hist, maxlen=self.PLANHIST_MAXLEN)
        saved_pending = state.get("_pending_dynamics")
        if saved_pending:
            self._pending_dynamics = dict(saved_pending)

        # v10: Restore mathematical signals
        self._grad_skew_ema = state.get("_grad_skew_ema", 0.0)
        self._grad_kurt_ema = state.get("_grad_kurt_ema", 0.0)
        self._grad_moments_init = state.get("_grad_moments_init", False)
        self._loss_lag1_corr = state.get("_loss_lag1_corr", 0.0)
        self._loss_lag2_corr = state.get("_loss_lag2_corr", 0.0)
        self._loss_corr_init = state.get("_loss_corr_init", False)
        self._hessian_trace_ema = state.get("_hessian_trace_ema", 0.0)
        self._hessian_trace_init = state.get("_hessian_trace_init", False)
        self._hessian_noise_ratio = state.get("_hessian_noise_ratio", 0.0)
        self._lip_bound = state.get("_lip_bound", float("inf"))
        self._lip_init = state.get("_lip_init", False)
        self._grad_spectral_ratio = state.get("_grad_spectral_ratio", 1.0)
        self._spectral_init = state.get("_spectral_init", False)
        self._kalman_x = state.get("_kalman_x", 0.0)
        self._kalman_P = state.get("_kalman_P", 1.0)
        self._kalman_init = state.get("_kalman_init", False)

    # ── v9: New Algorithms ──

    def suggest_accumulation(self):
        if self.stuck_steps > self.revive_thr * 0.5:
            return 2.0
        if self.plateau > self.plateau_thr * 0.5:
            return 1.5
        if self.gn_noise_ema > 1.0:
            return 1.5
        return 1.0

    def _estimate_landscape_roughness(self):
        if not self.gn_noise_init or not self.ur_init:
            return None
        roughness = self.gn_noise_ema * self.update_ratio_ema
        self.roughness_ema = 0.9 * self.roughness_ema + 0.1 * roughness
        self.roughness_init = True
        return self.roughness_ema

    def _schedule_momentum(self, step):
        base = 0.9
        if self.phase == "exploration":
            base = 0.95
        elif self.phase == "exploitation":
            base = 0.85
        decay = max(0, 1 - (step / self.total) * 0.1)
        return base * decay

    def _detect_plateau_risk(self, step):
        if self.plateau > self.plateau_thr:
            return True
        if self.update_ratio_ema < 0.0001 and self.gn_noise_ema < 0.5:
            return True
        if self.traj_fit and self.traj_fit["slope"] > -0.001:
            return True
        return False

    def _compute_cycle_lr(self, step):
        cycle_pos = (step % self._cycle_period) / self._cycle_period
        offset = self._cycle_amplitude * math.sin(2 * math.pi * cycle_pos)
        return offset

    def _compute_per_group_lr_suggestions(self, base_lr_mult):
        suggestions = {}
        for group, ratio in self.layer_ratios.items():
            if group in self.sick_layers:
                if ratio > 4.0:
                    suggestions[group] = base_lr_mult / (ratio * 0.25)
                elif ratio < 0.1:
                    suggestions[group] = base_lr_mult * ratio * 10.0
                else:
                    suggestions[group] = base_lr_mult
            else:
                suggestions[group] = base_lr_mult
        return suggestions

    def _update_per_group_stats(self, step):
        for group, grads in self.layer_gn.items():
            if len(grads) >= 5:
                gn = sorted(grads)[len(grads) // 2]
                if group not in self._per_group_grad_norms:
                    self._per_group_grad_norms[group] = deque(maxlen=20)
                self._per_group_grad_norms[group].append(gn)

    def effective_mom(self):
        return 0.9 * self.mom_boost + 0.1 * self._scheduled_mom

    # ── Dataset awareness ──

    def record_category_loss(self, category, loss, step):
        """Record per-category loss for dataset routing."""
        if not category:
            return
        if category not in self._cat_loss_ema:
            self._cat_loss_ema[category] = loss
            self._cat_loss_init[category] = False
        alpha = 0.05
        if not self._cat_loss_init[category]:
            self._cat_loss_ema[category] = loss
            self._cat_loss_init[category] = True
        else:
            self._cat_loss_ema[category] = alpha * loss + (1 - alpha) * self._cat_loss_ema[category]
        self.dataset_info[category] = {
            "loss_ema": self._cat_loss_ema[category],
            "last_step": step,
        }

    def suggest_dataset(self, step):
        """Suggest dataset category weight adjustments based on training signals.

        Returns dict of {category: weight_adjustment}.
        Works in manual/explore mode — no subagent needed.
        """
        suggestions = {}
        if step - self._last_dataset_suggest < 500:
            return suggestions
        self._last_dataset_suggest = step

        # Check per-category performance
        for cat, info in self.dataset_info.items():
            loss = info.get("loss_ema")
            if loss is None:
                continue
            # High loss → reduce weight
            if loss > self.best * 1.5:
                suggestions[cat] = -0.1
            # Low loss → increase weight
            elif loss < self.best * 0.8:
                suggestions[cat] = 0.15

        # Phase-based suggestions
        if self.phase == "warmup":
            suggestions["grammar"] = suggestions.get("grammar", 0) + 0.2
            suggestions["dialog"] = suggestions.get("dialog", 0) + 0.15
            suggestions["instruction"] = suggestions.get("instruction", 0) + 0.15
        elif self.phase == "exploration":
            suggestions["reasoning"] = suggestions.get("reasoning", 0) + 0.2
            suggestions["code"] = suggestions.get("code", 0) + 0.15
            suggestions["instruction"] = suggestions.get("instruction", 0) + 0.1
        elif self.phase == "balanced":
            suggestions["reasoning"] = suggestions.get("reasoning", 0) + 0.15
            suggestions["creative"] = suggestions.get("creative", 0) + 0.1
            suggestions["academic"] = suggestions.get("academic", 0) + 0.1
        elif self.phase == "exploitation":
            suggestions["reasoning"] = suggestions.get("reasoning", 0) + 0.2
            suggestions["summarization"] = suggestions.get("summarization", 0) + 0.15
            suggestions["academic"] = suggestions.get("academic", 0) + 0.1

        # Plateau → boost diversity
        if self.state == "plateau":
            for cat in ("creative", "dialog", "instruction"):
                suggestions[cat] = suggestions.get(cat, 0) + 0.1
        # Overfit → reinforce grammar + instruction
        if self.overfit_signal:
            suggestions["grammar"] = suggestions.get("grammar", 0) + 0.15
            suggestions["instruction"] = suggestions.get("instruction", 0) + 0.1
        # Sick layers → simplify
        if self.sick_layers:
            for cat in ("reasoning", "code"):
                suggestions[cat] = suggestions.get(cat, 0) - 0.15

        return suggestions

    def set_dataset_mode(self, mode, weights=None):
        """Set dataset routing mode.

        mode: 'auto' | 'manual' | 'explore'
        weights: optional dict {category: weight} for manual mode
        """
        self._dataset_mode = mode
        if weights:
            self._dataset_weights = dict(weights)

    def get_dataset_weights(self):
        if self._dataset_mode == "manual" and self._dataset_weights:
            return dict(self._dataset_weights)
        return {}

    def dataset_stats(self):
        return {
            "mode": self._dataset_mode,
            "categories": list(self.dataset_info.keys()),
            "cat_losses": {
                k: round(v.get("loss_ema", 0), 4)
                for k, v in self.dataset_info.items()
            },
        }

    # ═══════════════════════════════════════════════════════════════
    # Full Training Control — tuner controls ALL training aspects
    # ═══════════════════════════════════════════════════════════════

    def get_training_control(self, step: int) -> TrainingControl:
        """Get full training control for current step.

        Returns a TrainingControl object with all parameters adjusted
        based on current training signals, phase, and model state.
        """
        ctrl = TrainingControl()
        ctrl.step = step
        ctrl.phase = self.phase
        ctrl.mode = self._dataset_mode

        # ── 1. Hyperparameters (existing logic) ──
        ctrl.hyperparams.lr_mult = self.lr_mult
        ctrl.hyperparams.gn_mult = self.gn_mult
        ctrl.hyperparams.wd_mult = self.wd_mult
        ctrl.hyperparams.mom_boost = self.mom_boost
        ctrl.hyperparams.mom_decay = self.mom_decay

        # ── 2. Dataset control ──
        self._apply_dataset_control(ctrl, step)

        # ── 3. Batch control ──
        self._apply_batch_control(ctrl, step)

        # ── 4. Precision control ──
        self._apply_precision_control(ctrl, step)

        # ── 5. Model control ──
        self._apply_model_control(ctrl, step)

        # ── 6. Regularization control ──
        self._apply_regularization_control(ctrl, step)

        # ── 7. Schedule control ──
        self._apply_schedule_control(ctrl, step)

        # ── 8. MoE control ──
        self._apply_moe_control(ctrl, step)

        # ── 9. MTP control ──
        self._apply_mtp_control(ctrl, step)

        # ── 10. Optimizer control ──
        self._apply_optimizer_control(ctrl, step)

        return ctrl

    def _apply_dataset_control(self, ctrl, step):
        """Control dataset routing, difficulty, and mix."""
        progress = step / max(self.total, 1)

        # Difficulty increases with training progress
        base_difficulty = min(1.0, progress * 1.5)

        # Adjust based on model state
        if self.state == "diverging":
            base_difficulty *= 0.5  # Simplify when diverging
        elif self.state == "plateau":
            base_difficulty = max(0.3, base_difficulty - 0.2)  # Reduce to explore
        elif self.state == "improving":
            base_difficulty = min(1.0, base_difficulty + 0.1)  # Push harder

        ctrl.dataset.difficulty = base_difficulty

        # Category weights based on phase and performance
        if self.phase == "warmup":
            ctrl.dataset.mix_strategy = "curriculum"
            ctrl.dataset.category_weights = {
                "grammar": 3.0, "dialog": 2.0, "instruction": 2.0,
                "reasoning": 0.5, "code": 0.5, "creative": 1.0,
                "academic": 0.3, "summarization": 0.3,
            }
        elif self.phase == "exploration":
            ctrl.dataset.mix_strategy = "explore"
            ctrl.dataset.category_weights = {
                "reasoning": 3.0, "code": 2.5, "instruction": 2.0,
                "grammar": 1.5, "creative": 1.5, "dialog": 1.0,
                "academic": 0.5, "summarization": 0.5,
            }
        elif self.phase == "balanced":
            ctrl.dataset.mix_strategy = "balanced"
            ctrl.dataset.category_weights = {
                "reasoning": 2.5, "code": 2.0, "creative": 2.0,
                "academic": 1.5, "instruction": 1.5, "grammar": 1.0,
                "summarization": 1.0, "dialog": 1.0,
            }
        elif self.phase == "exploitation":
            ctrl.dataset.mix_strategy = "exploit"
            ctrl.dataset.category_weights = {
                "reasoning": 3.0, "summarization": 2.5, "academic": 2.0,
                "code": 1.5, "creative": 1.0, "instruction": 0.8,
                "grammar": 0.5, "dialog": 0.3,
            }

        # Adjust weights based on per-category performance
        for cat, info in self.dataset_info.items():
            loss = info.get("loss_ema")
            if loss is not None and loss > self.best * 1.3:
                # High loss category → reduce weight
                if cat in ctrl.dataset.category_weights:
                    ctrl.dataset.category_weights[cat] *= 0.8
            elif loss is not None and loss < self.best * 0.9:
                # Low loss category → increase weight
                if cat in ctrl.dataset.category_weights:
                    ctrl.dataset.category_weights[cat] *= 1.1

        # Sequence length based on difficulty
        ctrl.dataset.seq_len_mult = 0.5 + base_difficulty * 0.5

    def _apply_batch_control(self, ctrl, step):
        """Control batch size and gradient accumulation."""
        progress = step / max(self.total, 1)

        # Start with smaller batches, increase over time
        base_batch = 1.0 + progress * 1.0  # 1x → 2x

        # Reduce batch when diverging or oscillating
        if self.state in ("diverging", "oscillating"):
            base_batch *= 0.5
        # Increase batch when stable and improving
        elif self.state == "improving" and self.cos_sim_init and self.cos_sim_ema > 0.5:
            base_batch *= 1.2

        # Gradient accumulation: increase when batch is small or loss is noisy
        if self.gn_noise_init and self.gn_noise_ema > 1.0:
            ctrl.batch.grad_accum = min(ctrl.batch.max_grad_accum, 4)
        elif self.gn_noise_init and self.gn_noise_ema < 0.3:
            ctrl.batch.grad_accum = 1
        else:
            ctrl.batch.grad_accum = 2

        ctrl.batch.batch_mult = max(0.5, min(4.0, base_batch))
        ctrl.batch.dynamic_batch = self.state not in ("diverging", "spike")

    def _apply_precision_control(self, ctrl, step):
        """Control numerical precision."""
        # Default to bf16 for stability
        ctrl.precision.dtype = "bf16"

        # Use fp16 if loss is stable and gradients are well-behaved
        if (self.gn_var_init and self.gn_var < 0.01 and
                self.cos_sim_init and self.cos_sim_ema > 0.3):
            ctrl.precision.dtype = "fp16"

        # Use fp32 if diverging or NaN detected
        if self.state == "diverging" or self.anomalies > 5:
            ctrl.precision.dtype = "fp32"
            ctrl.precision.loss_scale = 1.0
        else:
            # Dynamic loss scaling based on gradient norm
            if self.gn_ema > 10.0:
                ctrl.precision.loss_scale = 32768.0
            elif self.gn_ema > 1.0:
                ctrl.precision.loss_scale = 65536.0
            else:
                ctrl.precision.loss_scale = 131072.0

    def _apply_model_control(self, ctrl, step):
        """Control model architecture parameters."""
        progress = step / max(self.total, 1)

        # Expert top-k: start low, increase as model learns
        if self.phase == "warmup":
            ctrl.model.expert_topk = 1
        elif self.phase == "exploration":
            ctrl.model.expert_topk = 2
        else:
            ctrl.model.expert_topk = min(4, 2 + int(progress * 2))

        # Layer dropout: higher early, lower later
        ctrl.model.layer_drop_rate = max(0.0, 0.1 * (1 - progress * 2))

        # Attention dropout: reduce as training stabilizes
        if self.state == "diverging":
            ctrl.model.attention_dropout = 0.1
        elif self.state == "oscillating":
            ctrl.model.attention_dropout = 0.05
        else:
            ctrl.model.attention_dropout = max(0.0, 0.05 * (1 - progress))

        # Hidden dropout: similar pattern
        ctrl.model.hidden_dropout = ctrl.model.attention_dropout * 0.5

    def _apply_regularization_control(self, ctrl, step):
        """Control regularization parameters."""
        progress = step / max(self.total, 1)

        # Dropout: higher early for exploration, lower later for fine-tuning
        ctrl.regularization.dropout = max(0.0, 0.1 * (1 - progress * 1.5))

        # Label smoothing: increase when overfitting
        if self.overfit_signal:
            ctrl.regularization.label_smoothing = 0.1
        else:
            ctrl.regularization.label_smoothing = 0.0

        # Gradient penalty: increase when gradients are unstable
        if self.gn_var_init and self.gn_var > 1.0:
            ctrl.regularization.grad_penalty = 0.01
        else:
            ctrl.regularization.grad_penalty = 0.0

        # Activation checkpointing: enable for memory savings on long sequences
        ctrl.regularization.activation_checkpointing = progress > 0.5

        # Clipping strategy: adaptive when unstable, norm when stable
        if self.state in ("diverging", "oscillating"):
            ctrl.regularization.clip_strategy = "adaptive"
        else:
            ctrl.regularization.clip_strategy = "norm"

    def _apply_schedule_control(self, ctrl, step):
        """Control learning rate schedule."""
        progress = step / max(self.total, 1)

        # Warmup ratio: fixed from config
        ctrl.schedule.warmup_ratio = self.warmup / max(self.total, 1)

        # Decay type: cosine for most cases, linear for exploitation
        if self.phase == "exploitation":
            ctrl.schedule.decay_type = "linear"
        else:
            ctrl.schedule.decay_type = "cosine"

        # Cyclical LR: enable during plateau to escape local minima
        ctrl.schedule.cyclical_lr = self.state == "plateau"
        ctrl.schedule.cyclical_amplitude = self._cycle_amplitude

        # Min LR ratio: lower for exploitation phase
        if self.phase == "exploitation":
            ctrl.schedule.min_lr_ratio = 0.05
        else:
            ctrl.schedule.min_lr_ratio = 0.1

    def _apply_moe_control(self, ctrl, step):
        """Control MoE-specific parameters."""
        progress = step / max(self.total, 1)

        # Routing temperature: start high (soft routing), decrease over time
        ctrl.moe.routing_temperature = max(0.5, 2.0 - progress * 1.5)

        # Load balance weight: higher early to encourage exploration
        ctrl.moe.load_balance_weight = 0.01 + 0.02 * (1 - progress)

        # Expert capacity: increase as model learns to use experts better
        ctrl.moe.expert_capacity_factor = 1.0 + progress * 0.5

        # Expert dropout: reduce over time
        ctrl.moe.expert_dropout = max(0.0, 0.1 * (1 - progress * 2))

    def _apply_mtp_control(self, ctrl, step):
        """Control Multi-Token Prediction parameters."""
        progress = step / max(self.total, 1)

        # MTP depth: start at 1, increase as model learns
        if progress < 0.1:
            ctrl.mtp.depth = 1
        elif progress < 0.5:
            ctrl.mtp.depth = 2
        else:
            ctrl.mtp.depth = 3

        # MTP weight: increase as model gets better at main task
        if self.state == "improving":
            ctrl.mtp.loss_weight = 0.1 + progress * 0.1
        elif self.state == "diverging":
            ctrl.mtp.loss_weight = 0.0  # Disable when unstable
        else:
            ctrl.mtp.loss_weight = 0.1

        # Enable MTP only after warmup
        ctrl.mtp.enabled = step > self.warmup

    def _apply_optimizer_control(self, ctrl, step):
        """Control optimizer parameters."""
        progress = step / max(self.total, 1)

        # Beta1: slightly higher for stable training
        if self.state == "diverging":
            ctrl.optimizer.beta1 = 0.85  # More responsive
        else:
            ctrl.optimizer.beta1 = 0.9

        # Beta2: lower for non-stationary objectives
        if self.state == "plateau":
            ctrl.optimizer.beta2 = 0.99  # More responsive to recent grads
        else:
            ctrl.optimizer.beta2 = 0.999

        # Weight decay: increase during exploitation for regularization
        if self.phase == "exploitation":
            ctrl.optimizer.weight_decay = 0.02
        elif self.phase == "exploration":
            ctrl.optimizer.weight_decay = 0.005
        else:
            ctrl.optimizer.weight_decay = 0.01

    def apply_control(self, control: TrainingControl, trainer=None):
        """Apply training control to actual training components.

        Args:
            control: TrainingControl from get_training_control()
            trainer: Optional trainer object to apply controls to
        """
        # Apply hyperparameters (existing)
        self.lr_mult = control.hyperparams.lr_mult
        self.gn_mult = control.hyperparams.gn_mult
        self.wd_mult = control.hyperparams.wd_mult
        self.mom_boost = control.hyperparams.mom_boost

        # Apply dataset control
        if hasattr(self, '_curriculum_router') and self._curriculum_router:
            router = self._curriculum_router
            router.set_phase(self.phase, control.step)
            if control.dataset.category_weights:
                router.set_manual_weights(control.dataset.category_weights)

        # Apply to trainer if provided
        if trainer:
            if hasattr(trainer, 'apply_training_control'):
                trainer.apply_training_control(control)
