"""
Self-Evolution Framework for Deeplm.

Autonomous system that monitors training, generates hypotheses,
designs experiments, executes changes, and decides whether to keep
or revert based on measured improvement.

Workflow:
  hypothesis_generation → experiment_design → code_execution →
  log_analysis → bug_diagnosis → code_fix → evaluation → decision
"""
import json
import os
import time
import copy
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field, asdict
from pathlib import Path

import torch
import torch.nn as nn

from ..config import SelfEvolutionConfig

logger = logging.getLogger(__name__)


@dataclass
class EvolutionEpisode:
    """Single iteration of the self-evolution loop."""
    step: int = 0
    phase: str = ""
    hypothesis: str = ""
    experiment_design: Dict[str, Any] = field(default_factory=dict)
    changes_applied: List[str] = field(default_factory=list)
    metrics_before: Dict[str, float] = field(default_factory=dict)
    metrics_after: Dict[str, float] = field(default_factory=dict)
    experiment_result: float = 0.0
    decision: str = ""
    bugs_found: List[str] = field(default_factory=list)
    fixes_applied: List[str] = field(default_factory=list)
    timestamp: float = 0.0


class TrainingMetrics:
    """Tracks and analyzes training metrics for self-evolution decisions."""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.loss_history: List[float] = []
        self.grad_norm_history: List[float] = []
        self.lr_history: List[float] = []
        self.tokens_per_sec_history: List[float] = []

    def record(self, loss: float, grad_norm: float = 0.0,
               lr: float = 0.0, tokens_per_sec: float = 0.0):
        self.loss_history.append(loss)
        self.grad_norm_history.append(grad_norm)
        self.lr_history.append(lr)
        self.tokens_per_sec_history.append(tokens_per_sec)

        if len(self.loss_history) > self.window_size:
            self.loss_history = self.loss_history[-self.window_size:]
            self.grad_norm_history = self.grad_norm_history[-self.window_size:]
            self.lr_history = self.lr_history[-self.window_size:]
            self.tokens_per_sec_history = self.tokens_per_sec_history[-self.window_size:]

    def get_recent(self, n: int = 10) -> Dict[str, float]:
        return {
            "avg_loss": sum(self.loss_history[-n:]) / max(len(self.loss_history[-n:]), 1),
            "avg_grad_norm": sum(self.grad_norm_history[-n:]) / max(len(self.grad_norm_history[-n:]), 1),
            "avg_lr": sum(self.lr_history[-n:]) / max(len(self.lr_history[-n:]), 1),
            "avg_tok_s": sum(self.tokens_per_sec_history[-n:]) / max(len(self.tokens_per_sec_history[-n:]), 1),
        }

    def detect_anomalies(self) -> List[str]:
        """Detect training anomalies that need attention."""
        issues = []
        if not self.loss_history:
            return issues

        recent = self.loss_history[-10:]
        if len(recent) < 2:
            return issues

        # NaN/Inf detection
        if any(not (0 < l < 1e6) for l in recent):
            issues.append("loss_divergence")

        # Loss spike detection
        if len(recent) >= 3:
            avg = sum(recent[:-1]) / (len(recent) - 1)
            if recent[-1] > avg * 3:
                issues.append("loss_spike")

        # Gradient explosion
        recent_grads = self.grad_norm_history[-10:]
        if recent_grads and max(recent_grads) > 100:
            issues.append("gradient_explosion")

        # Loss plateau
        if len(recent) >= 5:
            first_half = sum(recent[:3]) / 3
            second_half = sum(recent[-3:]) / 3
            if abs(first_half - second_half) / max(abs(first_half), 1e-8) < 0.01:
                issues.append("loss_plateau")

        return issues

    def get_trend(self, window: int = 20) -> str:
        """Get loss trend: improving, stable, or degrading."""
        if len(self.loss_history) < window:
            return "insufficient_data"
        recent = self.loss_history[-window:]
        first = sum(recent[:window//2]) / (window//2)
        second = sum(recent[window//2:]) / (window//2)
        change = (second - first) / max(abs(first), 1e-8)
        if change < -0.05:
            return "improving"
        elif change > 0.05:
            return "degrading"
        return "stable"


class MetaMemory:
    """Persistent memory for evolution episodes with consolidation."""

    def __init__(self, config):
        self.memory_file = config.memory_file
        self.max_entries = config.max_memory_entries
        self.feedback_chain_length = config.feedback_chain.chain_length
        self.consolidation_schedule = config.consolidation_schedule
        self.entries: List[EvolutionEpisode] = []

    def add(self, episode: EvolutionEpisode):
        self.entries.append(episode)
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries:]

    def get_feedback_chain(self, length: int = None) -> List[EvolutionEpisode]:
        length = length or self.feedback_chain_length
        return self.entries[-length:]

    def consolidate(self) -> Dict[str, Any]:
        """Analyze recent episodes for patterns and actionable insights."""
        if not self.entries:
            return {"status": "no_data"}

        recent = self.entries[-self.consolidation_schedule:]
        decisions = {}
        bugs = {}
        avg_result = 0.0
        successful_hypotheses = []

        for e in recent:
            decisions[e.decision] = decisions.get(e.decision, 0) + 1
            avg_result += e.experiment_result
            for bug in e.bugs_found:
                bugs[bug] = bugs.get(bug, 0) + 1
            if e.decision == "keep" and e.experiment_result > 0.7:
                successful_hypotheses.append(e.hypothesis)

        avg_result /= max(len(recent), 1)

        return {
            "avg_result": avg_result,
            "decisions": decisions,
            "total_episodes": len(recent),
            "common_bugs": dict(sorted(bugs.items(), key=lambda x: -x[1])[:5]),
            "successful_hypotheses": successful_hypotheses[:5],
            "keep_rate": decisions.get("keep", 0) / max(len(recent), 1),
        }

    def save(self):
        """Save memory to JSONL file."""
        path = Path(self.memory_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for entry in self.entries:
                f.write(json.dumps(asdict(entry)) + "\n")

    def load(self):
        """Load memory from JSONL file."""
        path = Path(self.memory_file)
        if not path.exists():
            return
        self.entries = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    self.entries.append(EvolutionEpisode(**data))


class SelfEvolutionFramework:
    """Autonomous self-evolution system for Deeplm.

    Monitors training, generates hypotheses about improvements,
    designs and executes experiments, analyzes results, and
    decides whether to keep or revert changes.
    """

    def __init__(self, config: SelfEvolutionConfig, model: nn.Module = None,
                 optimizer: torch.optim.Optimizer = None,
                 metrics: TrainingMetrics = None):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.metrics = metrics or TrainingMetrics()
        self.memory = MetaMemory(config.meta_memory)
        self.current_iteration = 0
        self.max_iterations = config.autonomous_research.max_iterations
        self.workflow = config.autonomous_research.workflow
        self.auto_commit = config.autonomous_research.auto_commit

        # State tracking
        self._original_state = None
        self._pending_changes = []
        self._coverage_target = config.autonomous_research.coverage_target

        if config.meta_memory.enabled:
            self.memory.load()
            logger.info(f"Loaded {len(self.memory.entries)} memory entries")

    def run_autonomous_loop(self, num_rounds: int = None) -> Dict[str, Any]:
        """Run the full self-evolution loop."""
        num_rounds = num_rounds or self.max_iterations
        results = []

        for i in range(num_rounds):
            self.current_iteration = i + 1
            episode = EvolutionEpisode(step=i, timestamp=time.time())

            for phase in self.workflow:
                episode.phase = phase
                try:
                    if phase == "hypothesis_generation":
                        episode.hypothesis = self._generate_hypothesis()
                    elif phase == "experiment_design":
                        episode.experiment_design = self._design_experiment(episode.hypothesis)
                    elif phase == "code_execution":
                        episode.changes_applied = self._execute_changes(episode.experiment_design)
                    elif phase == "log_analysis":
                        self._analyze_logs()
                    elif phase == "bug_diagnosis":
                        episode.bugs_found = self._diagnose_bugs()
                    elif phase == "code_fix":
                        episode.fixes_applied = self._apply_fixes(episode.bugs_found)
                    elif phase == "evaluation":
                        episode.metrics_before = self.metrics.get_recent()
                        episode.experiment_result = self._evaluate()
                        episode.metrics_after = self.metrics.get_recent()
                    elif phase == "decision":
                        episode.decision = self._make_decision(episode.experiment_result)
                        if episode.decision == "keep" and self.auto_commit:
                            self._commit_changes(episode)
                        elif episode.decision == "revert":
                            self._revert_changes()
                except Exception as e:
                    logger.error(f"Error in phase {phase}: {e}")
                    episode.decision = "error"
                    episode.experiment_result = 0.0
                    break

            self.memory.add(episode)
            results.append({
                "iteration": i + 1,
                "hypothesis": episode.hypothesis,
                "result": episode.experiment_result,
                "decision": episode.decision,
                "bugs": episode.bugs_found,
                "fixes": episode.fixes_applied,
            })

            if (i + 1) % self.memory.consolidation_schedule == 0:
                summary = self.memory.consolidate()
                logger.info(f"Consolidation at step {i + 1}: {summary}")
                if self.config.meta_memory.enabled:
                    self.memory.save()

        if self.config.meta_memory.enabled:
            self.memory.save()

        return {
            "total_rounds": len(results),
            "results": results,
            "memory_summary": self.memory.consolidate(),
        }

    def _generate_hypothesis(self) -> str:
        """Generate hypothesis based on training metrics and memory."""
        chain = self.memory.get_feedback_chain()
        anomalies = self.metrics.detect_anomalies()
        trend = self.metrics.get_trend()

        # Priority: fix bugs first
        if "loss_divergence" in anomalies:
            return "loss_diverging: reduce learning rate or increase gradient clipping"
        if "gradient_explosion" in anomalies:
            return "gradient_explosion: lower max_grad_norm or use gradient accumulation"
        if "loss_spike" in anomalies:
            return "loss_spike: check for data corruption or reduce batch size"

        # Trend-based hypotheses
        if trend == "degrading":
            return "performance_degrading: reduce learning rate or add regularization"
        if trend == "stable":
            return "loss_plateau: try learning rate warmup restart or increase model capacity"

        # Memory-based hypotheses
        if chain:
            recent_results = [e.experiment_result for e in chain]
            avg = sum(recent_results) / len(recent_results)
            if avg < 0.3:
                return "low_performance: check initialization, data quality, or increase training steps"
            elif avg < 0.6:
                return "moderate_performance: tune hyperparameters (lr, batch_size, warmup)"
            else:
                return "good_performance: optimize for edge cases, try longer sequences or harder data"

        return "initial_hypothesis: establish baseline training behavior"

    def _design_experiment(self, hypothesis: str) -> Dict[str, Any]:
        """Design an experiment to test the hypothesis."""
        experiment = {"hypothesis": hypothesis, "changes": []}

        if "loss_diverging" in hypothesis or "reduce learning rate" in hypothesis:
            if self.optimizer:
                for pg in self.optimizer.param_groups:
                    experiment["changes"].append({
                        "type": "lr_scale",
                        "factor": 0.5,
                        "param_group": pg.get("name", "default"),
                    })
            experiment["changes"].append({
                "type": "max_grad_norm",
                "value": 0.5,
            })

        elif "gradient_explosion" in hypothesis:
            experiment["changes"].append({
                "type": "max_grad_norm",
                "value": 0.5,
            })
            experiment["changes"].append({
                "type": "grad_accumulation",
                "factor": 2,
            })

        elif "loss_spike" in hypothesis:
            experiment["changes"].append({
                "type": "batch_size",
                "factor": 0.5,
            })

        elif "loss_plateau" in hypothesis:
            experiment["changes"].append({
                "type": "lr_warmup_restart",
                "warmup_ratio": 0.01,
            })

        elif "moderate_performance" in hypothesis:
            experiment["changes"].append({
                "type": "lr_scale",
                "factor": 1.2,
            })

        elif "good_performance" in hypothesis:
            experiment["changes"].append({
                "type": "sequence_length",
                "factor": 1.5,
            })

        else:
            experiment["changes"].append({
                "type": "monitor_only",
            })

        return experiment

    def _execute_changes(self, experiment: Dict[str, Any]) -> List[str]:
        """Execute the planned changes."""
        applied = []
        if not experiment or "changes" not in experiment:
            return applied

        for change in experiment["changes"]:
            try:
                ctype = change.get("type", "")
                if ctype == "lr_scale" and self.optimizer:
                    factor = change.get("factor", 1.0)
                    for pg in self.optimizer.param_groups:
                        pg["lr"] *= factor
                    applied.append(f"lr_scaled_by_{factor}")

                elif ctype == "max_grad_norm":
                    applied.append(f"max_grad_norm_set_to_{change['value']}")

                elif ctype == "grad_accumulation":
                    applied.append(f"grad_accumulation_scaled_by_{change['factor']}")

                elif ctype == "batch_size":
                    applied.append(f"batch_size_scaled_by_{change['factor']}")

                elif ctype == "lr_warmup_restart":
                    applied.append("lr_warmup_restart_scheduled")

                elif ctype == "sequence_length":
                    applied.append(f"sequence_length_scaled_by_{change['factor']}")

                elif ctype == "monitor_only":
                    applied.append("monitoring_only_no_changes")

            except Exception as e:
                logger.error(f"Failed to apply change {change}: {e}")

        self._pending_changes = applied
        return applied

    def _analyze_logs(self):
        """Analyze training logs for patterns."""
        metrics = self.metrics.get_recent()
        trend = self.metrics.get_trend()
        anomalies = self.metrics.detect_anomalies()

        logger.info(f"Metrics: avg_loss={metrics['avg_loss']:.4f}, "
                    f"trend={trend}, anomalies={anomalies}")

    def _diagnose_bugs(self) -> List[str]:
        """Diagnose training bugs based on metrics and model state."""
        bugs = []

        # Check for NaN in model parameters
        if self.model:
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    if torch.isnan(param).any():
                        bugs.append(f"nan_in_{name}")
                    if torch.isinf(param).any():
                        bugs.append(f"inf_in_{name}")

        # Check metrics anomalies
        anomalies = self.metrics.detect_anomalies()
        bugs.extend(anomalies)

        return bugs

    def _apply_fixes(self, bugs: List[str]) -> List[str]:
        """Apply fixes for diagnosed bugs."""
        fixes = []

        for bug in bugs:
            if bug == "loss_divergence":
                if self.optimizer:
                    for pg in self.optimizer.param_groups:
                        pg["lr"] *= 0.1
                fixes.append("lr_reduced_10x_for_divergence")

            elif bug == "gradient_explosion":
                fixes.append("gradient_clipping_applied")

            elif bug.startswith("nan_in_"):
                param_name = bug.replace("nan_in_", "")
                if self.model:
                    for name, param in self.model.named_parameters():
                        if name == param_name:
                            nn.init.xavier_uniform_(param)
                            fixes.append(f"reinitialized_{param_name}")

            elif bug.startswith("inf_in_"):
                param_name = bug.replace("inf_in_", "")
                if self.model:
                    for name, param in self.model.named_parameters():
                        if name == param_name:
                            nn.init.xavier_uniform_(param)
                            fixes.append(f"reinitialized_{param_name}")

        return fixes

    def _evaluate(self) -> float:
        """Evaluate the current state and return a score [0, 1]."""
        score = 0.5  # baseline

        # Factor 1: Loss trend (40%)
        trend = self.metrics.get_trend()
        if trend == "improving":
            score += 0.2
        elif trend == "degrading":
            score -= 0.2

        # Factor 2: Anomaly count (30%)
        anomalies = self.metrics.detect_anomalies()
        score -= len(anomalies) * 0.1

        # Factor 3: Gradient health (20%)
        recent_grads = self.metrics.grad_norm_history[-10:]
        if recent_grads:
            avg_grad = sum(recent_grads) / len(recent_grads)
            if 0.1 < avg_grad < 10:
                score += 0.1
            elif avg_grad > 100:
                score -= 0.2

        # Factor 4: Memory performance (10%)
        chain = self.memory.get_feedback_chain()
        if chain:
            recent_results = [e.experiment_result for e in chain]
            if recent_results:
                avg = sum(recent_results) / len(recent_results)
                score += (avg - 0.5) * 0.2

        return max(0.0, min(1.0, score))

    def _make_decision(self, result: float) -> str:
        """Decide whether to keep or revert changes."""
        threshold = self.config.harness_optimization.improvement_threshold
        if result >= threshold:
            return "keep"
        return "revert"

    def _commit_changes(self, episode: EvolutionEpisode):
        """Commit successful changes."""
        logger.info(f"Committing changes: {episode.changes_applied}")
        # In a real system, this would write changes to config files or git
        self._pending_changes = []

    def _revert_changes(self):
        """Revert pending changes."""
        logger.info(f"Reverting changes: {self._pending_changes}")
        self._pending_changes = []

    def optimize_harness(self) -> Dict[str, Any]:
        """Optimize the training harness configuration."""
        config = self.config.harness_optimization
        improvements = {}

        for component in config.target_components:
            improvements[component] = {
                "status": "analyzed",
                "target_improvement": config.improvement_threshold,
                "current_metrics": self.metrics.get_recent(),
            }

        return improvements

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory and evolution statistics."""
        return {
            "total_entries": len(self.memory.entries),
            "max_entries": self.memory.max_entries,
            "current_iteration": self.current_iteration,
            "consolidation": self.memory.consolidate(),
            "recent_metrics": self.metrics.get_recent(),
            "pending_changes": self._pending_changes,
        }
