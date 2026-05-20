"""
Curriculum Router — selects dataset category mix based on training phase + signals.

Works standalone or integrates with AutoTuner. No subagent dependency.
"""
import math
import random
from collections import defaultdict, Counter, deque
from copy import deepcopy

from torch.utils.data import DataLoader


# ── Phase → category weight presets ──
PHASE_PRESETS = {
    "warmup": {
        "grammar": 3.0,
        "dialog": 2.0,
        "instruction": 2.0,
        "creative": 1.0,
        "reasoning": 0.5,
        "code": 0.5,
        "academic": 0.3,
        "summarization": 0.3,
    },
    "exploration": {
        "reasoning": 3.0,
        "code": 2.5,
        "instruction": 2.0,
        "grammar": 1.5,
        "creative": 1.5,
        "dialog": 1.0,
        "academic": 0.5,
        "summarization": 0.5,
    },
    "balanced": {
        "reasoning": 2.5,
        "code": 2.0,
        "creative": 2.0,
        "academic": 1.5,
        "instruction": 1.5,
        "grammar": 1.0,
        "summarization": 1.0,
        "dialog": 1.0,
    },
    "exploitation": {
        "reasoning": 3.0,
        "summarization": 2.5,
        "academic": 2.0,
        "code": 1.5,
        "creative": 1.0,
        "instruction": 0.8,
        "grammar": 0.5,
        "dialog": 0.3,
    },
}

ALL_CATEGORIES = list(PHASE_PRESETS["balanced"].keys())


class CurriculumRouter:
    """Routes dataset selection based on training phase and signals.

    Maintains category weights that change dynamically during training.
    Can operate in manual/explore mode where it probes alternative mixes.
    """

    def __init__(self, registry=None):
        self.registry = registry
        self._phase = "warmup"
        self._weights = dict(PHASE_PRESETS["warmup"])
        self._active_categories = set(ALL_CATEGORIES)
        self._mode = "normal"  # normal | explore | probe
        self._mode_until = 0
        self._explore_counter = 0
        self._prev_weights = None

        # Per-category loss tracking
        self._cat_losses = defaultdict(lambda: deque(maxlen=500))
        self._cat_counts = Counter()

        # Manual override support
        self._manual_weights = None

    # ── Phase management ──

    def set_phase(self, phase: str, step: int):
        if phase != self._phase:
            self._phase = phase
            preset = PHASE_PRESETS.get(phase, PHASE_PRESETS["balanced"])
            if self._manual_weights is None:
                self._weights = dict(preset)

    def get_phase(self) -> str:
        return self._phase

    # ── Weight management ──

    def get_weights(self):
        if self._manual_weights is not None:
            return dict(self._manual_weights)
        return dict(self._weights)

    def set_manual_weights(self, weights: dict):
        """Override category weights manually."""
        self._manual_weights = dict(weights)

    def clear_manual_weights(self):
        self._manual_weights = None

    def adjust_weight(self, category: str, delta: float):
        """Adjust a single category weight."""
        if category in self._weights:
            self._weights[category] = max(0.1, self._weights[category] + delta)

    def normalize_weights(self):
        total = sum(self._weights.values())
        if total > 0:
            for k in self._weights:
                self._weights[k] /= total

    # ── Category activation ──

    def activate_categories(self, cats):
        if isinstance(cats, str):
            cats = {cats}
        self._active_categories = set(cats)

    def get_active_categories(self):
        return list(self._active_categories)

    # ── Record per-category loss ──

    def record_loss(self, category: str, loss: float, step: int):
        if category:
            self._cat_losses[category].append(loss)
            self._cat_counts[category] += 1

    def get_category_loss(self, category: str):
        losses = self._cat_losses.get(category, [])
        if len(losses) >= 10:
            return sum(losses) / len(losses)
        return None

    def get_category_trend(self, category: str):
        """Return recent loss trend: -1 (improving), 0 (flat), +1 (degrading)."""
        losses = list(self._cat_losses.get(category, []))
        if len(losses) < 20:
            return 0
        half = len(losses) // 2
        first_half = sum(losses[:half]) / half
        second_half = sum(losses[half:2 * half]) / half
        diff = second_half - first_half
        if diff < -0.01:
            return -1
        elif diff > 0.01:
            return 1
        return 0

    # ── Manual explore mode ──

    def explore(self, step: int, signals: dict = None):
        """Probe alternative category mixes.

        Called manually (no subagent). Tweaks weights based on
        available signals and checks per-category performance.
        """
        if signals is None:
            signals = {}

        # Determine which categories to boost/dampen
        for cat in ALL_CATEGORIES:
            trend = self.get_category_trend(cat)
            if trend == -1:
                self._weights[cat] = min(5.0, self._weights[cat] * 1.1)
            elif trend == 1:
                self._weights[cat] = max(0.3, self._weights[cat] * 0.85)

        # Phase-specific exploration
        if self._phase == "exploration":
            self._weights["reasoning"] = min(5.0, self._weights["reasoning"] * 1.05)
        elif self._phase == "exploitation":
            self._weights["reasoning"] = min(4.0, self._weights["reasoning"] * 1.03)
            self._weights["summarization"] = min(4.0, self._weights["summarization"] * 1.05)

        # Plateau recovery: boost diversity
        if signals.get("plateau"):
            for cat in ALL_CATEGORIES:
                if cat not in ("reasoning", "code", "creative"):
                    self._weights[cat] = min(3.0, self._weights[cat] * 1.15)

        # Overfit: reinforce grammar + instruction
        if signals.get("overfit"):
            self._weights["grammar"] = min(4.0, self._weights["grammar"] * 1.2)
            self._weights["instruction"] = min(4.0, self._weights["instruction"] * 1.15)

        # Sick layers: simplify
        if signals.get("sick"):
            for cat in ("reasoning", "code"):
                self._weights[cat] = max(0.5, self._weights[cat] * 0.8)

        self.normalize_weights()
        self._explore_counter += 1

    # ── Build dataloader from categorized sources ──

    def build_dataloader(
        self,
        dataset_source,
        tokenizer,
        max_seq_length,
        bos, eos, pad,
        cache_dir="/tmp/token_cache",
        bucket_size=64,
        batch_size=8,
        num_workers=4,
        pin_memory=True,
        seed=42,
        epoch_size=None,
    ):
        """Build a DataLoader respecting current category weights.

        dataset_source: CategorizedDataset or (texts, category_map) tuple.
        """
        from .data_pipeline import TokenCache, BucketDataset, WeightedBucketSampler

        if isinstance(dataset_source, tuple):
            texts, category_map = dataset_source
        else:
            texts = dataset_source.texts
            category_map = dataset_source.category_map

        if not texts:
            return None

        cache = TokenCache(cache_dir, tokenizer, max_seq_length, bos, eos, pad)
        dataset = BucketDataset(texts, cache, bucket_size=bucket_size)

        # Map categories from flat index to bucket
        bucket_categories = {}
        for flat_idx, info in dataset._text_map.items():
            b_idx = info["bucket"]
            cat = category_map[flat_idx] if flat_idx < len(category_map) else "other"
            bucket_categories.setdefault(b_idx, cat)

        bucket_counts = dataset.get_buckets()

        sampler = WeightedBucketSampler(
            bucket_counts=bucket_counts,
            bucket_ids={idx: info["bucket"] for idx, info in dataset._text_map.items()},
            bucket_categories=bucket_categories,
            category_weights=self.get_weights(),
            epoch_size=epoch_size or len(dataset),
            seed=seed,
        )

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
            prefetch_factor=4,
            persistent_workers=num_workers > 0,
        )
        return loader

    # ── Stats ──

    def stats(self):
        cat_losses = {}
        for cat in ALL_CATEGORIES:
            l = self.get_category_loss(cat)
            if l is not None:
                cat_losses[cat] = round(l, 4)
        return {
            "phase": self._phase,
            "mode": self._mode,
            "active_categories": len(self._active_categories),
            "weights": {k: round(v, 2) for k, v in self.get_weights().items() if v > 0.1},
            "cat_losses": cat_losses,
            "cat_counts": dict(self._cat_counts.most_common(10)),
            "explore_count": self._explore_counter,
        }



