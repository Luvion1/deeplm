"""
Training Control Plane — unified interface for all training parameters.

The AutoTuner outputs TrainingControl objects that adjust:
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
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class HyperparamControl:
    """Core hyperparameter multipliers."""
    lr_mult: float = 1.0
    gn_mult: float = 1.0
    wd_mult: float = 1.0
    mom_boost: float = 0.0
    mom_decay: float = 0.98


@dataclass
class DatasetControl:
    """Dataset routing and curriculum control."""
    # Category weights for weighted sampling
    category_weights: dict = field(default_factory=lambda: {
        "grammar": 1.0, "dialog": 1.0, "instruction": 1.0,
        "reasoning": 1.0, "code": 1.0, "creative": 1.0,
        "academic": 1.0, "summarization": 1.0,
    })
    # Active categories (None = all active)
    active_categories: Optional[set] = None
    # Difficulty level: 0.0 (easiest) to 1.0 (hardest)
    difficulty: float = 0.5
    # Mix strategy: "balanced", "explore", "exploit", "curriculum"
    mix_strategy: str = "balanced"
    # Per-source dataset weights (for multi-source training)
    source_weights: dict = field(default_factory=dict)
    # Active dataset sources (None = all)
    active_sources: Optional[set] = None
    # Sequence length multiplier
    seq_len_mult: float = 1.0
    # Token budget per step
    token_budget: Optional[int] = None


@dataclass
class BatchControl:
    """Batch size and throughput control."""
    # Batch size multiplier
    batch_mult: float = 1.0
    # Gradient accumulation steps
    grad_accum: int = 1
    # Max gradient accumulation
    max_grad_accum: int = 32
    # Dynamic batch sizing enabled
    dynamic_batch: bool = False
    # Target tokens per batch (for dynamic sizing)
    target_tokens: Optional[int] = None


@dataclass
class PrecisionControl:
    """Numerical precision control."""
    # dtype: "fp32", "fp16", "bf16", "fp8"
    dtype: str = "bf16"
    # Loss scale for mixed precision
    loss_scale: float = 65536.0
    # Grad scale factor
    grad_scale: float = 1.0
    # Enable gradient scaling
    grad_scaling: bool = True
    # Enable dynamic loss scaling
    dynamic_loss_scale: bool = True


@dataclass
class ModelControl:
    """Model architecture and behavior control."""
    # Expert top-k for MoE routing
    expert_topk: int = 2
    # Layer dropout rate (stochastic depth)
    layer_drop_rate: float = 0.0
    # Attention dropout
    attention_dropout: float = 0.0
    # Hidden layer dropout
    hidden_dropout: float = 0.0
    # Activation function: "gelu", "relu", "swiglu", "silu"
    activation: str = "swiglu"
    # Enable layer skipping (for faster training)
    layer_skip: bool = False
    # Layer skip probability
    layer_skip_prob: float = 0.0
    # Attention head pruning ratio
    head_prune_ratio: float = 0.0


@dataclass
class RegularizationControl:
    """Regularization parameters."""
    # Dropout rate (general)
    dropout: float = 0.0
    # Weight decay multiplier
    weight_decay_mult: float = 1.0
    # Label smoothing
    label_smoothing: float = 0.0
    # Gradient penalty coefficient
    grad_penalty: float = 0.0
    # Activation checkpointing (for memory savings)
    activation_checkpointing: bool = False
    # Gradient clipping strategy: "norm", "value", "adaptive"
    clip_strategy: str = "norm"


@dataclass
class ScheduleControl:
    """Learning rate schedule control."""
    # Warmup ratio (fraction of total steps)
    warmup_ratio: float = 0.03
    # Decay type: "cosine", "linear", "exponential", "constant"
    decay_type: str = "cosine"
    # Number of cosine restarts (for SGDR)
    cosine_restarts: int = 0
    # LR plateau patience (steps before reducing LR)
    lr_plateau_patience: int = 500
    # Min LR ratio (fraction of base LR)
    min_lr_ratio: float = 0.1
    # Enable cyclical LR
    cyclical_lr: bool = False
    # Cyclical LR amplitude
    cyclical_amplitude: float = 0.05


@dataclass
class MoEControl:
    """Mixture of Experts specific control."""
    # Routing temperature (lower = sharper routing)
    routing_temperature: float = 1.0
    # Load balance loss weight
    load_balance_weight: float = 0.01
    # Expert capacity factor
    expert_capacity_factor: float = 1.0
    # Auxiliary loss weight
    aux_loss_weight: float = 0.0
    # Expert dropout
    expert_dropout: float = 0.0
    # Enable expert gating
    expert_gating: bool = True


@dataclass
class MTPControl:
    """Multi-Token Prediction specific control."""
    # MTP depth (number of future tokens to predict)
    depth: int = 2
    # MTP loss weight
    loss_weight: float = 0.1
    # Enable MTP
    enabled: bool = True
    # MTP dropout
    dropout: float = 0.0


@dataclass
class OptimizerControl:
    """Optimizer-specific parameters."""
    # Adam beta1
    beta1: float = 0.9
    # Adam beta2
    beta2: float = 0.999
    # Adam epsilon
    eps: float = 1e-8
    # Base weight decay
    weight_decay: float = 0.01
    # Enable gradient accumulation
    grad_accum_enabled: bool = True


@dataclass
class TrainingControl:
    """Unified training control plane.

    All parameters that the AutoTuner can adjust during training.
    Each sub-control manages a specific domain.
    """
    # Core hyperparameters
    hyperparams: HyperparamControl = field(default_factory=HyperparamControl)

    # Dataset routing
    dataset: DatasetControl = field(default_factory=DatasetControl)

    # Batch & throughput
    batch: BatchControl = field(default_factory=BatchControl)

    # Precision
    precision: PrecisionControl = field(default_factory=PrecisionControl)

    # Model architecture
    model: ModelControl = field(default_factory=ModelControl)

    # Regularization
    regularization: RegularizationControl = field(default_factory=RegularizationControl)

    # Learning schedule
    schedule: ScheduleControl = field(default_factory=ScheduleControl)

    # MoE specific
    moe: MoEControl = field(default_factory=MoEControl)

    # MTP specific
    mtp: MTPControl = field(default_factory=MTPControl)

    # Optimizer
    optimizer: OptimizerControl = field(default_factory=OptimizerControl)

    # Metadata
    step: int = 0
    phase: str = "warmup"
    mode: str = "auto"  # auto, manual, explore

    def to_dict(self) -> dict:
        """Convert to flat dict for serialization."""
        return {
            "lr_mult": self.hyperparams.lr_mult,
            "gn_mult": self.hyperparams.gn_mult,
            "wd_mult": self.hyperparams.wd_mult,
            "mom_boost": self.hyperparams.mom_boost,
            "category_weights": self.dataset.category_weights,
            "difficulty": self.dataset.difficulty,
            "mix_strategy": self.dataset.mix_strategy,
            "batch_mult": self.batch.batch_mult,
            "grad_accum": self.batch.grad_accum,
            "dtype": self.precision.dtype,
            "loss_scale": self.precision.loss_scale,
            "expert_topk": self.model.expert_topk,
            "layer_drop_rate": self.model.layer_drop_rate,
            "dropout": self.regularization.dropout,
            "warmup_ratio": self.schedule.warmup_ratio,
            "decay_type": self.schedule.decay_type,
            "routing_temperature": self.moe.routing_temperature,
            "mtp_depth": self.mtp.depth,
            "mtp_weight": self.mtp.loss_weight,
            "beta1": self.optimizer.beta1,
            "beta2": self.optimizer.beta2,
            "step": self.step,
            "phase": self.phase,
            "mode": self.mode,
        }

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Training Control (step={self.step}, phase={self.phase}, mode={self.mode})",
            f"  Hyperparams: LR×{self.hyperparams.lr_mult:.2f} GN×{self.hyperparams.gn_mult:.2f} "
            f"WD×{self.hyperparams.wd_mult:.2f} mom={self.hyperparams.mom_boost:.2f}",
            f"  Dataset: difficulty={self.dataset.difficulty:.2f} "
            f"strategy={self.dataset.mix_strategy}",
            f"  Batch: mult={self.batch.batch_mult:.2f} "
            f"grad_accum={self.batch.grad_accum}",
            f"  Precision: {self.precision.dtype} "
            f"loss_scale={self.precision.loss_scale:.0f}",
            f"  Model: topk={self.model.expert_topk} "
            f"layer_drop={self.model.layer_drop_rate:.2f}",
            f"  Regularization: dropout={self.regularization.dropout:.3f} "
            f"clip={self.regularization.clip_strategy}",
            f"  Schedule: warmup={self.schedule.warmup_ratio:.2f} "
            f"decay={self.schedule.decay_type}",
            f"  MoE: temp={self.moe.routing_temperature:.2f} "
            f"capacity={self.moe.expert_capacity_factor:.2f}",
            f"  MTP: depth={self.mtp.depth} weight={self.mtp.loss_weight:.2f}",
            f"  Optimizer: β1={self.optimizer.beta1} β2={self.optimizer.beta2} "
            f"wd={self.optimizer.weight_decay}",
        ]
        return "\n".join(lines)
