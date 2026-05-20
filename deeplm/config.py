import yaml
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class TokenizerConfig:
    type: str = "BBPE"
    vocab_size: int = 32000
    pad_token: str = "<|pad|>"
    bos_token: str = "<|begin_of_sentence|>"
    eos_token: str = "<|end_of_sentence|>"
    unk_token: str = "<|unk|>"
    mask_token: str = "<|mask|>"
    special_tokens: Dict[str, str] = field(default_factory=dict)


@dataclass
class ArchitectureConfig:
    type: str = "decoder_only"
    num_layers: int = 10
    hidden_size: int = 512
    intermediate_size: int = 2048
    num_attention_heads: int = 8
    num_key_value_heads: int = 1
    head_dim: int = 128
    rope_head_dim: int = 64
    nope_head_dim: int = 64
    max_seq_length: int = 4096
    rope_theta: float = 50000.0


@dataclass
class MLAConfig:
    enabled: bool = True
    q_lora_rank: int = 192
    kv_lora_rank: int = 64
    qk_rope_head_dim: int = 64
    qk_nope_head_dim: int = 64
    v_head_dim: int = 128
    num_heads: int = 8
    kv_heads: int = 1
    o_groups: int = 4
    use_absorption_trick: bool = True
    decoupled_rope: bool = True


@dataclass
class RouterConfig:
    scoring_function: str = "sqrtsoftplus"
    noaux_tc: bool = True
    bias_update_speed: float = 0.001
    load_balance_tolerance: float = 0.1
    max_expert_capacity: float = 1.5


@dataclass
class ExpertAffinityConfig:
    enabled: bool = True
    memory_size: int = 1024
    decay_factor: float = 0.95


@dataclass
class MoEConfig:
    enabled: bool = True
    num_routed_experts: int = 4
    num_shared_experts: int = 1
    top_k: int = 2
    expert_intermediate_size: int = 768
    expert_activation: str = "swiglu"
    shared_expert_intermediate_size: int = 768
    router: RouterConfig = field(default_factory=RouterConfig)


@dataclass
class HyperConnectionsConfig:
    enabled: bool = True
    hc_mult: int = 4
    hc_dim: int = 384
    sinkhorn_iterations: int = 2
    sinkhorn_temperature: float = 0.1
    connection_types: List[str] = field(default_factory=lambda: ["skip", "identity", "transform", "gate"])
    initial_weights: Dict[str, float] = field(default_factory=lambda: {"identity": 0.6, "skip": 0.0, "transform": 0.2, "gate": 0.2})


@dataclass
class MTPConfig:
    enabled: bool = True
    num_mtp_layers: int = 2
    mtp_depth: int = 2
    mtp_hidden_size: int = 384
    mtp_loss_weight: float = 0.3
    mtp_positional_encoding: str = "rope"
    mtp_head: str = "tied"


@dataclass
class LinearAttentionConfig:
    type: str = "lightning_attention_v2"
    block_size: int = 256
    intra_block_type: str = "left_product"
    inter_block_type: str = "right_product"
    activation: str = "swish"
    use_tiling: bool = True


@dataclass
class HybridAttentionConfig:
    enabled: bool = True
    softmax_layers: List[int] = field(default_factory=lambda: [0, 4, 8])
    linear_layers: List[int] = field(default_factory=lambda: [1, 2, 3, 5, 6, 7, 9])
    linear_attention_config: LinearAttentionConfig = field(default_factory=LinearAttentionConfig)


@dataclass
class AutonomousResearchConfig:
    enabled: bool = True
    max_iterations: int = 100
    workflow: List[str] = field(default_factory=lambda: [
        "hypothesis_generation", "experiment_design", "code_execution",
        "log_analysis", "bug_diagnosis", "code_fix", "evaluation", "decision"
    ])
    auto_commit: bool = True
    coverage_target: float = 0.35


@dataclass
class HarnessOptimizationConfig:
    enabled: bool = True
    target_components: List[str] = field(default_factory=lambda: ["scaffold", "sampling_params", "workflow_strategy"])
    improvement_threshold: float = 0.30


@dataclass
class FeedbackChainConfig:
    enabled: bool = True
    chain_length: int = 5


@dataclass
class MetaMemoryConfig:
    enabled: bool = True
    memory_type: str = "short_term"
    memory_file: str = "deeplm_memory.jsonl"
    max_memory_entries: int = 10000
    feedback_chain: FeedbackChainConfig = field(default_factory=FeedbackChainConfig)
    consolidation_schedule: int = 100


@dataclass
class SelfEvolutionConfig:
    enabled: bool = True
    autonomous_research: AutonomousResearchConfig = field(default_factory=AutonomousResearchConfig)
    harness_optimization: HarnessOptimizationConfig = field(default_factory=HarnessOptimizationConfig)
    meta_memory: MetaMemoryConfig = field(default_factory=MetaMemoryConfig)


@dataclass
class AgentHarnessConfig:
    enabled: bool = True
    framework: str = "OpenClaw"
    build_autonomously: bool = True
    auto_buildable: List[str] = field(default_factory=lambda: ["skills", "memory", "guardrails", "evaluation", "mcp_servers"])


@dataclass
class PretrainingConfig:
    dataset: str = "FineWeb-Edu"
    max_tokens: int = 5_000_000_000
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 6.0e-4
    min_learning_rate: float = 6.0e-6
    lr_schedule: str = "cosine"
    warmup_steps: int = 150
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1.0e-8
    max_grad_norm: float = 1.0
    use_gradient_checkpointing: bool = True
    use_flash_attention: bool = True


@dataclass
class SFTConfig:
    dataset: str = "SmolTalk"
    epochs: int = 3
    steps: int = 3000
    batch_size: int = 4
    gradient_accumulation_steps: int = 2
    learning_rate: float = 1.0e-4
    lr_schedule: str = "cosine"
    warmup_steps: int = 100


@dataclass
class TrainingConfig:
    pretraining: PretrainingConfig = field(default_factory=PretrainingConfig)
    sft: SFTConfig = field(default_factory=SFTConfig)


@dataclass
class GenerationConfig:
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.05
    max_new_tokens: int = 1024
    do_sample: bool = True


@dataclass
class ThinkingModeConfig:
    enabled: bool = True
    think_token_start: str = "<|think_start|>"
    think_token_end: str = "<|think_end|>"
    max_think_tokens: int = 512
    think_temperature: float = 0.6


@dataclass
class InferenceConfig:
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    thinking_mode: ThinkingModeConfig = field(default_factory=ThinkingModeConfig)


@dataclass
class LMHeadConfig:
    type: str = "tied"
    bias: bool = False


@dataclass
class MTPHeadSubConfig:
    num_heads: int = 1
    type: str = "tied"


@dataclass
class SelfEvolutionHeadConfig:
    enabled: bool = True
    type: str = "separate"
    hidden_size: int = 384
    output_size: int = 128000


@dataclass
class OutputHeadsConfig:
    lm_head: LMHeadConfig = field(default_factory=LMHeadConfig)
    mtp_heads: MTPHeadSubConfig = field(default_factory=MTPHeadSubConfig)
    self_evolution_head: SelfEvolutionHeadConfig = field(default_factory=SelfEvolutionHeadConfig)


@dataclass
class DeeplmConfig:
    model_name: str = "Deeplm"
    version: str = "2.0.0"
    total_params: int = 108_000_000
    non_embedding_params: int = 91_600_000
    embedding_params: int = 16_400_000
    max_seq_length: int = 4096
    vocab_size: int = 32000
    dtype: str = "float32"

    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    architecture: ArchitectureConfig = field(default_factory=ArchitectureConfig)
    mla: MLAConfig = field(default_factory=MLAConfig)
    moe: MoEConfig = field(default_factory=MoEConfig)
    hyper_connections: HyperConnectionsConfig = field(default_factory=HyperConnectionsConfig)
    mtp: MTPConfig = field(default_factory=MTPConfig)
    hybrid_attention: HybridAttentionConfig = field(default_factory=HybridAttentionConfig)
    self_evolution: SelfEvolutionConfig = field(default_factory=SelfEvolutionConfig)
    agent_harness: AgentHarnessConfig = field(default_factory=AgentHarnessConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    output_heads: OutputHeadsConfig = field(default_factory=OutputHeadsConfig)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "DeeplmConfig":
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)

        def safe_get(d, key, default=None):
            return d.get(key, default) if d else default

        tokenizer_data = safe_get(data, "tokenizer", {})
        special_tokens = safe_get(tokenizer_data, "special_tokens", {})

        arch_data = safe_get(data, "architecture", {})
        mla_data = safe_get(data, "mla", {})

        moe_data = safe_get(data, "moe", {})
        router_data = safe_get(moe_data, "router", {})

        hc_data = safe_get(data, "hyper_connections", {})
        mtp_data = safe_get(data, "mtp", {})

        ha_data = safe_get(data, "hybrid_attention", {})
        la_data = safe_get(ha_data, "linear_attention_config", {})

        se_data = safe_get(data, "self_evolution", {})
        ar_data = safe_get(se_data, "autonomous_research", {})
        ho_data = safe_get(se_data, "harness_optimization", {})
        mm_data = safe_get(se_data, "meta_memory", {})
        fc_data = safe_get(mm_data, "feedback_chain", {})

        agent_data = safe_get(data, "agent_harness", {})

        training_data = safe_get(data, "training", {})
        pretrain_data = safe_get(training_data, "pretraining", {})
        sft_data = safe_get(training_data, "sft", {})

        inference_data = safe_get(data, "inference", {})
        gen_data = safe_get(inference_data, "generation", {})
        think_data = safe_get(inference_data, "thinking_mode", {})

        output_heads_data = safe_get(data, "output_heads", {})
        lm_head_data = safe_get(output_heads_data, "lm_head", {})
        mtp_heads_data = safe_get(output_heads_data, "mtp_heads", {})
        se_head_data = safe_get(output_heads_data, "self_evolution_head", {})

        return cls(
            model_name=safe_get(data, "model_name", "Deeplm"),
            version=safe_get(data, "version", "2.0.0"),
            total_params=safe_get(data, "total_params", 108_000_000),
            vocab_size=safe_get(data, "vocab_size", 32000),
            max_seq_length=safe_get(data, "max_seq_length", 4096),
            dtype=safe_get(data, "dtype", "float32"),
            tokenizer=TokenizerConfig(
                type=safe_get(tokenizer_data, "type", "BBPE"),
                vocab_size=safe_get(tokenizer_data, "vocab_size", 32000),
                pad_token=safe_get(tokenizer_data, "pad_token", "<|pad|>"),
                bos_token=safe_get(tokenizer_data, "bos_token", "<|begin_of_sentence|>"),
                eos_token=safe_get(tokenizer_data, "eos_token", "<|end_of_sentence|>"),
                unk_token=safe_get(tokenizer_data, "unk_token", "<|unk|>"),
                mask_token=safe_get(tokenizer_data, "mask_token", "<|mask|>"),
                special_tokens=special_tokens,
            ),
            architecture=ArchitectureConfig(
                type=safe_get(arch_data, "type", "decoder_only"),
                num_layers=safe_get(arch_data, "num_layers", 10),
                hidden_size=safe_get(arch_data, "hidden_size", 512),
                intermediate_size=safe_get(arch_data, "intermediate_size", 2048),
                num_attention_heads=safe_get(arch_data, "num_attention_heads", 8),
                num_key_value_heads=safe_get(arch_data, "num_key_value_heads", 1),
                head_dim=safe_get(arch_data, "head_dim", 128),
                rope_head_dim=safe_get(arch_data, "rope_head_dim", 64),
                nope_head_dim=safe_get(arch_data, "nope_head_dim", 64),
                max_seq_length=safe_get(arch_data, "max_seq_length", 4096),
                rope_theta=safe_get(arch_data, "rope_theta", 50000.0),
            ),
            mla=MLAConfig(
                enabled=safe_get(mla_data, "enabled", True),
                q_lora_rank=safe_get(mla_data, "q_lora_rank", 192),
                kv_lora_rank=safe_get(mla_data, "kv_lora_rank", 64),
                qk_rope_head_dim=safe_get(mla_data, "qk_rope_head_dim", 64),
                qk_nope_head_dim=safe_get(mla_data, "qk_nope_head_dim", 64),
                v_head_dim=safe_get(mla_data, "v_head_dim", 128),
                num_heads=safe_get(mla_data, "num_heads", 8),
                kv_heads=safe_get(mla_data, "kv_heads", 1),
                o_groups=safe_get(mla_data, "o_groups", 4),
                use_absorption_trick=safe_get(mla_data, "use_absorption_trick", True),
                decoupled_rope=safe_get(mla_data, "decoupled_rope", True),
            ),
            moe=MoEConfig(
                enabled=safe_get(moe_data, "enabled", True),
                num_routed_experts=safe_get(moe_data, "num_routed_experts", 4),
                num_shared_experts=safe_get(moe_data, "num_shared_experts", 1),
                top_k=safe_get(moe_data, "top_k", 2),
                expert_intermediate_size=safe_get(moe_data, "expert_intermediate_size", 768),
                expert_activation=safe_get(moe_data, "expert_activation", "swiglu"),
                shared_expert_intermediate_size=safe_get(moe_data, "shared_expert_intermediate_size", 768),
                router=RouterConfig(
                    scoring_function=safe_get(router_data, "scoring_function", "sqrtsoftplus"),
                    noaux_tc=safe_get(router_data, "noaux_tc", True),
                    bias_update_speed=safe_get(router_data, "bias_update_speed", 0.001),
                    load_balance_tolerance=safe_get(router_data, "load_balance_tolerance", 0.1),
                    max_expert_capacity=safe_get(router_data, "max_expert_capacity", 1.5),
                ),
            ),
            hyper_connections=HyperConnectionsConfig(
                enabled=safe_get(hc_data, "enabled", True),
                hc_mult=safe_get(hc_data, "hc_mult", 4),
                hc_dim=safe_get(hc_data, "hc_dim", 384),
                sinkhorn_iterations=safe_get(hc_data, "sinkhorn_iterations", 2),
                sinkhorn_temperature=safe_get(hc_data, "sinkhorn_temperature", 0.1),
                connection_types=safe_get(hc_data, "connection_types", ["skip", "identity", "transform", "gate"]),
                initial_weights=safe_get(hc_data, "initial_weights", {"identity": 0.6, "skip": 0.0, "transform": 0.2, "gate": 0.2}),
            ),
            mtp=MTPConfig(
                enabled=safe_get(mtp_data, "enabled", True),
                num_mtp_layers=safe_get(mtp_data, "num_mtp_layers", 2),
                mtp_depth=safe_get(mtp_data, "mtp_depth", 2),
                mtp_hidden_size=safe_get(mtp_data, "mtp_hidden_size", 384),
                mtp_loss_weight=safe_get(mtp_data, "mtp_loss_weight", 0.3),
                mtp_positional_encoding=safe_get(mtp_data, "mtp_positional_encoding", "rope"),
                mtp_head=safe_get(mtp_data, "mtp_head", "tied"),
            ),
            hybrid_attention=HybridAttentionConfig(
                enabled=safe_get(ha_data, "enabled", True),
                softmax_layers=safe_get(ha_data, "softmax_layers", [0, 4, 8]),
                linear_layers=safe_get(ha_data, "linear_layers", [1, 2, 3, 5, 6, 7, 9]),
                linear_attention_config=LinearAttentionConfig(
                    type=safe_get(la_data, "type", "lightning_attention_v2"),
                    block_size=safe_get(la_data, "block_size", 256),
                    intra_block_type=safe_get(la_data, "intra_block_type", "left_product"),
                    inter_block_type=safe_get(la_data, "inter_block_type", "right_product"),
                    activation=safe_get(la_data, "activation", "swish"),
                    use_tiling=safe_get(la_data, "use_tiling", True),
                ),
            ),
            self_evolution=SelfEvolutionConfig(
                enabled=safe_get(se_data, "enabled", True),
                autonomous_research=AutonomousResearchConfig(
                    enabled=safe_get(ar_data, "enabled", True),
                    max_iterations=safe_get(ar_data, "max_iterations", 100),
                    workflow=safe_get(ar_data, "workflow", [
                        "hypothesis_generation", "experiment_design", "code_execution",
                        "log_analysis", "bug_diagnosis", "code_fix", "evaluation", "decision"
                    ]),
                    auto_commit=safe_get(ar_data, "auto_commit", True),
                    coverage_target=safe_get(ar_data, "coverage_target", 0.35),
                ),
                harness_optimization=HarnessOptimizationConfig(
                    enabled=safe_get(ho_data, "enabled", True),
                    target_components=safe_get(ho_data, "target_components", ["scaffold", "sampling_params", "workflow_strategy"]),
                    improvement_threshold=safe_get(ho_data, "improvement_threshold", 0.30),
                ),
                meta_memory=MetaMemoryConfig(
                    enabled=safe_get(mm_data, "enabled", True),
                    memory_type=safe_get(mm_data, "memory_type", "short_term"),
                    memory_file=safe_get(mm_data, "memory_file", "deeplm_memory.jsonl"),
                    max_memory_entries=safe_get(mm_data, "max_memory_entries", 10000),
                    feedback_chain=FeedbackChainConfig(
                        enabled=safe_get(fc_data, "enabled", True),
                        chain_length=safe_get(fc_data, "chain_length", 5),
                    ),
                    consolidation_schedule=safe_get(mm_data, "consolidation_schedule", 100),
                ),
            ),
            agent_harness=AgentHarnessConfig(
                enabled=safe_get(agent_data, "enabled", True),
                framework=safe_get(agent_data, "framework", "OpenClaw"),
                build_autonomously=safe_get(agent_data, "build_autonomously", True),
                auto_buildable=safe_get(agent_data, "auto_buildable", ["skills", "memory", "guardrails", "evaluation", "mcp_servers"]),
            ),
            training=TrainingConfig(
                pretraining=PretrainingConfig(
                    dataset=safe_get(pretrain_data, "dataset", "FineWeb-Edu"),
                    max_tokens=safe_get(pretrain_data, "max_tokens", 5_000_000_000),
                    batch_size=safe_get(pretrain_data, "batch_size", 8),
                    gradient_accumulation_steps=safe_get(pretrain_data, "gradient_accumulation_steps", 4),
                    learning_rate=safe_get(pretrain_data, "learning_rate", 6.0e-4),
                    min_learning_rate=safe_get(pretrain_data, "min_learning_rate", 6.0e-6),
                    lr_schedule=safe_get(pretrain_data, "lr_schedule", "cosine"),
                    warmup_steps=safe_get(pretrain_data, "warmup_steps", 150),
                    weight_decay=safe_get(pretrain_data, "weight_decay", 0.1),
                    adam_beta1=safe_get(pretrain_data, "adam_beta1", 0.9),
                    adam_beta2=safe_get(pretrain_data, "adam_beta2", 0.95),
                    adam_epsilon=safe_get(pretrain_data, "adam_epsilon", 1.0e-8),
                    max_grad_norm=safe_get(pretrain_data, "max_grad_norm", 1.0),
                    use_gradient_checkpointing=safe_get(pretrain_data, "use_gradient_checkpointing", True),
                    use_flash_attention=safe_get(pretrain_data, "use_flash_attention", True),
                ),
                sft=SFTConfig(
                    dataset=safe_get(sft_data, "dataset", "SmolTalk"),
                    epochs=safe_get(sft_data, "epochs", 3),
                    steps=safe_get(sft_data, "steps", 3000),
                    batch_size=safe_get(sft_data, "batch_size", 4),
                    gradient_accumulation_steps=safe_get(sft_data, "gradient_accumulation_steps", 2),
                    learning_rate=safe_get(sft_data, "learning_rate", 1.0e-4),
                    lr_schedule=safe_get(sft_data, "lr_schedule", "cosine"),
                    warmup_steps=safe_get(sft_data, "warmup_steps", 100),
                ),
            ),
            inference=InferenceConfig(
                generation=GenerationConfig(
                    temperature=safe_get(gen_data, "temperature", 0.7),
                    top_k=safe_get(gen_data, "top_k", 50),
                    top_p=safe_get(gen_data, "top_p", 0.9),
                    repetition_penalty=safe_get(gen_data, "repetition_penalty", 1.05),
                    max_new_tokens=safe_get(gen_data, "max_new_tokens", 1024),
                    do_sample=safe_get(gen_data, "do_sample", True),
                ),
                thinking_mode=ThinkingModeConfig(
                    enabled=safe_get(think_data, "enabled", True),
                    think_token_start=safe_get(think_data, "think_token_start", "<|think_start|>"),
                    think_token_end=safe_get(think_data, "think_token_end", "<|think_end|>"),
                    max_think_tokens=safe_get(think_data, "max_think_tokens", 512),
                    think_temperature=safe_get(think_data, "think_temperature", 0.6),
                ),
            ),
            output_heads=OutputHeadsConfig(
                lm_head=LMHeadConfig(
                    type=safe_get(lm_head_data, "type", "tied"),
                    bias=safe_get(lm_head_data, "bias", False),
                ),
                mtp_heads=MTPHeadSubConfig(
                    num_heads=safe_get(mtp_heads_data, "num_heads", 1),
                    type=safe_get(mtp_heads_data, "type", "tied"),
                ),
            self_evolution_head=SelfEvolutionHeadConfig(
                enabled=safe_get(se_head_data, "enabled", True),
                type=safe_get(se_head_data, "type", "separate"),
                hidden_size=safe_get(se_head_data, "hidden_size", 512),
                output_size=safe_get(se_head_data, "output_size", 32000),
                ),
            ),
        )
