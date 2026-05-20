
Konfigurasi Model Deeplm

```yaml
# =============================================================================
# Deeplm Configuration File (v1.0)
# Model   : Deeplm (Deepseek 4 Pro Max Think / Kimi K2.6 High Think /
#           MiniMax M2.7 High Think Inspired)
# Total Params : ~108.5M
# Date    : 2026-05-15
# =============================================================================

model_name: "Deeplm"
version: "1.0.0"
total_params: 108_500_000
non_embedding_params: 70_100_000
embedding_params: 38_400_000
max_seq_length: 2048
vocab_size: 128000
dtype: "float32"  # Wajib karena Hyper-Connections rentan NaN di bf16/fp16


# =============================================================================
# 1. Tokenizer Configuration
# =============================================================================
tokenizer:
  type: "BBPE"  # Byte-level BPE
  vocab_size: 128000
  pad_token: "<|pad|>"
  bos_token: "<|begin_of_sentence|>"
  eos_token: "<|end_of_sentence|>"
  unk_token: "<|unk|>"
  mask_token: "<|mask|>"
  special_tokens:
    system: "<|system|>"
    user: "<|user|>"
    assistant: "<|assistant|>"
    tool_call: "<|tool_call|>"
    tool_response: "<|tool_response|>"
    think_start: "<|think_start|>"
    think_end: "<|think_end|>"
    self_evolve: "<|self_evolve|>"  # Token khusus untuk Self-Evolution
    memory_write: "<|memory_write|>"
    memory_read: "<|memory_read|>"


# =============================================================================
# 2. Transformer Core Architecture
# =============================================================================
architecture:
  type: "decoder_only"
  num_layers: 8  # Semua layer MoE kecuali layer pertama (opsional dense)
  hidden_size: 384
  intermediate_size: 1536  # 4x hidden_size (untuk SwiGLU)
  num_attention_heads: 8  # Jumlah query head
  num_key_value_heads: 1  # Multi-Query Attention (MQA) — hanya 1 KV head
  head_dim: 96  # Total dimensi per head (32 RoPE + 64 NoPE)
  rope_head_dim: 32  # Dimensi yang dirotasi RoPE
  nope_head_dim: 64  # Dimensi non-RoPE
  max_seq_length: 2048
  rope_theta: 10000.0  # Base theta RoPE (dapat ditingkatkan ke 1M untuk konteks panjang)


# =============================================================================
# 3. Multi-head Latent Attention (MLA) Configuration
#    Sumber: DeepSeek V4 & Kimi K2.6
#    Fungsi: Kompresi KV Cache untuk efisiensi memori
# =============================================================================
mla:
  enabled: true
  q_lora_rank: 192  # Rank kompresi Query (dense: hidden_size)
  kv_lora_rank: 128  # Rank kompresi latent KV (jantung MLA)
  qk_rope_head_dim: 32  # Dimensi Q/K yang dikenai RoPE
  qk_nope_head_dim: 64  # Dimensi Q/K tanpa RoPE (content-only)
  v_head_dim: 96  # Dimensi Value
  num_heads: 8  # Jumlah query head
  kv_heads: 1  # Hanya 1 KV head (MQA) → dikompresi lebih jauh
  # Rumus KV Cache saving: ~ (num_heads / kv_heads) * (hidden / kv_lora_rank)
  #                      ≈ 8 * (384/128) = 24x lebih kecil dari MHA penuh
  o_groups: 4  # Jumlah grup output projection (mengurangi parameter output)
  use_absorption_trick: true  # Pre-komputasi W_UK * W_UV agar inference lebih cepat
  decoupled_rope: true  # RoPE dipisah dari content path (ala DeepSeek V4)


# =============================================================================
# 4. Mixture of Experts (MoE) Configuration
#    Sumber: DeepSeek V4, Kimi K2.6, MiniMax M2.7
#    Fungsi: Kapasitas besar dengan komputasi aktif kecil
# =============================================================================
moe:
  enabled: true
  num_routed_experts: 4  # Jumlah routed expert per MoE layer
  num_shared_experts: 1  # Shared expert (selalu aktif) — konsep dari DeepSeek & Kimi
  top_k: 2  # Setiap token hanya mengaktifkan 2 dari 4 routed expert
  expert_intermediate_size: 768  # Dimensi inner FFN per expert (2x hidden_size)
  expert_activation: "swiglu"  # SwiGLU activation
  shared_expert_intermediate_size: 768  # Dimensi FFN shared expert (sama)
  
  # Routing Strategy: sqrtsoftplus + noaux_tc (DeepSeek V4 style)
  router:
    scoring_function: "sqrtsoftplus"  # sqrt(softplus(x)) — lebih stabil dari sigmoid
    noaux_tc: true  # Tanpa auxiliary loss (menggunakan dynamic bias correction)
    bias_update_speed: 0.001  # Kecepatan update bias koreksi
    load_balance_tolerance: 0.1  # Toleransi ketidakseimbangan
    max_expert_capacity: 1.5  # Kapasitas buffer expert (faktor dari rata-rata)
  
  # Expert affinity (MiniMax M2.7 style)
  expert_affinity:
    enabled: true
    # Token secara dinamis memilih expert yang pernah menanganinya dengan baik
    memory_size: 1024  # Jumlah token yang di-cache affinity-nya
    decay_factor: 0.95  # Decay untuk mengurangi bobot lama


# =============================================================================
# 5. Hyper-Connections (DeepSeek V4)
#    Fungsi: Menggantikan residual connection standar
#    Multiplisitas hidden state dengan Sinkhorn routing
# =============================================================================
hyper_connections:
  enabled: true
  hc_mult: 4  # Jumlah salinan hidden state per layer
  sinkhorn_iterations: 2  # Iterasi algoritma Sinkhorn (normalisasi baris & kolom)
  sinkhorn_temperature: 0.1  # Temperature untuk soft assignment
  hc_dim: 384  # Dimensi internal Hyper-Connection (sama dengan hidden_size)
  
  # Jenis koneksi yang dipertimbangkan
  connection_types:
    - "skip"         # Tidak ada koneksi (seperti residual nol)
    - "identity"     # Residual standar
    - "transform"    # Melalui transformasi linear
    - "gate"         # Melalui gate mechanism
  
  # Bobot awal koneksi (bisa dipelajari)
  initial_weights:
    identity: 0.6
    skip: 0.0
    transform: 0.2
    gate: 0.2


# =============================================================================
# 6. Multi-Token Prediction (MTP)
#    Sumber: DeepSeek V4
#    Fungsi: Memprediksi 2 token sekaligus untuk sinyal pembelajaran lebih kaya
# =============================================================================
mtp:
  enabled: true
  num_mtp_layers: 1  # Jumlah layer MTP tambahan
  mtp_depth: 2  # Memprediksi 2 token ke depan
  mtp_hidden_size: 384  # Dimensi hidden state MTP (sama dengan main model)
  mtp_loss_weight: 0.3  # Bobot loss MTP terhadap loss utama (cross-entropy)
  
  # MTP menggunakan embedding terpisah untuk positional offset
  mtp_positional_encoding: "rope"  # RoPE dengan offset
  mtp_head: "tied"  # Head output MTP berbagi bobot dengan main LM head


# =============================================================================
# 7. Hybrid Attention (MiniMax M2.7 Style)
#    Fungsi: Kombinasi Softmax Attention + Linear Attention
#    Sumber: MiniMax M2.7 (Hybrid Lightning + Softmax)
# =============================================================================
hybrid_attention:
  enabled: true
  # Layer 0, 4, 7 menggunakan Full Softmax Attention (posisi kunci)
  # Layer lainnya menggunakan Linear Attention (Lightning-style)
  softmax_layers: [0, 4, 7]  # ~3 dari 8 layer = ~37.5% Softmax
  linear_layers: [1, 2, 3, 5, 6]  # ~5 dari 8 layer = ~62.5% Linear
  
  linear_attention_config:
    type: "lightning_attention_v2"  # MiniMax Lightning Attention v2
    block_size: 256  # Ukuran blok untuk tile-based computation
    intra_block_type: "left_product"  # Perkalian kiri dalam blok (causal mask)
    inter_block_type: "right_product"  # Perkalian kanan antar blok (O(nd²))
    activation: "swish"  # Fungsi aktivasi pengganti softmax
    use_tiling: true  # Optimasi tile-based untuk GPU


# =============================================================================
# 8. Self-Evolution Framework (SEF) — INOVASI MINIMAX M2.7
#    Fungsi: Model dapat berpartisipasi dalam pelatihan dan optimasi dirinya sendiri
#    Sumber: MiniMax M2.7 "自我进化" (Self-Evolution)
# =============================================================================
self_evolution:
  enabled: true
  description: >
    Deeplm dilengkapi framework self-evolution ala MiniMax M2.7 yang memungkinkan
    model untuk:
    1. Menganalisis trajectory kegagalan (failure analysis)
    2. Mengusulkan perbaikan (improvement proposal)
    3. Mengeksekusi perubahan (code modification)
    4. Menjalankan evaluasi (evaluation)
    5. Memutuskan keep/revert (decision making)
    
    Framework ini bekerja melalui tiga lapis:
    Layer 1: Autonomous Research Loop — menjalankan eksperimen RL secara mandiri
    Layer 2: Harness Self-Optimization — mengoptimalkan Agent Harness miliknya
    Layer 3: Meta-Learning — memperbarui strategi pembelajaran berdasarkan hasil

  # Layer 1: Autonomous Research Loop
  autonomous_research:
    enabled: true
    max_iterations: 100  # Maksimum 100+ putaran (seperti MiniMax M2.7)
    workflow:
      - "hypothesis_generation"  # Menghasilkan hipotesis
      - "experiment_design"      # Merancang eksperimen
      - "code_execution"         # Menjalankan kode
      - "log_analysis"           # Menganalisis log
      - "bug_diagnosis"          # Mendiagnosis bug
      - "code_fix"               # Memperbaiki kode
      - "evaluation"             # Mengevaluasi hasil
      - "decision"               # Keep atau revert
    auto_commit: true  # Otomatis commit perubahan yang berhasil
    coverage_target: 0.35  # Target menangani 30-50% workflow (MiniMax klaim)
    
  # Layer 2: Harness Self-Optimization
  harness_optimization:
    enabled: true
    target_components:
      - "scaffold"       # Kerangka Agent Harness
      - "sampling_params"  # Parameter sampling (temperature, top-k, dll)
      - "workflow_strategy"  # Strategi workflow Agent
    improvement_threshold: 0.30  # Target peningkatan 30% (MiniMax klaim)
    
  # Layer 3: Meta-Learning Memory
  meta_memory:
    enabled: true
    memory_type: "short_term"  # File memori pendek per episode
    memory_file: "deeplm_memory.jsonl"  # File penyimpanan memori
    max_memory_entries: 10000
    feedback_chain:
      enabled: true  # Membangun rantai feedback dari episode sebelumnya
      chain_length: 5  # Jumlah episode yang dirantai
    consolidation_schedule: 100  # Konsolidasi memori setiap N step
    
  # Self-Evolution: Special Tokens
  evolution_tokens:
    analyze_start: "<|analyze_start|>"
    analyze_end: "<|analyze_end|>"
    propose_start: "<|propose_start|>"
    propose_end: "<|propose_end|>"
    execute_start: "<|execute_start|>"
    execute_end: "<|execute_end|>"
    evaluate_start: "<|evaluate_start|>"
    evaluate_end: "<|evaluate_end|>"
    decide_start: "<|decide_start|>"
    decide_end: "<|decide_end|>"


# =============================================================================
# 9. Agent Harness (OpenClaw Integration)
#    Sumber: MiniMax M2.7
#    Fungsi: Model dapat membangun kerangka Agent sendiri
# =============================================================================
agent_harness:
  enabled: true
  framework: "OpenClaw"  # Framework Agent standar MiniMax
  build_autonomously: true  # Model dapat membangun harness sendiri (inovasi kunci)
  
  # Komponen yang dapat dibangun model secara otonom
  auto_buildable:
    - "skills"        # Hierarchical Skills
    - "memory"        # Persistent Memory
    - "guardrails"    # Safety Guardrails
    - "evaluation"    # Evaluation Infrastructure
    - "mcp_servers"   # MCP (Model Context Protocol) Servers
  
  # Dev Harness: MiniMax melaporkan 1 engineer, 4 hari, 0 kode manusia
  dev_harness_config:
    auto_generate: true
    human_code_required: false
    target_build_time: "4_days"
    
  # Native Tool Calling
  tool_calling:
    enabled: true
    protocols:
      - "HTTP"
      - "RESTful"
      - "gRPC"
      - "MCP"  # Model Context Protocol
    max_tools_per_call: 10
    tool_call_accuracy_target: 0.987  # MiniMax klaim 98.7%


# =============================================================================
# 10. Training Configuration
# =============================================================================
training:
  # Pretraining
  pretraining:
    dataset: "FineWeb-Edu"  # Dataset edukasi berkualitas
    max_tokens: 5_000_000_000  # ~5B token (skala eksperimen; produksi perlu 1T+)
    batch_size: 8  # Batch size per GPU
    gradient_accumulation_steps: 4  # Effective batch = 32
    learning_rate: 6.0e-4
    min_learning_rate: 6.0e-6  # 1% dari LR awal
    lr_schedule: "cosine"
    warmup_steps: 150  # ~3% dari total steps
    weight_decay: 0.1
    adam_beta1: 0.9
    adam_beta2: 0.95
    adam_epsilon: 1.0e-8
    max_grad_norm: 1.0
    
    # Optimasi memori
    use_gradient_checkpointing: true
    use_flash_attention: true  # Flash Attention untuk efisiensi
    
  # Fine-Tuning (SFT)
  sft:
    dataset: "SmolTalk"  # 460K percakapan
    epochs: 3
    steps: 3000
    batch_size: 4
    gradient_accumulation_steps: 2
    learning_rate: 1.0e-4
    lr_schedule: "cosine"
    warmup_steps: 100

  # Self-Evolution Training (MiniMax M2.7 style)
  self_evolution_training:
    enabled: true
    # Model terlibat dalam ~35 hari siklus iterasi (MiniMax: 35 hari M2.5→M2.7)
    cycle_duration: "35_days"
    phases:
      - phase: "data_curation"
        model_involvement: "filter_and_augment"  # Model memfilter data training
      - phase: "training"
        model_involvement: "hyperparameter_suggestion"  # Model menyarankan hyperparameter
      - phase: "evaluation"
        model_involvement: "benchmark_analysis"  # Model menganalisis hasil benchmark
      - phase: "iteration"
        model_involvement: "improvement_planning"  # Model merencanakan perbaikan
    autonomous_rounds: 100  # MiniMax: 100+ round optimasi mandiri
    performance_improvement_target: 0.30  # Target peningkatan 30%


# =============================================================================
# 11. Inference Configuration
# =============================================================================
inference:
  # Standard Generation
  generation:
    temperature: 0.7
    top_k: 50
    top_p: 0.9
    repetition_penalty: 1.05
    max_new_tokens: 1024
    do_sample: true
    
  # Thinking Mode (DeepSeek-style)
  thinking_mode:
    enabled: true
    think_token_start: "<|think_start|>"
    think_token_end: "<|think_end|>"
    max_think_tokens: 512
    think_temperature: 0.6  # Sedikit lebih rendah untuk reasoning
    
  # Self-Evolution Inference
  self_evolution_mode:
    enabled: true
    # Mode khusus di mana model menghasilkan proposal perbaikan untuk dirinya sendiri
    evolution_temperature: 0.8  # Sedikit lebih tinggi untuk eksplorasi
    max_evolution_tokens: 2048
    
  # KV Cache Optimization
  kv_cache:
    type: "mla_compressed"
    compression_ratio: 24  # ~24x lebih kecil dari MHA penuh
    max_cache_size_mb: 256  # Batas maksimum cache dalam MB


# =============================================================================
# 12. Hardware Requirements
# =============================================================================
hardware:
  training:
    gpu: "A100_80GB"  # Atau H100 80GB
    num_gpus: 1
    memory_required_gb: 40  # Dengan gradient checkpointing
    disk_space_gb: 100  # Untuk dataset dan checkpoint
    
  inference:
    gpu: "A10_24GB"  # Atau RTX 4090 24GB
    num_gpus: 1
    memory_required_gb: 8  # Dengan MLA compression


# =============================================================================
# 13. Output Heads
# =============================================================================
output_heads:
  lm_head:
    type: "tied"  # Berbagi bobot dengan embedding
    bias: false
    
  mtp_heads:
    num_heads: 1  # Untuk 1 token tambahan
    type: "tied"  # Berbagi dengan LM head
    
  self_evolution_head:
    # Head khusus untuk menghasilkan proposal perbaikan (MiniMax M2.7)
    enabled: true
    type: "separate"  # Head terpisah untuk self-evolution
    hidden_size: 384
    output_size: 128000  # Sama dengan vocab_size


# =============================================================================
# 14. Special Features (Innovation Flags)
# =============================================================================
innovations:
  # Dari DeepSeek V4
  - name: "Multi-head Latent Attention (MLA)"
    source: "DeepSeek V4"
    paper: "DeepSeek-V2 Technical Report (2024)"
    benefit: "KV cache compression up to 24x"
    
  - name: "Hyper-Connections + Sinkhorn Routing"
    source: "DeepSeek V4"
    paper: "DeepSeek-V4 Technical Report (2025)"
    benefit: "Better training stability, replaces residual connections"
    
  - name: "Multi-Token Prediction (MTP)"
    source: "DeepSeek V4"
    paper: "DeepSeek-V3 Technical Report (2024)"
    benefit: "Richer learning signal, 5-8% downstream task improvement"
    
  # Dari Kimi K2.6
  - name: "Shared Expert (Always-On)"
    source: "Kimi K2.6"
    paper: "Moonshot AI Kimi K2 Series (2025)"
    benefit: "Ensures baseline knowledge always available"
    
  - name: "Agent Swarm Architecture"
    source: "Kimi K2.6"
    paper: "Kimi K2.6 Model Card (2026)"
    benefit: "Multi-agent collaboration capability"
    
  # Dari MiniMax M2.7
  - name: "Self-Evolution Framework (SEF)"
    source: "MiniMax M2.7"
    paper: "MiniMax M2.7 Self-Evolution Blog (2026)"
    benefit: "Model participates in own training & optimization; 30% improvement"
    
  - name: "Hybrid Attention (Lightning + Softmax)"
    source: "MiniMax M2.7"
    paper: "MiniMax-01 Technical Report (2025)"
    benefit: "Balanced efficiency (linear) and precision (softmax)"
    
  - name: "Agent Harness Auto-Construction"
    source: "MiniMax M2.7"
    paper: "MiniMax M2.7 Self-Evolution Blog (2026)"
    benefit: "Model builds its own execution framework (0 human code)"
    
  # Inovasi gabungan
  - name: "Hybrid Attention + MLA Co-design"
    source: "Deeplm (Gabungan)"
    benefit: "Linear attention + MLA compression = double efficiency gain"
    
  - name: "MTP + Self-Evolution Feedback"
    source: "Deeplm (Gabungan)"
    benefit: "MTP prediction quality used as self-evolution signal"
```

---

Penjelasan Inovasi Kunci

1. Self-Evolution Framework (SEF) — Inovasi MiniMax M2.7

Framework ini memungkinkan model untuk berpartisipasi dalam pelatihan dan optimasi dirinya sendiri—fitur kunci yang disebut MiniMax sebagai “model self-evolution” di mana model mampu melakukan analisis trajectory kegagalan, mengusulkan perbaikan, mengeksekusi perubahan, menjalankan evaluasi, dan memutuskan untuk keep atau revert secara mandiri.

Fitur yang memungkinkan model membangun kerangka kerja (framework) miliknya sendiri secara otonom, sesuai dengan prinsip MiniMax M2.7 di mana model membangun Agent Harness sendiri dengan 0 kode manusia.

2. Hybrid Attention (Lightning + Softmax) — Inovasi MiniMax M2.7

Model ini menggunakan Hybrid Attention yang menggabungkan Lightning Attention (linear) pada 5 layer dan Softmax Attention pada 3 layer kunci (0, 4, 7), mengikuti pendekatan MiniMax M2.7 yang menggunakan Hybrid Lightning + Softmax untuk efisiensi dan presisi.

3. Multi-head Latent Attention (MLA) — DeepSeek V4 & Kimi K2.6

Mengompresi KV cache dengan decoupled RoPE untuk mengurangi memori hingga 24x.

4. MoE + Shared Expert — DeepSeek V4 & Kimi K2.6

Menggunakan 4 routed expert + 1 shared expert per layer, dengan routing sqrtsoftplus dan tanpa auxiliary loss (menggunakan dynamic bias correction).

5. Hyper-Connections + Sinkhorn Routing — DeepSeek V4

Menggantikan residual connection standar dengan 4 salinan hidden state dan Sinkhorn routing untuk stabilitas training yang lebih baik.

6. Multi-Token Prediction (MTP) — DeepSeek V4

Memprediksi 2 token sekaligus untuk memberikan sinyal pembelajaran yang lebih kaya.

7. Agent Harness Auto-Construction — Inovasi MiniMax M2.7

Model dapat membangun framework eksekusinya sendiri, termasuk skills, memory, guardrails, evaluation infrastructure, dan MCP servers.

---

Metrik Utama

Metrik Nilai
Total Parameter ~108.5M
Arsitektur Decoder-only Transformer dengan MLA, MoE, Hybrid Attention, Hyper-Connections, MTP
KV Cache Compression ~24x lebih kecil dari MHA penuh
Self-Evolution Capability 100+ round autonomous optimization loop
Self-Evolution Improvement Target 30% peningkatan performa
Agent Harness Auto-constructed (0 human code)
