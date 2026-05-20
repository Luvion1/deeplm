"""
Train a 128k BBPE tokenizer from KBBI + Corpus-Indonesia + adapted-wordnet.

This is run ONCE on Modal to produce the final tokenizer. The output is
saved to the persistent volume and reused for all training runs.

Usage:
    modal run scripts/train_tokenizer_full.py --vocab-size 128000
"""
import os
from pathlib import Path

import modal

volume = modal.Volume.from_name("deeplm-checkpoints", create_if_missing=True)
VOLUME_PATH = "/vol"

image = (
    modal.Image.from_registry("pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel")
    .run_commands("apt-get update -qq && apt-get install -y -qq git")
    .pip_install("datasets", "tokenizers", "tqdm")
)

app = modal.App("deeplm-tokenizer", image=image)


@app.function(
    cpu=8,
    memory=32768,
    timeout=3600 * 2,
    volumes={VOLUME_PATH: volume},
)
def train_tokenizer(vocab_size: int = 32000):
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders
    from tokenizers.normalizers import NFKC
    from datasets import load_dataset, concatenate_datasets
    import json

    output_dir = Path(VOLUME_PATH) / "tokenizer"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Special tokens ──────────────────────────────────────────────
    special_tokens = [
        "<|pad|>",
        "<|unk|>",
        "<|begin_of_sentence|>",
        "<|end_of_sentence|>",
        "<|think_start|>",
        "<|think_end|>",
        "<|system|>",
        "<|user|>",
        "<|assistant|>",
        "<|tool_call|>",
        "<|tool_response|>",
        "<|self_evolve|>",
        "<|memory_write|>",
        "<|memory_read|>",
        "<|mask|>",
    ]

    # ── Text iterator from all datasets ─────────────────────────────
    def text_iterator():
        # 1. KBBI — dictionary entries formatted as natural text
        print("Loading KBBI dataset...")
        kbi = load_dataset("Lyon28/kamus-besar-bahasa-indonesia", split="train", streaming=True)
        for i, row in enumerate(kbi):
            parts = []
            for key in ["nama", "submakna", "contoh", "etimologi", "gabungan_kata", "peribahasa"]:
                val = row.get(key)
                if val:
                    parts.append(str(val))
            if parts:
                yield " ".join(parts)
            if i % 50000 == 0 and i > 0:
                print(f"  KBBI: {i:,} rows processed")

        # 2. Corpus-Indonesia — 19.5M general Indonesian text
        print("Loading Corpus-Indonesia...")
        corp = load_dataset("Lyon28/Corpus-Indonesia", split="train", streaming=True)
        for i, row in enumerate(corp):
            text = row.get("text")
            if text:
                yield text
            if i % 500000 == 0 and i > 0:
                print(f"  Corpus-Indonesia: {i:,} rows processed")

        # 3. adapted-wordnet — dictionary-style definitions for extra vocab
        print("Loading adapted-wordnet...")
        try:
            wnet = load_dataset("cestwc/adapted-wordnet", split="train", streaming=True)
            for i, row in enumerate(wnet):
                for key in ["long", "short", "synthetic"]:
                    text = row.get(key)
                    if text:
                        yield text
                if i % 10000 == 0 and i > 0:
                    print(f"  WordNet: {i:,} rows processed")
        except Exception as e:
            print(f"  WordNet skip: {e}")

    # ── Train tokenizer ─────────────────────────────────────────────
    print(f"Training BPE tokenizer with vocab_size={vocab_size:,}...")
    tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))
    tokenizer.normalizer = NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=special_tokens,
    )

    tokenizer.train_from_iterator(text_iterator(), trainer=trainer)

    # ── Save ────────────────────────────────────────────────────────
    tokenizer.save(str(output_dir / "tokenizer.json"))

    with open(output_dir / "tokenizer_config.json", "w") as f:
        json.dump({
            "vocab_size": tokenizer.get_vocab_size(),
            "model_max_length": 2048,
            "tokenizer_class": "BPETokenizer",
            "special_tokens": {
                "pad_token": "<|pad|>",
                "unk_token": "<|unk|>",
                "bos_token": "<|begin_of_sentence|>",
                "eos_token": "<|end_of_sentence|>",
            },
        }, f, indent=2)

    print(f"\nTokenizer saved to {output_dir}")
    print(f"Vocab size: {tokenizer.get_vocab_size():,}")
    print(f"Done!")


@app.local_entrypoint()
def main(vocab_size: int = 32000):
    train_tokenizer.remote(vocab_size=vocab_size)
