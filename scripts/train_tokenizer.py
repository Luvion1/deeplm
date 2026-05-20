"""
Train a BPE tokenizer from the KBBI dataset.

Usage:
    python scripts/train_tokenizer.py --vocab_size 128000 --output_dir tokenizer/
"""
import argparse
import json
import os
from pathlib import Path

from datasets import load_dataset
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from tokenizers.normalizers import NFKC
from tqdm import tqdm


def kbi_iterator(num_rows=None):
    """Iterate over KBBI dataset and yield text samples."""
    ds = load_dataset("Lyon28/kamus-besar-bahasa-indonesia", split="train", streaming=True)

    count = 0
    for row in ds:
        texts = []

        nama = row.get("nama")
        if nama:
            texts.append(nama)

        kata_dasar = row.get("kata_dasar")
        if kata_dasar:
            texts.append(kata_dasar)

        pelafalan = row.get("pelafalan")
        if pelafalan:
            texts.append(pelafalan)

        kelas = row.get("kelas")
        if kelas:
            texts.append(kelas)

        submakna = row.get("submakna")
        if submakna:
            texts.append(submakna)

        contoh = row.get("contoh")
        if contoh:
            texts.append(contoh)

        etimologi = row.get("etimologi")
        if etimologi:
            texts.append(etimologi)

        gabungan_kata = row.get("gabungan_kata")
        if gabungan_kata:
            texts.append(gabungan_kata)

        peribahasa = row.get("peribahasa")
        if peribahasa:
            texts.append(peribahasa)

        if texts:
            yield " ".join(texts)

        count += 1
        if num_rows and count >= num_rows:
            break


def train_tokenizer(vocab_size: int, output_dir: str, num_rows: int = None):
    """Train a BPE tokenizer from KBBI dataset."""
    os.makedirs(output_dir, exist_ok=True)

    tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))
    tokenizer.normalizer = NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=[
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
        ],
    )

    print("Training tokenizer from KBBI dataset...")
    tokenizer.train_from_iterator(kbi_iterator(num_rows), trainer=trainer)

    vocab = tokenizer.get_vocab()
    print(f"Vocabulary size: {len(vocab)}")

    tokenizer.save(os.path.join(output_dir, "tokenizer.json"))

    with open(os.path.join(output_dir, "tokenizer_config.json"), "w") as f:
        json.dump(
            {
                "vocab_size": vocab_size,
                "model_max_length": 2048,
                "tokenizer_class": "BPETokenizer",
                "special_tokens": {
                    "pad_token": "<|pad|>",
                    "unk_token": "<|unk|>",
                    "bos_token": "<|begin_of_sentence|>",
                    "eos_token": "<|end_of_sentence|>",
                },
            },
            f,
            indent=2,
        )

    print(f"Tokenizer saved to {output_dir}")
    return tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", type=int, default=128000)
    parser.add_argument("--output_dir", type=str, default="tokenizer/")
    parser.add_argument("--num_rows", type=int, default=None, help="Limit rows for quick testing")
    args = parser.parse_args()

    train_tokenizer(args.vocab_size, args.output_dir, args.num_rows)
