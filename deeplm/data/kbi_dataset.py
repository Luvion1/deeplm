"""
KBBI Dataset with memory-mapped storage for minimal RAM usage.

Uses numpy memmap to store tokenized data on disk, loading only
the needed batches into RAM during training.
"""
import json
import os
import random
from typing import Dict, Optional, List

import numpy as np
import torch
from datasets import load_dataset
from tokenizers import Tokenizer


class KBBIFormatter:
    """Format KBBI dictionary entries into training text."""

    TEMPLATES = [
        "{nama} adalah kata yang berarti {submakna}.",
        "Kata {nama} termasuk kelas kata {kelas}. Artinya: {submakna}",
        "{nama} ({pelafalan}) berarti {submakna}.",
        "Apa arti {nama}? {nama} berarti {submakna}.",
        "{submakna}. Itulah arti dari kata {nama}.",
        "Dalam bahasa Indonesia, {nama} berarti {submakna}.",
        "{nama}: {submakna}",
        "Definisi {nama}: {submakna}",
    ]

    CONTOH_TEMPLATES = [
        "Contoh penggunaan: {contoh}",
        "Misalnya: {contoh}",
        "Penggunaan dalam kalimat: {contoh}",
    ]

    ETIMOLOGI_TEMPLATES = [
        "Asal kata: {etimologi}",
        "Etimologi: {etimologi}",
    ]

    @classmethod
    def format_entry(cls, row: Dict) -> Optional[str]:
        """Format a single KBBI dictionary entry into training text."""
        nama = row.get("nama")
        if not nama or not isinstance(nama, str) or len(nama.strip()) == 0:
            return None

        submakna = row.get("submakna", "") or "tidak diketahui"
        kelas = row.get("kelas", "") or "tidak diketahui"
        pelafalan = row.get("pelafalan", "")
        contoh = row.get("contoh", "")
        etimologi = row.get("etimologi", "")
        gabungan_kata = row.get("gabungan_kata", "")
        peribahasa = row.get("peribahasa", "")

        parts = []

        template = random.choice(cls.TEMPLATES)
        try:
            main_text = template.format(
                nama=nama,
                submakna=submakna,
                kelas=kelas,
                pelafalan=pelafalan if pelafalan else nama,
            )
        except KeyError:
            main_text = f"{nama} berarti {submakna}."
        parts.append(main_text)

        if contoh and len(contoh) > 5 and random.random() > 0.3:
            try:
                parts.append(random.choice(cls.CONTOH_TEMPLATES).format(contoh=contoh))
            except KeyError:
                parts.append(f"Contoh: {contoh}")

        if etimologi and len(etimologi) > 3 and random.random() > 0.5:
            try:
                parts.append(random.choice(cls.ETIMOLOGI_TEMPLATES).format(etimologi=etimologi))
            except KeyError:
                parts.append(f"Asal: {etimologi}")

        if gabungan_kata and len(gabungan_kata) > 3 and random.random() > 0.5:
            parts.append(f"Gabungan kata: {gabungan_kata}")

        if peribahasa and len(peribahasa) > 5 and random.random() > 0.7:
            parts.append(f"Peribahasa: {peribahasa}")

        return " ".join(parts)


class MappedKBBIDataset(torch.utils.data.Dataset):
    """
    Memory-mapped KBBI dataset.

    Tokenizes all data once and stores as numpy memmap files on disk.
    Only the current batch is loaded into RAM during training.

    Memory usage: O(batch_size * seq_len) instead of O(dataset_size * seq_len)
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        max_seq_length: int = 2048,
        cache_dir: str = "data_cache/",
        num_rows: int = None,
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        self.input_ids_path = os.path.join(cache_dir, "input_ids.mmap")
        self.attention_mask_path = os.path.join(cache_dir, "attention_mask.mmap")
        self.labels_path = os.path.join(cache_dir, "labels.mmap")
        self.meta_path = os.path.join(cache_dir, "dataset_meta.json")

        self.bos = self.tokenizer.token_to_id("<|begin_of_sentence|>")
        self.eos = self.tokenizer.token_to_id("<|end_of_sentence|>")
        self.pad_id = self.tokenizer.token_to_id("<|pad|>")

        # Validate special tokens
        for name, tok_id in [("bos", self.bos), ("eos", self.eos), ("pad", self.pad_id)]:
            if tok_id is None:
                raise ValueError(f"Tokenizer missing special token: <|{name}|>")

        if self._is_cached():
            print(f"Loading memory-mapped dataset from {cache_dir}")
            self._load_meta()
        else:
            print("Creating memory-mapped dataset...")
            self._build_mmap(num_rows)

        self.input_ids_mmap = np.memmap(
            self.input_ids_path, dtype=np.int64, mode="r",
            shape=(self.num_samples, self.max_seq_length),
        )
        self.attention_mask_mmap = np.memmap(
            self.attention_mask_path, dtype=np.int64, mode="r",
            shape=(self.num_samples, self.max_seq_length),
        )
        self.labels_mmap = np.memmap(
            self.labels_path, dtype=np.int64, mode="r",
            shape=(self.num_samples, self.max_seq_length),
        )

        print(f"Dataset: {self.num_samples:,} samples, {self.max_seq_length} seq_len")
        print(f"  Memory-mapped files: ~{self._disk_size_mb():.1f} MB on disk")
        print(f"  RAM usage per batch: ~{self._batch_ram_mb(8):.1f} MB (batch_size=8)")

    def _is_cached(self) -> bool:
        return all(os.path.exists(p) for p in [
            self.input_ids_path, self.attention_mask_path,
            self.labels_path, self.meta_path,
        ])

    def _load_meta(self):
        with open(self.meta_path, "r") as f:
            meta = json.load(f)
        self.num_samples = meta["num_samples"]
        self.max_seq_length = meta["max_seq_length"]

    def _build_mmap(self, num_rows: int = None):
        """Tokenize dataset and store as memory-mapped files.

        Uses a single-pass streaming approach: iterate dataset, tokenize
        in batches, and write directly to memmap. Only accumulates one
        batch of texts in RAM at a time.
        """
        ds = load_dataset("Lyon28/kamus-besar-bahasa-indonesia", split="train", streaming=True)

        # First pass: count valid rows (no text storage)
        print("  Counting rows (pass 1/2)...")
        num_samples = 0
        for row in ds:
            text = KBBIFormatter.format_entry(row)
            if text and len(text) > 5:
                num_samples += 1
            if num_rows and num_samples >= num_rows:
                break

        if num_samples == 0:
            raise ValueError("No valid texts collected from KBBI dataset")
        print(f"  Total valid texts: {num_samples:,}")

        # Create memmap files
        dtype = np.int64
        input_ids_mmap = np.memmap(
            self.input_ids_path, dtype=dtype, mode="w+",
            shape=(num_samples, self.max_seq_length),
        )
        attention_mask_mmap = np.memmap(
            self.attention_mask_path, dtype=dtype, mode="w+",
            shape=(num_samples, self.max_seq_length),
        )
        labels_mmap = np.memmap(
            self.labels_path, dtype=dtype, mode="w+",
            shape=(num_samples, self.max_seq_length),
        )

        # Second pass: tokenize and write (no text accumulation)
        print("  Tokenizing and writing to disk (pass 2/2)...")
        ds = load_dataset("Lyon28/kamus-besar-bahasa-indonesia", split="train", streaming=True)
        batch_size = 500
        idx = 0
        batch_texts = []
        for row in ds:
            text = KBBIFormatter.format_entry(row)
            if not text or len(text) <= 5:
                continue

            batch_texts.append(text)

            if len(batch_texts) == batch_size:
                self._write_encoded_batch(batch_texts, input_ids_mmap, attention_mask_mmap, labels_mmap, idx)
                idx += len(batch_texts)
                batch_texts = []
                if idx % 5000 == 0:
                    print(f"    Processed {idx:,}/{num_samples:,}")

            if idx >= num_samples:
                break

        # Flush remaining batch
        if batch_texts:
            self._write_encoded_batch(batch_texts, input_ids_mmap, attention_mask_mmap, labels_mmap, idx)
            del batch_texts

        # Flush to disk
        input_ids_mmap.flush()
        attention_mask_mmap.flush()
        labels_mmap.flush()

        del input_ids_mmap, attention_mask_mmap, labels_mmap

        # Save metadata
        with open(self.meta_path, "w") as f:
            json.dump(
                {
                    "num_samples": num_samples,
                    "max_seq_length": self.max_seq_length,
                    "bos_token_id": self.bos,
                    "eos_token_id": self.eos,
                    "pad_token_id": self.pad_id,
                },
                f,
                indent=2,
            )

        self.num_samples = num_samples

    def _write_encoded_batch(self, texts, input_ids_mmap, attention_mask_mmap, labels_mmap, start_idx):
        """Tokenize a batch of texts and write directly to memmap files."""
        encodings = self.tokenizer.encode_batch(texts)
        for j, enc in enumerate(encodings):
            ids = [self.bos] + enc.ids + [self.eos]
            if len(ids) > self.max_seq_length:
                ids = ids[:self.max_seq_length]
                ids[-1] = self.eos
            actual_len = len(ids)
            if actual_len < self.max_seq_length:
                ids = ids + [self.pad_id] * (self.max_seq_length - actual_len)
            write_idx = start_idx + j
            input_ids_mmap[write_idx] = ids
            mask = [1] * actual_len + [0] * (self.max_seq_length - actual_len)
            attention_mask_mmap[write_idx] = mask
            labels = list(ids)
            for k in range(actual_len, self.max_seq_length):
                labels[k] = -100
            labels_mmap[write_idx] = labels

    def _disk_size_mb(self) -> float:
        total = 0
        for path in [self.input_ids_path, self.attention_mask_path, self.labels_path]:
            if os.path.exists(path):
                total += os.path.getsize(path)
        return total / (1024 * 1024)

    def _batch_ram_mb(self, batch_size: int) -> float:
        return (batch_size * self.max_seq_length * 8 * 3) / (1024 * 1024)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        input_ids = torch.from_numpy(self.input_ids_mmap[idx].copy())
        attention_mask = torch.from_numpy(self.attention_mask_mmap[idx].copy())
        labels = torch.from_numpy(self.labels_mmap[idx].copy())

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def close(self):
        """Close memory-mapped file handles to release OS resources."""
        for attr in ("input_ids_mmap", "attention_mask_mmap", "labels_mmap"):
            mmap = getattr(self, attr, None)
            if mmap is not None:
                del mmap
            setattr(self, attr, None)

    def __del__(self):
        self.close()

    def cleanup(self):
        """Remove cached memmap files (close handles first)."""
        self.close()
        for path in [self.input_ids_path, self.attention_mask_path, self.labels_path, self.meta_path]:
            if os.path.exists(path):
                os.remove(path)
        print(f"Cleaned up cached files in {self.cache_dir}")
