"""
Strict data pipeline: token cache, multi-stage filtering, bucket batching, weighted sampling.
"""
import hashlib
import json
import math
import os
import re
import struct
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import Dataset, Sampler, DataLoader


# ── URL / HTML / Emoji patterns ──
_RE_URL = re.compile(r"https?://[^\s]+|www\.[^\s]+", re.IGNORECASE)
_RE_HTML = re.compile(r"<[^>]+>")
_RE_EMOJI = re.compile(
    "[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U000024C2-\U0001F251]+"
)
_RE_EMAIL = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
_RE_MULTISPACE = re.compile(r" +")

# ── Language word lists ──
_ID_WORDS = {
    "yang", "dan", "di", "dengan", "ini", "itu", "dari", "dalam", "untuk",
    "pada", "adalah", "merupakan", "atau", "bahwa", "karena", "oleh", "juga",
    "tidak", "akan", "sudah", "dapat", "ke", "se", "ter", "me", "ber", "pe",
    "kan", "nya", "saya", "kami", "kita", "mereka", "dia", "anda", "dia",
    "telah", "sedang", "sangat", "semua", "setelah", "antara", "seperti",
    "tentang", "tanpa", "sambil", "hingga", "sejak", "hanya", "bisa", "ada",
    "lebih", "saat", "jika", "maka", "ketika", "sebagai", "besar", "kecil",
}

_EN_WORDS = {
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "it",
    "for", "not", "on", "with", "he", "as", "you", "do", "at", "this",
    "but", "his", "by", "from", "they", "we", "say", "her", "she", "or",
    "an", "will", "my", "one", "all", "would", "there", "their", "what",
    "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
    "when", "make", "can", "like", "time", "no", "just", "him", "know",
    "take", "people", "into", "year", "your", "good", "some", "could",
    "them", "see", "other", "than", "then", "now", "look", "only", "come",
    "its", "over", "think", "also", "back", "after", "use", "two", "how",
    "our", "work", "first", "well", "way", "even", "new", "want", "because",
    "any", "these", "give", "day", "most", "us",
}


def strip_noise(text):
    text = _RE_URL.sub("", text)
    text = _RE_HTML.sub("", text)
    text = _RE_EMAIL.sub("", text)
    text = _RE_EMOJI.sub("", text)
    text = _RE_MULTISPACE.sub(" ", text).strip()
    return text


def rep_score(text, n=4):
    words = text.split()
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    return 1.0 - len(set(ngrams)) / len(ngrams)


def char_ratio(text, allowed_chars=None):
    if allowed_chars is None:
        allowed_chars = set(
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 "
        )
    if not text:
        return 0.0
    return sum(1 for c in text if c in allowed_chars) / len(text)


def language_score(text, lang="id"):
    words = set(text.lower().split())
    ref = _ID_WORDS if lang == "id" else _EN_WORDS
    return len(words & ref) / max(len(words), 1)


def ngram_overlap(a, b, n=8):
    """Fraction of a's n-grams that appear in b (asymmetric)."""
    wa = a.split()
    wb = b.split()
    if len(wa) < n or len(wb) < n:
        return 0.0
    nga = set(tuple(wa[i:i + n]) for i in range(len(wa) - n + 1))
    ngb = set(tuple(wb[i:i + n]) for i in range(len(wb) - n + 1))
    if not nga:
        return 0.0
    return len(nga & ngb) / len(nga)


# ═══════════════════════════════════════════════════════════════════
# Multi-stage strict filter
# ═══════════════════════════════════════════════════════════════════

class StrictFilter:
    """Chainable strict text filter with detailed logging."""

    def __init__(self, config=None):
        self.cfg = config or {}
        self.stats = defaultdict(int)

    def filter(self, texts, lang="id"):
        self.stats.clear()
        self.stats["input"] = len(texts)
        out = []

        # Stage 1: strip noise
        stripped = []
        for t in texts:
            t = strip_noise(t)
            if t:
                stripped.append(t)
            else:
                self.stats["stripped_to_empty"] += 1
        self.stats["after_strip"] = len(stripped)

        # Stage 2: length
        min_l = self.cfg.get("min_length", 50)
        max_l = self.cfg.get("max_length", 8000)
        len_ok = []
        for t in stripped:
            if min_l <= len(t) <= max_l:
                len_ok.append(t)
            else:
                self.stats["length_rejected"] += 1
        self.stats["after_length"] = len(len_ok)

        # Stage 3: char ratio
        cr_min = self.cfg.get("min_char_ratio", 0.25)
        cr_ok = []
        for t in len_ok:
            cr = char_ratio(t)
            if cr >= cr_min:
                cr_ok.append(t)
            else:
                self.stats["char_ratio_rejected"] += 1
        self.stats["after_char_ratio"] = len(cr_ok)

        # Stage 4: language score (more strict than the old 1-word check)
        lang_min = self.cfg.get("min_lang_score", 0.0)
        lang_ok = []
        if lang_min > 0:
            for t in cr_ok:
                ls = language_score(t, lang)
                if ls >= lang_min:
                    lang_ok.append(t)
                else:
                    self.stats["lang_rejected"] += 1
        else:
            lang_ok = list(cr_ok)
        self.stats["after_language"] = len(lang_ok)

        # Stage 5: repetition
        rep_max = self.cfg.get("max_repetition_ratio", 0.4)
        rep_ok = []
        for t in lang_ok:
            rs = rep_score(t, n=self.cfg.get("rep_ngram", 4))
            if rs <= rep_max:
                rep_ok.append(t)
            else:
                self.stats["rep_rejected"] += 1
        self.stats["after_repetition"] = len(rep_ok)

        # Stage 6: word count
        min_w = self.cfg.get("min_words", 10)
        wc_ok = []
        for t in rep_ok:
            if len(t.split()) >= min_w:
                wc_ok.append(t)
            else:
                self.stats["word_count_rejected"] += 1
        self.stats["after_word_count"] = len(wc_ok)

        out = wc_ok
        self.stats["output"] = len(out)
        return out

    def summary(self):
        s = self.stats
        parts = []
        if s["input"]:
            parts.append(f"in={s['input']:,}")
        if s.get("stripped_to_empty"):
            parts.append(f"noise={s['stripped_to_empty']:,}")
        if s.get("length_rejected"):
            parts.append(f"len={s['length_rejected']:,}")
        if s.get("char_ratio_rejected"):
            parts.append(f"cr={s['char_ratio_rejected']:,}")
        if s.get("lang_rejected"):
            parts.append(f"lang={s['lang_rejected']:,}")
        if s.get("rep_rejected"):
            parts.append(f"rep={s['rep_rejected']:,}")
        if s.get("word_count_rejected"):
            parts.append(f"wc={s['word_count_rejected']:,}")
        parts.append(f"out={s['output']:,}")
        return " | ".join(parts)


# ═══════════════════════════════════════════════════════════════════
# Disk-backed token cache
# ═══════════════════════════════════════════════════════════════════

class TokenCache:
    """On-disk tokenization cache keyed by SHA-256 of text."""

    def __init__(self, cache_dir: str, tokenizer, max_seq_length: int, bos: int, eos: int, pad: int):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.bos = bos
        self.eos = eos
        self.pad = pad
        self._mem_cache = {}  # small in-memory LRU
        self._mem_max = 5000

    def _key(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _path(self, key: str) -> Path:
        return self.cache_dir / key[:2] / f"{key}.pt"

    def get(self, text: str):
        key = self._key(text)
        # In-memory hit
        if key in self._mem_cache:
            return self._mem_cache[key]
        # Disk hit
        p = self._path(key)
        if p.exists():
            try:
                tensors = torch.load(p, map_location="cpu", weights_only=True)
                self._mem_cache[key] = tensors
                if len(self._mem_cache) > self._mem_max:
                    self._mem_cache.pop(next(iter(self._mem_cache)))
                return tensors
            except Exception:
                p.unlink(missing_ok=True)
        return None

    def put(self, text: str, ids, attn, labels):
        key = self._key(text)
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tensors = (
            torch.tensor(ids, dtype=torch.long),
            torch.tensor(attn, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
        )
        if not p.parent.is_dir():
            p.parent.mkdir(parents=True, exist_ok=True)
        try:
            torch.save(tensors, p)
        except (RuntimeError, OSError) as e:
            print(f"  [WARN] TokenCache disk write failed ({p.name}): {e}", flush=True)
        self._mem_cache[key] = tensors
        if len(self._mem_cache) > self._mem_max:
            self._mem_cache.pop(next(iter(self._mem_cache)))

    def __len__(self):
        return sum(1 for _ in self.cache_dir.rglob("*.pt"))


# ═══════════════════════════════════════════════════════════════════
# Bucket dataset (groups texts by length for efficient padding)
# ═══════════════════════════════════════════════════════════════════

class BucketDataset(Dataset):
    """Groups texts into length buckets; tokenizes once, caches forever."""

    def __init__(self, texts, token_cache, bucket_size=64):
        self.cache = token_cache
        self.bucket_size = bucket_size
        self._buf = {"ids": [], "attn": [], "labels": []}
        self._pre_tokenized = False

        # Sort texts by length, then assign to buckets
        indexed = sorted(
            [(i, t) for i, t in enumerate(texts)],
            key=lambda x: len(x[1]),
        )
        self.buckets = []
        self._text_map = {}  # flat index → (text, bucket_idx, offset_in_bucket)
        flat_idx = 0
        for start in range(0, len(indexed), bucket_size):
            chunk = indexed[start:start + bucket_size]
            buf_i = []
            for orig_i, t in chunk:
                self._text_map[flat_idx] = {"text": t, "bucket": len(self.buckets), "offset": len(buf_i)}
                buf_i.append({"orig_idx": orig_i, "text": t})
                flat_idx += 1
            self.buckets.append(buf_i)

    def pre_tokenize(self):
        """Batch-pre-tokenize all texts into cache."""
        done = 0
        for b_idx, bucket in enumerate(self.buckets):
            for item in bucket:
                cached = self.cache.get(item["text"])
                if cached is None:
                    ids, attn, labels = self._tokenize(item["text"])
                    self.cache.put(item["text"], ids, attn, labels)
                    item["_cached"] = True
                else:
                    item["_cached"] = True
                done += 1
                if done % 10000 == 0:
                    print(f"    Pre-tokenized {done:,}...", flush=True)
        self._pre_tokenized = True
        return done

    @staticmethod
    def _tokenize_fn(text, tokenizer, max_seq_length, bos, eos, pad):
        enc = tokenizer.encode(text)
        if not enc.ids:
            ids = [bos, eos]
        else:
            ids = [bos] + enc.ids[:max_seq_length - 2] + [eos]
        ln = len(ids)
        if ln < max_seq_length:
            ids += [pad] * (max_seq_length - ln)
        attn = [1] * ln + [0] * (max_seq_length - ln)
        lbl = ids[:ln] + [-100] * (max_seq_length - ln)
        lbl[0] = -100
        return ids, attn, lbl

    def _tokenize(self, text):
        return self._tokenize_fn(text, self.cache.tokenizer, self.cache.max_seq_length,
                                  self.cache.bos, self.cache.eos, self.cache.pad)

    def __len__(self):
        return len(self._text_map)

    def __getitem__(self, idx):
        info = self._text_map[idx]
        cached = self.cache.get(info["text"])
        if cached is not None:
            return cached
        ids, attn, labels = self._tokenize(info["text"])
        self.cache.put(info["text"], ids, attn, labels)
        return (
            torch.tensor(ids, dtype=torch.long),
            torch.tensor(attn, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
        )

    def get_buckets(self):
        """Return list of (bucket_idx, item_count) for sampler."""
        return [(i, len(b)) for i, b in enumerate(self.buckets)]


# ═══════════════════════════════════════════════════════════════════
# Weighted / bucketed sampler
# ═══════════════════════════════════════════════════════════════════

class WeightedBucketSampler(Sampler):
    """Samples buckets (length groups) with optional category weights,
    then shuffles within the selected bucket."""

    def __init__(self, bucket_counts, bucket_ids, bucket_categories=None,
                 category_weights=None, epoch_size=None, seed=42):
        """
        bucket_counts: list of (bucket_idx, n_items)
        bucket_ids: mapping from flat_idx → bucket_idx
        bucket_categories: optional mapping from bucket_idx → category string
        category_weights: optional dict {category: weight}
        """
        self.bucket_counts = bucket_counts
        self.bucket_ids = bucket_ids
        self.bucket_categories = bucket_categories or {}
        self.category_weights = category_weights or {}
        self.epoch_size = epoch_size or sum(n for _, n in bucket_counts)
        self.seed = seed
        self.g = torch.Generator().manual_seed(seed)

    def _bucket_weight(self, b_idx):
        cat = self.bucket_categories.get(b_idx)
        if cat and cat in self.category_weights:
            return self.category_weights[cat]
        return 1.0

    def __iter__(self):
        """Yield flat indices by: pick bucket (weighted) → shuffle → yield."""
        bi = [b_idx for b_idx, n in self.bucket_counts]
        bw = [self._bucket_weight(b_idx) for b_idx in bi]

        # Build flat_idx → bucket_idx lookup
        flat_to_bucket = {}
        bucket_starts = {}
        offset = 0
        for b_idx, n in self.bucket_counts:
            bucket_starts[b_idx] = offset
            for j in range(n):
                flat_to_bucket[offset + j] = b_idx
            offset += n

        if not bw:
            return iter(range(self.epoch_size))

        total = sum(bw)
        probs = [w / total for w in bw]

        sampled = 0
        while sampled < self.epoch_size:
            b_idx = bi[torch.multinomial(torch.tensor(probs), 1).item()]
            start = bucket_starts[b_idx]
            n = dict(self.bucket_counts)[b_idx]
            order = torch.randperm(n, generator=self.g).tolist()
            for j in order:
                if sampled >= self.epoch_size:
                    break
                yield start + j
                sampled += 1

    def __len__(self):
        return self.epoch_size


# ═══════════════════════════════════════════════════════════════════
# Convenience: build pipeline from raw texts
# ═══════════════════════════════════════════════════════════════════

def build_pipeline(
    texts,
    tokenizer,
    max_seq_length,
    bos,
    eos,
    pad,
    cache_dir="/tmp/token_cache",
    bucket_size=64,
    batch_size=8,
    num_workers=4,
    pin_memory=True,
    category_map=None,        # list of (flat_idx → category string)
    category_weights=None,    # dict {category: weight}
    seed=42,
):
    cache = TokenCache(cache_dir, tokenizer, max_seq_length, bos, eos, pad)
    dataset = BucketDataset(texts, cache, bucket_size=bucket_size)

    # Map categories from flat index to bucket
    bucket_categories = {}
    if category_map:
        for flat_idx, cat in enumerate(category_map):
            info = dataset._text_map.get(flat_idx)
            if info:
                b_idx = info["bucket"]
                bucket_categories.setdefault(b_idx, cat)

    # Build bucket_counts
    bucket_counts = dataset.get_buckets()
    bucket_ids = {idx: info["bucket"] for idx, info in dataset._text_map.items()}

    sampler = WeightedBucketSampler(
        bucket_counts=bucket_counts,
        bucket_ids=bucket_ids,
        bucket_categories=bucket_categories,
        category_weights=category_weights,
        epoch_size=len(dataset),
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

    return dataset, sampler, loader, cache
