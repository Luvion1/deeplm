"""
Dataset Registry + Semantic Categorizer.

Persistent dataset cache: tokenizes once, categorizes, and reuses on reload.
SemanticCategorizer classifies texts into training-relevant categories.
"""
import hashlib
import json
import math
import os
import re
import shutil
from collections import defaultdict, Counter
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader


# ── Category definitions ──
CATEGORIES = [
    "reasoning",     # math/logic/chain-of-thought
    "grammar",       # syntax/language-rules/linguistic
    "code",          # programming
    "creative",      # stories/poetry/narrative
    "instruction",   # how-to/guide/tutorial
    "dialog",        # conversation/Q&A
    "academic",      # scientific/formal-writing
    "summarization", # summarization/paraphrase
]

CATEGORY_ALIASES = {
    "math": "reasoning",
    "logic": "reasoning",
    "chain-of-thought": "reasoning",
    "syntax": "grammar",
    "linguistic": "grammar",
    "programming": "code",
    "story": "creative",
    "narrative": "creative",
    "poetry": "creative",
    "how-to": "instruction",
    "guide": "instruction",
    "tutorial": "instruction",
    "qa": "dialog",
    "conversation": "dialog",
    "question-answering": "dialog",
    "scientific": "academic",
    "formal": "academic",
    "summarizing": "summarization",
    "paraphrase": "summarization",
    "other": "other",
}

# Phase → preferred categories for curriculum routing
PHASE_CATEGORIES = {
    "warmup":       ["grammar", "dialog", "instruction"],
    "exploration":  ["reasoning", "code", "instruction"],
    "balanced":     ["reasoning", "code", "creative", "academic"],
    "exploitation": ["reasoning", "summarization", "academic"],
}


# ═══════════════════════════════════════════════════════════════════
# Semantic Categorizer
# ═══════════════════════════════════════════════════════════════════

class SemanticCategorizer:
    """Lightweight semantic categorizer for text classification.

    Uses keyword heuristics + structural patterns.
    """

    # Keyword signatures per category (lowercase)
    _SIGNATURES = {
        "reasoning": {
            "keywords": {
                "therefore", "because", "since", "thus", "hence", "if", "then",
                "implies", "conclude", "proof", "prove", "assume", "suppose",
                "calculate", "compute", "equation", "formula", "solve", "solution",
                "deduce", "infer", "logical", "reason", "argument", "premise",
                "contradiction", "hypothesis", "theorem", "lemma", "corollary",
                "step", "first", "second", "finally", "let x", "define",
            },
            "min_score": 0.03,
        },
        "grammar": {
            "keywords": {
                "sentence", "grammar", "noun", "verb", "adjective", "adverb",
                "preposition", "conjunction", "tense", "plural", "singular",
                "syntax", "clause", "phrase", "subject", "predicate", "article",
                "pronoun", "prefix", "suffix", "inflection", "morphology",
                "correct", "incorrect", "rewrite", "paraphrase", "grammatical",
                "punctuation", "spelling", "vocabulary", "word", "meaning",
            },
            "min_score": 0.02,
        },
        "code": {
            "keywords": {
                "def ", "class ", "import ", "return ", "function", "variable",
                "loop", "array", "string", "integer", "boolean", "compile",
                "syntax error", "runtime", "debug", "print", "input", "output",
                "argument", "parameter", "method", "attribute", "object",
                "inheritance", "exception", "try", "catch", "throw",
            },
            "min_score": 0.02,
            "bonus_patterns": [
                (r"\bdef\s+\w+\s*\(", 0.5),
                (r"\bclass\s+\w+", 0.3),
                (r"^\s*(import|from)\s+\w+", 0.3),
                (r"^\s*#.*$", 0.2),
                (r"->\s*\w+:", 0.2),
            ],
        },
        "creative": {
            "keywords": {
                "story", "once upon", "tale", "novel", "poem", "poetry",
                "imagine", "dream", "fantasy", "mystery", "adventure",
                "character", "hero", "villain", "plot", "setting",
                "describe", "feeling", "emotion", "beautiful", "scary",
                "chapter", "epilogue", "prologue", "narrative",
            },
            "min_score": 0.02,
        },
        "instruction": {
            "keywords": {
                "how to", "step", "first", "next", "then", "finally",
                "tutorial", "guide", "instruction", "manual", "directions",
                "follow", "repeat", "example", "tip", "advice", "recommend",
                "way to", "method", "technique", "approach", "strategy",
            },
            "min_score": 0.02,
        },
        "dialog": {
            "keywords": {
                "hello", "hi", "thanks", "please", "question", "answer",
                "what is", "how do", "can you", "would you", "tell me",
                "explain", "help", "suggest", "recommend", "opinion",
            },
            "min_score": 0.01,
            "bonus_patterns": [
                (r"\?$", 0.3),
                (r"^[A-Z][a-z]+:", 0.2),
            ],
        },
        "academic": {
            "keywords": {
                "research", "study", "analysis", "experiment", "theory",
                "literature", "review", "methodology", "conclusion",
                "abstract", "introduction", "section", "table", "figure",
                "significant", "correlation", "hypothesis", "data",
                "participant", "sample", "finding", "discussion",
            },
            "min_score": 0.02,
        },
        "summarization": {
            "keywords": {
                "summary", "summarize", "brief", "overview", "key point",
                "main idea", "in short", "to summarize", "in conclusion",
                "essentially", "in other words", "simply put",
            },
            "min_score": 0.01,
        },
    }

    def __init__(self):
        self._cache = {}

    def classify(self, text: str) -> str:
        """Return best-guess category label."""
        if not text:
            return "other"
        # Check cache
        h = hashlib.md5(text.encode()).hexdigest()
        if h in self._cache:
            return self._cache[h]

        lower = text.lower()
        scores = {}
        for cat, sig in self._SIGNATURES.items():
            score = 0.0
            n_terms = len(sig["keywords"])
            matches = sum(1 for kw in sig["keywords"] if kw in lower)
            if n_terms > 0:
                score = matches / n_terms
            # Bonus patterns
            for pattern, bonus in sig.get("bonus_patterns", []):
                if re.search(pattern, text):
                    score += bonus
            scores[cat] = score

        # Check min thresholds
        best_cat = "other"
        best_score = 0.0
        for cat, score in scores.items():
            sig = self._SIGNATURES[cat]
            if score >= sig.get("min_score", 0.01) and score > best_score:
                best_cat = cat
                best_score = score

        self._cache[h] = best_cat
        return best_cat

    def classify_batch(self, texts, batch_size=1000):
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            results.extend([self.classify(t) for t in batch])
        return results

    def category_distribution(self, texts):
        cats = Counter(self.classify_batch(texts))
        return dict(cats)


# ═══════════════════════════════════════════════════════════════════
# Dataset Registry — persistent on-disk
# ═══════════════════════════════════════════════════════════════════

class DatasetRegistry:
    """Persistent dataset registry.

    Each dataset source is stored as a named entry. Tokenized + categorized
    data is cached to disk. On reload, only new/changed texts are processed.
    """

    def __init__(self, registry_dir: str, categorizer: SemanticCategorizer = None):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.categorizer = categorizer or SemanticCategorizer()
        self._entries = {}  # name → entry dict
        self._load_index()

    def _index_path(self):
        return self.registry_dir / "index.json"

    def _load_index(self):
        if self._index_path().exists():
            with open(self._index_path()) as f:
                data = json.load(f)
                self._entries = data.get("entries", {})

    def _save_index(self):
        with open(self._index_path(), "w") as f:
            json.dump({"entries": self._entries}, f, indent=2)

    def _entry_dir(self, name: str):
        return self.registry_dir / hashlib.md5(name.encode()).hexdigest()[:16]

    def has_entry(self, name: str) -> bool:
        return name in self._entries

    def get_entry(self, name: str):
        return self._entries.get(name)

    def register(self, name: str, texts: list, force_rebuild=False):
        """Register a dataset. Tokenizes + categorizes if not cached."""
        entry = self._entries.get(name, {})
        entry["name"] = name
        entry["num_texts"] = len(texts)

        edir = self._entry_dir(name)
        edir.mkdir(parents=True, exist_ok=True)

        # Check if already fully cached
        manifest_path = edir / "manifest.json"
        if manifest_path.exists() and not force_rebuild:
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("num_texts") == len(texts) and manifest.get("version"):
                entry["manifest"] = manifest
                entry["category_counts"] = manifest.get("category_counts", {})
                self._entries[name] = entry
                self._save_index()
                return  # Already cached, reuse

        # Categorize
        categories = self.categorizer.classify_batch(texts)
        cat_counts = Counter(categories)

        # Build per-category index
        cat_indices = defaultdict(list)
        for idx, cat in enumerate(categories):
            cat_indices[cat].append(idx)

        # Save manifest
        manifest = {
            "version": 2,
            "num_texts": len(texts),
            "category_counts": dict(cat_counts),
            "categories": list(cat_indices.keys()),
        }
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        # Save category index (flat index → category mapping)
        cat_idx_path = edir / "category_index.pt"
        torch.save(
            torch.tensor([CATEGORIES.index(c) if c in CATEGORIES else -1 for c in categories],
                          dtype=torch.long),
            cat_idx_path,
        )

        # Save per-category text indices
        for cat, indices in cat_indices.items():
            cat_file = edir / f"cat_{cat}.pt"
            torch.save(torch.tensor(indices, dtype=torch.long), cat_file)

        # Save original texts only once
        texts_path = edir / "texts.pt"
        if not texts_path.exists() or force_rebuild:
            torch.save(texts, texts_path)

        entry["manifest"] = manifest
        entry["category_counts"] = dict(cat_counts)
        self._entries[name] = entry
        self._save_index()

    def get_texts(self, name: str):
        """Return all texts for a named dataset."""
        edir = self._entry_dir(name)
        texts_path = edir / "texts.pt"
        if texts_path.exists():
            return torch.load(texts_path, map_location="cpu", weights_only=True)
        return []

    def get_category_indices(self, name: str, category: str):
        """Return text indices belonging to a category."""
        edir = self._entry_dir(name)
        cat_file = edir / f"cat_{category}.pt"
        if cat_file.exists():
            return torch.load(cat_file, map_location="cpu", weights_only=True).tolist()
        return []

    def get_category_texts(self, name: str, category: str):
        """Return all texts of a specific category."""
        texts = self.get_texts(name)
        indices = self.get_category_indices(name, category)
        return [texts[i] for i in indices]

    def categories(self, name: str):
        """Return list of available categories for a dataset."""
        entry = self._entries.get(name)
        if entry:
            return list(entry.get("category_counts", {}).keys())
        return []

    def summary(self, name: str):
        entry = self._entries.get(name)
        if not entry:
            return f"Dataset '{name}' not registered"
        cc = entry.get("category_counts", {})
        total = entry.get("num_texts", 0)
        parts = [f"Dataset '{name}': {total:,} texts"]
        for cat in sorted(cc, key=lambda c: cc[c], reverse=True):
            pct = cc[cat] / total * 100 if total > 0 else 0
            parts.append(f"  {cat}: {cc[cat]:,} ({pct:.1f}%)")
        return "\n".join(parts)

    def list_datasets(self):
        return list(self._entries.keys())


# ═══════════════════════════════════════════════════════════════════
# CategorizedDataset — wraps texts + category metadata
# ═══════════════════════════════════════════════════════════════════

class CategorizedDataset(Dataset):
    """A Dataset that knows its category and can filter by category."""

    def __init__(self, texts, tokenizer, max_seq_length, bos, eos, pad,
                 category=None, category_map=None, cache_dir=None):
        """
        texts: list of strings
        tokenizer: tokenizer with .encode()
        category: str — single category (for homogeneous datasets)
        category_map: list of str — per-text category labels
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.bos = bos
        self.eos = eos
        self.pad = pad
        self.category = category
        self.category_map = category_map or [category] * len(texts) if category else ["other"] * len(texts)
        self._cache_dir = Path(cache_dir) if cache_dir else None
        self._tok_cache = {}

    def filter_by_category(self, categories):
        if isinstance(categories, str):
            categories = {categories}
        else:
            categories = set(categories)
        indices = [i for i, c in enumerate(self.category_map) if c in categories]
        if not indices:
            return CategorizedDataset([], self.tokenizer, self.max_seq_length,
                                       self.bos, self.eos, self.pad)
        subset = CategorizedDataset(
            [self.texts[i] for i in indices],
            self.tokenizer, self.max_seq_length, self.bos, self.eos, self.pad,
            category_map=[self.category_map[i] for i in indices],
        )
        return subset

    def _tokenize(self, text):
        h = hashlib.md5(text.encode()).hexdigest()
        if h in self._tok_cache:
            return self._tok_cache[h]
        enc = self.tokenizer.encode(text)
        if not enc.ids:
            ids = [self.bos, self.eos]
        else:
            ids = [self.bos] + enc.ids[:self.max_seq_length - 2] + [self.eos]
        ln = len(ids)
        if ln < self.max_seq_length:
            ids += [self.pad] * (self.max_seq_length - ln)
        attn = [1] * ln + [0] * (self.max_seq_length - ln)
        lbl = ids[:ln] + [-100] * (self.max_seq_length - ln)
        lbl[0] = -100
        result = (
            torch.tensor(ids, dtype=torch.long),
            torch.tensor(attn, dtype=torch.long),
            torch.tensor(lbl, dtype=torch.long),
        )
        self._tok_cache[h] = result
        return result

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        ids, attn, labels = self._tokenize(self.texts[idx])
        return {
            "input_ids": ids,
            "attention_mask": attn,
            "labels": labels,
            "category": self.category_map[idx] if idx < len(self.category_map) else "other",
        }

    def category_stats(self):
        counts = Counter(self.category_map)
        return dict(counts)


# ═══════════════════════════════════════════════════════════════════
# Convenience: load categorized datasets from HF sources
# ═══════════════════════════════════════════════════════════════════

def load_hf_dataset(source: str, split="train", max_rows=None):
    """Load dataset from HuggingFace."""
    from datasets import load_dataset
    ds = load_dataset(source, split=split, streaming=False)
    if max_rows:
        ds = ds.select(range(min(max_rows, len(ds))))
    return ds


def extract_texts_from_hf(ds, text_field="text"):
    """Extract text field from HF dataset."""
    texts = []
    for row in ds:
        t = row.get(text_field)
        if t and isinstance(t, str) and len(t) > 20:
            texts.append(t)
    return texts


def extract_conversation_texts(ds):
    """Extract text from conversation-format datasets."""
    texts = []
    for row in ds:
        conv = row.get("conversations") or row.get("messages") or []
        for msg in conv:
            t = msg.get("content", "") if isinstance(msg, dict) else str(msg)
            if len(t) > 20:
                texts.append(t)
    return texts
