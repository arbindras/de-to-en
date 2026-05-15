"""
dataset.py — Multi30k Dataset for German→English NMT
DA6401 Assignment 3: "Attention Is All You Need"

Loads the Multi30k dataset from HuggingFace, builds vocabularies using
spaCy tokenisers, and converts sentences to integer token sequences.
"""

import torch
from torch.utils.data import Dataset
from collections import Counter
from datasets import load_dataset
import spacy

from typing import Optional

# ──────────────────────────────────────────────────────────────────────
# Helper: load spaCy models (download if missing)
# ──────────────────────────────────────────────────────────────────────

def _load_spacy(model_name: str):
    try:
        return spacy.load(model_name)
    except OSError:
        import subprocess, sys
        subprocess.run(
            [sys.executable, "-m", "spacy", "download", model_name], check=True
        )
        return spacy.load(model_name)


# ──────────────────────────────────────────────────────────────────────
# Collate function for DataLoader
# ──────────────────────────────────────────────────────────────────────

def collate_fn(batch, src_pad_idx: int = 1, tgt_pad_idx: int = 1):
    """
    Pads a batch of (src_ids, tgt_ids) lists to uniform length.

    Args:
        batch        : list of (src_ids, tgt_ids) tuples.
        src_pad_idx  : Index used for source padding.
        tgt_pad_idx  : Index used for target padding.

    Returns:
        src_batch : LongTensor [batch, max_src_len]
        tgt_batch : LongTensor [batch, max_tgt_len]
    """
    src_seqs, tgt_seqs = zip(*batch)
    max_src = max(len(s) for s in src_seqs)
    max_tgt = max(len(t) for t in tgt_seqs)

    padded_src = [s + [src_pad_idx] * (max_src - len(s)) for s in src_seqs]
    padded_tgt = [t + [tgt_pad_idx] * (max_tgt - len(t)) for t in tgt_seqs]

    return (
        torch.tensor(padded_src, dtype=torch.long),
        torch.tensor(padded_tgt, dtype=torch.long),
    )


# ──────────────────────────────────────────────────────────────────────
# Dataset class
# ──────────────────────────────────────────────────────────────────────

class Multi30kDataset(Dataset):
    """
    Wraps the bentrevett/multi30k HuggingFace dataset for German→English NMT.

    Workflow:
        1. Instantiate with split='train' / 'validation' / 'test'.
        2. Call build_vocab() on the training split.
        3. Pass the resulting vocabs to other splits via build_vocab().
        4. Call process_data() on every split.
        5. Use as a standard PyTorch Dataset with the provided collate_fn.

    Special tokens (same indices for both vocabs):
        <pad> → 1   <unk> → 0   <sos> → 2   <eos> → 3
    """

    # Special tokens
    PAD_TOKEN     = "<pad>"
    UNK_TOKEN     = "<unk>"
    SOS_TOKEN     = "<sos>"
    EOS_TOKEN     = "<eos>"
    SPECIAL_TOKENS = [UNK_TOKEN, PAD_TOKEN, SOS_TOKEN, EOS_TOKEN]
    # Indices:          0           1          2           3

    PAD_IDX = 1
    UNK_IDX = 0
    SOS_IDX = 2
    EOS_IDX = 3

    def __init__(self, split: str = "train") -> None:
        """
        Loads the Multi30k dataset and prepares tokenisers.

        Args:
            split : One of 'train', 'validation', 'test'.
        """
        self.split   = split
        self.dataset = load_dataset("bentrevett/multi30k", split=split)

        # spaCy tokenisers
        self.spacy_de = _load_spacy("de_core_news_sm")
        self.spacy_en = _load_spacy("en_core_web_sm")

        self.src_vocab: dict | None = None   # de → int
        self.tgt_vocab: dict | None = None   # en → int
        self.data:      list | None = None   # list of (src_ids, tgt_ids)

    # ── tokenisers ────────────────────────────────────────────────────

    def tokenize_de(self, text: str) -> list:
        """Lowercase German tokenisation via spaCy."""
        return [tok.text.lower() for tok in self.spacy_de(text.strip())]

    def tokenize_en(self, text: str) -> list:
        """Lowercase English tokenisation via spaCy."""
        return [tok.text.lower() for tok in self.spacy_en(text.strip())]

    # ── vocabulary ────────────────────────────────────────────────────

    def build_vocab(
        self,
        min_freq: int = 2,
        src_vocab: Optional[dict] = None,
        tgt_vocab: Optional[dict] = None,
    ) -> None:
        """
        Builds (or receives) the vocabulary mappings for source (de) and
        target (en), including all four special tokens.

        Call on the training split without arguments to build from scratch.
        Pass `src_vocab` and `tgt_vocab` to reuse a training-split vocab
        on validation / test splits.

        Args:
            min_freq  : Minimum token frequency to include in vocabulary.
            src_vocab : Pre-built source vocab dict (token → index).
            tgt_vocab : Pre-built target vocab dict (token → index).
        """
        if src_vocab is not None and tgt_vocab is not None:
            self.src_vocab = src_vocab
            self.tgt_vocab = tgt_vocab
            return

        # Count token frequencies over this split
        src_counter: Counter = Counter()
        tgt_counter: Counter = Counter()

        for example in self.dataset:
            src_counter.update(self.tokenize_de(example["de"]))
            tgt_counter.update(self.tokenize_en(example["en"]))

        # Build vocab: special tokens first, then by frequency (descending)
        self.src_vocab = {tok: idx for idx, tok in enumerate(self.SPECIAL_TOKENS)}
        for tok, freq in src_counter.most_common():
            if freq >= min_freq and tok not in self.src_vocab:
                self.src_vocab[tok] = len(self.src_vocab)

        self.tgt_vocab = {tok: idx for idx, tok in enumerate(self.SPECIAL_TOKENS)}
        for tok, freq in tgt_counter.most_common():
            if freq >= min_freq and tok not in self.tgt_vocab:
                self.tgt_vocab[tok] = len(self.tgt_vocab)

    # ── data processing ───────────────────────────────────────────────

    def process_data(self) -> None:
        """
        Converts German and English sentences into integer token lists using
        the spaCy tokenisers and the built vocabulary.

        Must call build_vocab() before process_data().
        """
        assert self.src_vocab is not None and self.tgt_vocab is not None, (
            "Call build_vocab() before process_data()."
        )

        self.data = []
        for example in self.dataset:
            # Source: <sos> tokens <eos>
            src_ids = (
                [self.SOS_IDX]
                + [self.src_vocab.get(t, self.UNK_IDX) for t in self.tokenize_de(example["de"])]
                + [self.EOS_IDX]
            )
            # Target: <sos> tokens <eos>
            tgt_ids = (
                [self.SOS_IDX]
                + [self.tgt_vocab.get(t, self.UNK_IDX) for t in self.tokenize_en(example["en"])]
                + [self.EOS_IDX]
            )
            self.data.append((src_ids, tgt_ids))

    # ── PyTorch Dataset interface ──────────────────────────────────────

    def __len__(self) -> int:
        assert self.data is not None, "Call process_data() first."
        return len(self.data)

    def __getitem__(self, idx: int):
        assert self.data is not None, "Call process_data() first."
        return self.data[idx]

    # ── convenience properties ─────────────────────────────────────────

    @property
    def src_vocab_size(self) -> int:
        assert self.src_vocab is not None
        return len(self.src_vocab)

    @property
    def tgt_vocab_size(self) -> int:
        assert self.tgt_vocab is not None
        return len(self.tgt_vocab)