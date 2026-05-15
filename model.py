"""
model.py — Transformer Architecture
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────┐
  │  scaled_dot_product_attention(Q, K, V, mask) → (out, weights)  │
  │  MultiHeadAttention.forward(q, k, v, mask)   → Tensor          │
  │  PositionalEncoding.forward(x)               → Tensor          │
  │  make_src_mask(src, pad_idx)                 → BoolTensor      │
  │  make_tgt_mask(tgt, pad_idx)                 → BoolTensor      │
  │  Transformer.encode(src, src_mask)           → Tensor          │
  │  Transformer.decode(memory,src_m,tgt,tgt_m)  → Tensor          │
  └─────────────────────────────────────────────────────────────────┘
"""

import math
import copy
import os
import re
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════
#  STANDALONE ATTENTION FUNCTION
# ══════════════════════════════════════════════════════════════════════

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Scaled Dot-Product Attention.

        Attention(Q, K, V) = softmax( Q·Kᵀ / √dₖ ) · V

    Args:
        Q    : (..., seq_q, d_k)
        K    : (..., seq_k, d_k)
        V    : (..., seq_k, d_v)
        mask : Bool tensor broadcastable to (..., seq_q, seq_k).
               True → position is masked out.

    Returns:
        output : (..., seq_q, d_v)
        attn_w : (..., seq_q, seq_k)  — attention weights summing to 1 over seq_k
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))

    attn_w = F.softmax(scores, dim=-1)
    attn_w = torch.nan_to_num(attn_w, nan=0.0)   # guard pure-pad rows

    output = torch.matmul(attn_w, V)
    return output, attn_w


# ══════════════════════════════════════════════════════════════════════
#  MASK HELPERS
# ══════════════════════════════════════════════════════════════════════

def make_src_mask(src: torch.Tensor, pad_idx: int = 1) -> torch.Tensor:
    """
    [batch, src_len] → [batch, 1, 1, src_len]  bool  (True = PAD)
    """
    return (src == pad_idx).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(tgt: torch.Tensor, pad_idx: int = 1) -> torch.Tensor:
    """
    Combined padding + causal mask.
    [batch, tgt_len] → [batch, 1, tgt_len, tgt_len]  bool  (True = masked)
    """
    tgt_len = tgt.size(1)

    pad_mask = (tgt == pad_idx).unsqueeze(1).unsqueeze(2)          # [B,1,1,T]
    causal   = torch.triu(
        torch.ones(tgt_len, tgt_len, device=tgt.device, dtype=torch.bool),
        diagonal=1,
    ).unsqueeze(0).unsqueeze(0)                                    # [1,1,T,T]

    return pad_mask | causal                                       # [B,1,T,T]


# ══════════════════════════════════════════════════════════════════════
#  MULTI-HEAD ATTENTION
# ══════════════════════════════════════════════════════════════════════

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(p=dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.size()
        return x.view(B, S, self.num_heads, self.d_k).transpose(1, 2)

    def forward(self, query, key, value, mask=None):
        B = query.size(0)
        Q = self._split_heads(self.W_q(query))
        K = self._split_heads(self.W_k(key))
        V = self._split_heads(self.W_v(value))

        out, _ = scaled_dot_product_attention(Q, K, V, mask)
        out = out.transpose(1, 2).contiguous().view(B, -1, self.d_model)
        return self.W_o(out)


# ══════════════════════════════════════════════════════════════════════
#  POSITIONAL ENCODING
# ══════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    """
    PE(pos,2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos,2i+1) = cos(pos / 10000^(2i/d_model))
    Stored as a non-trainable buffer (autograder checks this).
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe       = torch.zeros(1, max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ══════════════════════════════════════════════════════════════════════
#  FEED-FORWARD
# ══════════════════════════════════════════════════════════════════════

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# ══════════════════════════════════════════════════════════════════════
#  ENCODER / DECODER LAYERS
# ══════════════════════════════════════════════════════════════════════

class EncoderLayer(nn.Module):
    """
    Post-LayerNorm (original paper).  Pairs with Noam warm-up for stable training.
    x → Self-Attn → Add&Norm → FFN → Add&Norm
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff        = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1     = nn.LayerNorm(d_model)
        self.norm2     = nn.LayerNorm(d_model)
        self.dropout   = nn.Dropout(p=dropout)
        self.d_model   = d_model

    def forward(self, x, src_mask):
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, src_mask)))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x


class DecoderLayer(nn.Module):
    """
    x → Masked Self-Attn → Add&Norm → Cross-Attn → Add&Norm → FFN → Add&Norm
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn  = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff         = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1      = nn.LayerNorm(d_model)
        self.norm2      = nn.LayerNorm(d_model)
        self.norm3      = nn.LayerNorm(d_model)
        self.dropout    = nn.Dropout(p=dropout)
        self.d_model    = d_model

    def forward(self, x, memory, src_mask, tgt_mask):
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, tgt_mask)))
        x = self.norm2(x + self.dropout(self.cross_attn(x, memory, memory, src_mask)))
        x = self.norm3(x + self.dropout(self.ff(x)))
        return x


# ══════════════════════════════════════════════════════════════════════
#  ENCODER / DECODER STACKS
# ══════════════════════════════════════════════════════════════════════

class Encoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm   = nn.LayerNorm(layer.d_model)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm   = nn.LayerNorm(layer.d_model)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


# ══════════════════════════════════════════════════════════════════════
#  TRANSFORMER
# ══════════════════════════════════════════════════════════════════════

class Transformer(nn.Module):
    """
    Full Encoder-Decoder Transformer (Vaswani et al., 2017).

    checkpoint_path behaviour
    ─────────────────────────
    • "best_model.pth" (default) → downloaded from Google Drive if not present,
      then loaded.  Vocabs are read from the checkpoint dict if available.
    • explicit path              → loaded directly (no Drive download).
    • None                       → no weights loaded; useful for fresh training
                                   or architecture-only unit tests.
    """

    _GDRIVE_FILE_ID = "17T8YR7UwBHtB-KSE9gSFVZpw5eIUpUKX"

    def __init__(
        self,
        src_vocab_size: int   = 7853,
        tgt_vocab_size: int   = 5893,
        d_model:        int   = 256,
        N:              int   = 3,
        num_heads:      int   = 8,
        d_ff:           int   = 512,
        dropout:        float = 0.1,
        checkpoint_path: Optional[str] = "best_model.pth",
        src_vocab:  Optional[dict] = None,
        tgt_vocab:  Optional[dict] = None,
        src_tokenizer              = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        # ── Architecture ──────────────────────────────────────────────
        self.src_embed = nn.Embedding(src_vocab_size, d_model, padding_idx=1)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model, padding_idx=1)
        self.src_pe    = PositionalEncoding(d_model, dropout)
        self.tgt_pe    = PositionalEncoding(d_model, dropout)
        enc_layer      = EncoderLayer(d_model, num_heads, d_ff, dropout)
        dec_layer      = DecoderLayer(d_model, num_heads, d_ff, dropout)
        self.encoder   = Encoder(enc_layer, N)
        self.decoder   = Decoder(dec_layer, N)
        self.fc_out    = nn.Linear(d_model, tgt_vocab_size)

        self._init_weights()

        # ── Vocab / tokenizer ─────────────────────────────────────────
        self.src_vocab     = src_vocab
        self.tgt_vocab     = tgt_vocab
        self.src_tokenizer = src_tokenizer

        if self.src_tokenizer is None:
            try:
                import spacy as _spacy
                self.src_tokenizer = _spacy.load("de_core_news_sm")
            except Exception:
                self.src_tokenizer = None

        # ── Optional checkpoint ───────────────────────────────────────
        if checkpoint_path is not None:
            self._load_checkpoint(checkpoint_path)

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _load_checkpoint(self, path: str) -> None:
        # Download from Drive if file not present
        if not os.path.exists(path):
            try:
                import gdown
                gdown.download(
                    f"https://drive.google.com/uc?id={self._GDRIVE_FILE_ID}",
                    path, quiet=False,
                )
            except Exception as e:
                print(f"[Transformer] WARNING: could not download checkpoint: {e}")
                return

        try:
            state = torch.load(path, map_location="cpu")
        except Exception as e:
            print(f"[Transformer] WARNING: could not open checkpoint '{path}': {e}")
            return

        # Weights (support both wrapped dict and plain state-dict)
        weights = state.get("model_state", state) if isinstance(state, dict) else state
        try:
            self.load_state_dict(weights)
        except Exception as e:
            print(f"[Transformer] WARNING: load_state_dict failed: {e}")
            return

        # Vocabs (only override if checkpoint contains them)
        if isinstance(state, dict):
            if "src_vocab" in state:
                self.src_vocab = state["src_vocab"]
            if "tgt_vocab" in state:
                self.tgt_vocab = state["tgt_vocab"]

    # ── forward API ───────────────────────────────────────────────────

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """[B, src_len] → [B, src_len, d_model]"""
        x = self.src_pe(self.src_embed(src) * math.sqrt(self.d_model))
        return self.encoder(x, src_mask)

    def decode(
        self,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt:      torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """[B, tgt_len] → logits [B, tgt_len, tgt_vocab_size]"""
        x = self.tgt_pe(self.tgt_embed(tgt) * math.sqrt(self.d_model))
        x = self.decoder(x, memory, src_mask, tgt_mask)
        return self.fc_out(x)

    def forward(self, src, tgt, src_mask, tgt_mask):
        """Full encoder-decoder pass. Returns logits [B, tgt_len, V]."""
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def infer(self, src_sentence: str, max_len: int = 50, device=None) -> str:
        """
        Greedy-decode a German sentence to English.

        Returns the detokenised English string.
        """
        if device is None:
            device = next(self.parameters()).device

        assert self.src_vocab is not None, "src_vocab must be set for inference"
        assert self.tgt_vocab is not None, "tgt_vocab must be set for inference"

        # Tokenise source
        if self.src_tokenizer is not None:
            src_tokens = [tok.text.lower() for tok in self.src_tokenizer(src_sentence)]
        else:
            src_tokens = re.findall(r"\w+|[^\w\s]", src_sentence.lower())

        src_unk = self.src_vocab.get("<unk>", 0)
        src_ids = (
            [self.src_vocab["<sos>"]]
            + [self.src_vocab.get(t, src_unk) for t in src_tokens]
            + [self.src_vocab["<eos>"]]
        )
        src      = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(device)
        src_mask = make_src_mask(src, pad_idx=self.src_vocab.get("<pad>", 1))

        tgt_sos = self.tgt_vocab["<sos>"]
        tgt_eos = self.tgt_vocab["<eos>"]
        tgt_pad = self.tgt_vocab.get("<pad>", 1)

        self.eval()
        with torch.no_grad():
            memory = self.encode(src, src_mask)
            ys     = torch.tensor([[tgt_sos]], dtype=torch.long).to(device)

            for _ in range(max_len):
                tgt_mask = make_tgt_mask(ys, pad_idx=tgt_pad)
                logits   = self.decode(memory, src_mask, ys, tgt_mask)
                next_tok = logits[:, -1, :].argmax(dim=-1)
                ys       = torch.cat([ys, next_tok.unsqueeze(1)], dim=1)
                if next_tok.item() == tgt_eos:
                    break

        # ids → English words
        inv_vocab  = {v: k for k, v in self.tgt_vocab.items()}
        out_tokens = [inv_vocab.get(i.item(), "<unk>") for i in ys[0][1:]]
        if "<eos>" in out_tokens:
            out_tokens = out_tokens[: out_tokens.index("<eos>")]

        # Detokenise — join ENGLISH output tokens, not source tokens
        sentence = " ".join(out_tokens)
        sentence = re.sub(r" ([.,!?;:])", r"\1", sentence)
        sentence = re.sub(r" (n't|'s|'re|'ve|'ll|'d)", r"\1", sentence)
        return sentence