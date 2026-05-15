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

from sympy import re

import torch
import torch.nn as nn
import torch.nn.functional as F

import gdown
import spacy
from typing import Optional

# ══════════════════════════════════════════════════════════════════════
#  STANDALONE ATTENTION FUNCTION
#  Exposed at module level so the autograder can import and test it
#  independently of MultiHeadAttention.
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
        Q    : Query tensor,  shape (..., seq_q, d_k)
        K    : Key tensor,    shape (..., seq_k, d_k)
        V    : Value tensor,  shape (..., seq_k, d_v)
        mask : Optional Boolean mask, shape broadcastable to
               (..., seq_q, seq_k).
               Positions where mask is True are MASKED OUT
               (set to -inf before softmax).

    Returns:
        output : Attended output,   shape (..., seq_q, d_v)
        attn_w : Attention weights, shape (..., seq_q, seq_k)
    """
    d_k = Q.size(-1)
    # (..., seq_q, seq_k)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))

    attn_w = F.softmax(scores, dim=-1)
    # Replace NaN rows (all -inf → softmax → NaN) with 0 for pure pad rows
    attn_w = torch.nan_to_num(attn_w, nan=0.0)

    output = torch.matmul(attn_w, V)
    return output, attn_w


# ══════════════════════════════════════════════════════════════════════
#  MASK HELPERS
#  Exposed at module level so they can be tested independently and
#  reused inside Transformer.forward.
# ══════════════════════════════════════════════════════════════════════

def make_src_mask(
    src: torch.Tensor,
    pad_idx: int = 1,
) -> torch.Tensor:
    """
    Build a padding mask for the encoder (source sequence).

    Args:
        src     : Source token-index tensor, shape [batch, src_len]
        pad_idx : Vocabulary index of the <pad> token (default 1)

    Returns:
        Boolean mask, shape [batch, 1, 1, src_len]
        True  → position is a PAD token (will be masked out)
        False → real token
    """
    # [batch, src_len] → [batch, 1, 1, src_len]
    return (src == pad_idx).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(
    tgt: torch.Tensor,
    pad_idx: int = 1,
) -> torch.Tensor:
    """
    Build a combined padding + causal (look-ahead) mask for the decoder.

    Args:
        tgt     : Target token-index tensor, shape [batch, tgt_len]
        pad_idx : Vocabulary index of the <pad> token (default 1)

    Returns:
        Boolean mask, shape [batch, 1, tgt_len, tgt_len]
        True → position is masked out (PAD or future token)
    """
    tgt_len = tgt.size(1)

    # Padding mask: [batch, 1, 1, tgt_len]
    pad_mask = (tgt == pad_idx).unsqueeze(1).unsqueeze(2)

    # Causal (look-ahead) mask: upper-triangular [1, 1, tgt_len, tgt_len]
    causal_mask = torch.triu(
        torch.ones(tgt_len, tgt_len, device=tgt.device, dtype=torch.bool),
        diagonal=1,
    ).unsqueeze(0).unsqueeze(0)   # [1, 1, tgt_len, tgt_len]

    # Combine: True if PAD *or* future position → [batch, 1, tgt_len, tgt_len]
    return pad_mask | causal_mask


# ══════════════════════════════════════════════════════════════════════
#  MULTI-HEAD ATTENTION
# ══════════════════════════════════════════════════════════════════════

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention as in "Attention Is All You Need", §3.2.2.

        MultiHead(Q,K,V) = Concat(head_1,...,head_h) · W_O
        head_i = Attention(Q·W_Qi, K·W_Ki, V·W_Vi)

    Args:
        d_model   (int)  : Total model dimensionality. Must be divisible by num_heads.
        num_heads (int)  : Number of parallel attention heads h.
        dropout   (float): Dropout probability applied to attention weights.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads   # depth per head

        # Projection matrices
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(p=dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """[batch, seq, d_model] → [batch, heads, seq, d_k]"""
        batch, seq, _ = x.size()
        return x.view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)

    def forward(
        self,
        query: torch.Tensor,
        key:   torch.Tensor,
        value: torch.Tensor,
        mask:  Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query : shape [batch, seq_q, d_model]
            key   : shape [batch, seq_k, d_model]
            value : shape [batch, seq_k, d_model]
            mask  : Optional BoolTensor broadcastable to
                    [batch, num_heads, seq_q, seq_k]
                    True → masked out (attend nowhere)

        Returns:
            output : shape [batch, seq_q, d_model]
        """
        batch = query.size(0)

        # Linear projections + split into heads
        Q = self._split_heads(self.W_q(query))   # [batch, heads, seq_q, d_k]
        K = self._split_heads(self.W_k(key))     # [batch, heads, seq_k, d_k]
        V = self._split_heads(self.W_v(value))   # [batch, heads, seq_k, d_k]

        # Scaled dot-product attention
        out, _ = scaled_dot_product_attention(Q, K, V, mask)
        # out: [batch, heads, seq_q, d_k]

        # Concatenate heads and project back
        out = (
            out.transpose(1, 2)           # [batch, seq_q, heads, d_k]
               .contiguous()
               .view(batch, -1, self.d_model)   # [batch, seq_q, d_model]
        )
        return self.W_o(out)


# ══════════════════════════════════════════════════════════════════════
#  POSITIONAL ENCODING
# ══════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding as in "Attention Is All You Need", §3.5.

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Args:
        d_model  (int)  : Embedding dimensionality.
        dropout  (float): Dropout applied after adding encodings.
        max_len  (int)  : Maximum sequence length to pre-compute (default 5000).
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Build PE table: [1, max_len, d_model]
        pe = torch.zeros(1, max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)   # [max_len, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )                                                                        # [d_model/2]

        pe[0, :, 0::2] = torch.sin(position * div_term)   # even dims
        pe[0, :, 1::2] = torch.cos(position * div_term)   # odd dims

        # Register as non-trainable buffer
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : Input embeddings, shape [batch, seq_len, d_model]

        Returns:
            Tensor of same shape [batch, seq_len, d_model]
            = x  +  PE[:, :seq_len, :]
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ══════════════════════════════════════════════════════════════════════
#  FEED-FORWARD NETWORK
# ══════════════════════════════════════════════════════════════════════

class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network, §3.3:

        FFN(x) = max(0, x·W₁ + b₁)·W₂ + b₂

    Args:
        d_model (int)  : Input / output dimensionality (e.g. 512).
        d_ff    (int)  : Inner-layer dimensionality (e.g. 2048).
        dropout (float): Dropout applied between the two linears.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : shape [batch, seq_len, d_model]
        Returns:
              shape [batch, seq_len, d_model]
        """
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# ══════════════════════════════════════════════════════════════════════
#  ENCODER LAYER
# ══════════════════════════════════════════════════════════════════════

class EncoderLayer(nn.Module):
    """
    Single Transformer encoder sub-layer (Post-LayerNorm as in the paper):
        x → Self-Attention → Add & Norm → FFN → Add & Norm

    Justification for Post-LN: we follow the original paper exactly. While
    Pre-LN (norm before sub-layer) can train more stably without warm-up,
    Post-LN matches the paper and benefits from the Noam warm-up schedule.

    Args:
        d_model   (int)  : Model dimensionality.
        num_heads (int)  : Number of attention heads.
        d_ff      (int)  : FFN inner dimensionality.
        dropout   (float): Dropout probability.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff        = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1     = nn.LayerNorm(d_model)
        self.norm2     = nn.LayerNorm(d_model)
        self.dropout   = nn.Dropout(p=dropout)
        self.d_model   = d_model

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x        : shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]

        Returns:
            shape [batch, src_len, d_model]
        """
        # Sublayer 1: Self-attention + Add & Norm
        attn_out = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + self.dropout(attn_out))

        # Sublayer 2: FFN + Add & Norm
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))

        return x


# ══════════════════════════════════════════════════════════════════════
#  DECODER LAYER
# ══════════════════════════════════════════════════════════════════════

class DecoderLayer(nn.Module):
    """
    Single Transformer decoder sub-layer (Post-LayerNorm):
        x → Masked Self-Attn → Add & Norm
          → Cross-Attn(memory) → Add & Norm
          → FFN → Add & Norm

    Args:
        d_model   (int)  : Model dimensionality.
        num_heads (int)  : Number of attention heads.
        d_ff      (int)  : FFN inner dimensionality.
        dropout   (float): Dropout probability.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn  = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff         = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1      = nn.LayerNorm(d_model)
        self.norm2      = nn.LayerNorm(d_model)
        self.norm3      = nn.LayerNorm(d_model)
        self.dropout    = nn.Dropout(p=dropout)
        self.d_model    = d_model

    def forward(
        self,
        x:        torch.Tensor,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x        : shape [batch, tgt_len, d_model]
            memory   : Encoder output, shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]

        Returns:
            shape [batch, tgt_len, d_model]
        """
        # Sublayer 1: Masked self-attention + Add & Norm
        self_attn_out = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_out))

        # Sublayer 2: Cross-attention + Add & Norm
        cross_out = self.cross_attn(x, memory, memory, src_mask)
        x = self.norm2(x + self.dropout(cross_out))

        # Sublayer 3: FFN + Add & Norm
        ff_out = self.ff(x)
        x = self.norm3(x + self.dropout(ff_out))

        return x


# ══════════════════════════════════════════════════════════════════════
#  ENCODER & DECODER STACKS
# ══════════════════════════════════════════════════════════════════════

class Encoder(nn.Module):
    """Stack of N identical EncoderLayer modules with final LayerNorm."""

    def __init__(self, layer: EncoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm   = nn.LayerNorm(layer.d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x    : shape [batch, src_len, d_model]
            mask : shape [batch, 1, 1, src_len]
        Returns:
            shape [batch, src_len, d_model]
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    """Stack of N identical DecoderLayer modules with final LayerNorm."""

    def __init__(self, layer: DecoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm   = nn.LayerNorm(layer.d_model)

    def forward(
        self,
        x:        torch.Tensor,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x        : shape [batch, tgt_len, d_model]
            memory   : shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]
        Returns:
            shape [batch, tgt_len, d_model]
        """
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


# ══════════════════════════════════════════════════════════════════════
#  FULL TRANSFORMER
# ══════════════════════════════════════════════════════════════════════

class Transformer(nn.Module):
    """
    Full Encoder-Decoder Transformer for sequence-to-sequence tasks.

    Args:
        src_vocab_size (int)  : Source vocabulary size.
        tgt_vocab_size (int)  : Target vocabulary size.
        d_model        (int)  : Model dimensionality (default 512).
        N              (int)  : Number of encoder/decoder layers (default 6).
        num_heads      (int)  : Number of attention heads (default 8).
        d_ff           (int)  : FFN inner dimensionality (default 2048).
        dropout        (float): Dropout probability (default 0.1).
        checkpoint_path(str)  : Optional path to load saved weights.
        src_vocab      (dict) : Token→index vocab for source language (for inference).
        tgt_vocab      (dict) : Token→index vocab for target language (for inference).
        src_tokenizer         : spaCy tokenizer for source language (for inference).
    """

    def __init__(
        self,
        src_vocab_size: int = 7853,
        tgt_vocab_size: int = 5893,
        d_model:   int   = 256,
        N:         int   = 3,
        num_heads: int   = 8,
        d_ff:      int   = 512,
        dropout:   float = 0.1,
        checkpoint_path: Optional[str] = None,
        src_vocab: Optional[dict] = None,
        tgt_vocab: Optional[dict] = None,
        src_tokenizer = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        # ------------------------------------------------------------
        # Default tokenizer
        # ------------------------------------------------------------
        self.src_tokenizer = None

        try:
            self.src_tokenizer = spacy.load("de_core_news_sm").tokenizer
        except Exception:
            self.src_tokenizer = None


        # ------------------------------------------------------------
        # Auto-download checkpoint from Google Drive
        # ------------------------------------------------------------
        if checkpoint_path is None:
            checkpoint_path = "best_model.pth"

        if not os.path.exists(checkpoint_path):

            FILE_ID = "17T8YR7UwBHtB-KSE9gSFVZpw5eIUpUKX"

            url = f"https://drive.google.com/uc?id={FILE_ID}"

            gdown.download(url, checkpoint_path, quiet=False)


        # Embeddings
        self.src_embed = nn.Embedding(src_vocab_size, d_model, padding_idx=1)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model, padding_idx=1)

        # Positional encodings
        self.src_pe = PositionalEncoding(d_model, dropout)
        self.tgt_pe = PositionalEncoding(d_model, dropout)

        # Encoder / Decoder stacks
        enc_layer = EncoderLayer(d_model, num_heads, d_ff, dropout)
        dec_layer = DecoderLayer(d_model, num_heads, d_ff, dropout)
        self.encoder = Encoder(enc_layer, N)
        self.decoder = Decoder(dec_layer, N)

        # Final linear projection to vocabulary
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

        # Vocab / tokenizer references (needed for .infer())
        self.src_vocab      = src_vocab
        self.tgt_vocab      = tgt_vocab
        self.src_tokenizer  = src_tokenizer

        # Weight initialisation (Xavier uniform, as in the paper)
        self._init_weights()

        # ------------------------------------------------------------
        # Load checkpoint
        # ------------------------------------------------------------
        state = torch.load(checkpoint_path, map_location="cpu")

        # load weights
        if "model_state" in state:
            self.load_state_dict(state["model_state"])
        else:
            self.load_state_dict(state)

        # load vocabs
        self.src_vocab = state["src_vocab"]
        self.tgt_vocab = state["tgt_vocab"]

    # ── weight init ────────────────────────────────────────────────────
    def _init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # ── AUTOGRADER HOOKS ── keep these signatures exactly ─────────────

    def encode(
        self,
        src:      torch.Tensor,
        src_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run the full encoder stack.

        Args:
            src      : Token indices, shape [batch, src_len]
            src_mask : shape [batch, 1, 1, src_len]

        Returns:
            memory : Encoder output, shape [batch, src_len, d_model]
        """
        # Scale embeddings by √d_model (§3.4 of the paper)
        x = self.src_pe(self.src_embed(src) * math.sqrt(self.d_model))
        return self.encoder(x, src_mask)

    def decode(
        self,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt:      torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run the full decoder stack and project to vocabulary logits.

        Args:
            memory   : Encoder output,  shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]
            tgt      : Token indices,   shape [batch, tgt_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]

        Returns:
            logits : shape [batch, tgt_len, tgt_vocab_size]
        """
        x = self.tgt_pe(self.tgt_embed(tgt) * math.sqrt(self.d_model))
        x = self.decoder(x, memory, src_mask, tgt_mask)
        return self.fc_out(x)

    def forward(
        self,
        src:      torch.Tensor,
        tgt:      torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Full encoder-decoder forward pass.

        Args:
            src      : shape [batch, src_len]
            tgt      : shape [batch, tgt_len]
            src_mask : shape [batch, 1, 1, src_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]

        Returns:
            logits : shape [batch, tgt_len, tgt_vocab_size]
        """
        memory = self.encode(src, src_mask)
        return self.decode(memory, src_mask, tgt, tgt_mask)


    def infer(self, src_sentence: str, max_len: int = 50, device= None) -> str:
        """
        Translates a German sentence to English using greedy autoregressive decoding.

        Args:
            src_sentence : The raw German text.
            max_len      : Maximum tokens to generate.
            device       : Torch device string.

        Returns:
            The fully translated English string, detokenised and clean.
        """
        if device is None:
            device = next(self.parameters()).device
        
        assert self.src_vocab is not None, "src_vocab must be set for inference"
        assert self.tgt_vocab is not None, "tgt_vocab must be set for inference"
        # assert self.src_tokenizer is not None, "src_tokenizer must be set for inference"
        from re import findall
        if self.src_tokenizer is not None:
            tokens = [tok.text.lower() for tok in self.src_tokenizer(src_sentence)]
        else:
            tokens = findall(r"\w+|[^\w\s]", src_sentence.lower())

        self.eval()
        with torch.no_grad():
            # Tokenise source
            # tokens = [tok.text.lower() for tok in self.src_tokenizer(src_sentence)]
            if self.src_tokenizer is not None:
                tokens = [tok.text.lower() for tok in self.src_tokenizer(src_sentence)]
            else:
                tokens = findall(r"\w+|[^\w\s]", src_sentence.lower())

            pad_idx = self.src_vocab.get("<pad>", 1)
            sos_idx = self.src_vocab["<sos>"]
            eos_idx = self.src_vocab["<eos>"]
            unk_idx = self.src_vocab.get("<unk>", 0)

            src_ids = (
                [sos_idx]
                + [self.src_vocab.get(t, unk_idx) for t in tokens]
                + [eos_idx]
            )
            src = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(device)
            src_mask = make_src_mask(src, pad_idx=pad_idx)
            memory = self.encode(src, src_mask)

            tgt_sos = self.tgt_vocab["<sos>"]
            tgt_eos = self.tgt_vocab["<eos>"]
            tgt_pad = self.tgt_vocab.get("<pad>", 1)

            # Greedy decoding
            ys = torch.tensor([[tgt_sos]], dtype=torch.long).to(device)
            for _ in range(max_len):
                tgt_mask = make_tgt_mask(ys, pad_idx=tgt_pad)
                logits   = self.decode(memory, src_mask, ys, tgt_mask)
                next_tok = logits[:, -1, :].argmax(dim=-1)   # [batch]
                ys = torch.cat([ys, next_tok.unsqueeze(1)], dim=1)
                if next_tok.item() == tgt_eos:
                    break

            # Convert ids → words
            inv_vocab = {v: k for k, v in self.tgt_vocab.items()}
            out_tokens = [inv_vocab.get(idx.item(), "<unk>") for idx in ys[0][1:]]
            # Strip <eos> and everything after
            if "<eos>" in out_tokens:
                out_tokens = out_tokens[: out_tokens.index("<eos>")]

            sentence = " ".join(tokens)

        sentence = sentence.replace(" .", ".")
        sentence = sentence.replace(" ,", ",")
        sentence = sentence.replace(" !", "!")
        sentence = sentence.replace(" ?", "?")
        sentence = sentence.replace(" n't", "n't")

        return sentence