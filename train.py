"""
train.py — Training Pipeline for German→English NMT Transformer
DA6401 Assignment 3: "Attention Is All You Need"

Usage:
    python train.py [--d_model 256] [--N 3] [--num_heads 8] [--d_ff 512]
                    [--dropout 0.1] [--warmup 4000] [--epochs 20]
                    [--batch_size 128] [--max_len 100] [--min_freq 2]
                    [--checkpoint best_model.pth] [--wandb_project DA6401_A3]
"""

import argparse
import math
import os
import time
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import wandb

from dataset import Multi30kDataset, collate_fn
from model import Transformer, make_src_mask, make_tgt_mask
from lr_scheduler import NoamScheduler


# ══════════════════════════════════════════════════════════════════════
#  LABEL SMOOTHING LOSS
# ══════════════════════════════════════════════════════════════════════

class LabelSmoothingLoss(nn.Module):
    """
    Cross-entropy loss with label smoothing (ε = smoothing).

    True label probability becomes (1 - ε); the remaining ε is spread
    uniformly across all other classes, excluding the padding index.

    Reference: §5.4 of "Attention Is All You Need".
    """

    def __init__(self, vocab_size: int, pad_idx: int = 1, smoothing: float = 0.1) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx    = pad_idx
        self.smoothing  = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits  : shape [N, vocab_size]  (raw model output, NOT softmaxed)
            targets : shape [N]              (ground-truth token indices)

        Returns:
            Scalar mean loss over non-padding positions.
        """
        V = self.vocab_size
        # Log-softmax for KL divergence
        log_probs = F.log_softmax(logits, dim=-1)           # [N, V]

        # Build smoothed target distribution
        # Every class gets smoothing / (V - 2) probability
        # (exclude the correct class and the pad class)
        with torch.no_grad():
            smooth_val = self.smoothing / max(V - 2, 1)
            dist = torch.full_like(log_probs, smooth_val)   # [N, V]
            dist[:, self.pad_idx] = 0.0                     # no prob mass on <pad>
            dist.scatter_(1, targets.unsqueeze(1), self.confidence)

        # KL divergence: sum(dist * (-log_probs)); ignore pad positions
        loss = -(dist * log_probs).sum(dim=-1)              # [N]

        # Mask out padding positions
        non_pad = (targets != self.pad_idx)
        loss = loss[non_pad]

        return loss.mean() if non_pad.any() else loss.sum()


# ══════════════════════════════════════════════════════════════════════
#  GREEDY DECODING  (standalone, operates on a batch)
# ══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def greedy_decode(
    model:    Transformer,
    src:      torch.Tensor,
    src_mask: torch.Tensor,
    tgt_sos:  int,
    tgt_eos:  int,
    tgt_pad:  int,
    max_len:  int = 100,
    device:   str = "cpu",
) -> torch.Tensor:
    """
    Autoregressive greedy decoding for a batch.

    Args:
        model    : Trained Transformer.
        src      : Source token ids, [batch, src_len].
        src_mask : Encoder padding mask, [batch, 1, 1, src_len].
        tgt_sos  : <sos> index in target vocab.
        tgt_eos  : <eos> index in target vocab.
        tgt_pad  : <pad> index in target vocab.
        max_len  : Maximum generation length.
        device   : Torch device string.

    Returns:
        ys : Generated token ids, [batch, gen_len] (includes leading <sos>).
    """
    model.eval()
    batch = src.size(0)

    memory = model.encode(src, src_mask)   # [batch, src_len, d_model]
    ys     = torch.full((batch, 1), tgt_sos, dtype=torch.long, device=device)
    done   = torch.zeros(batch, dtype=torch.bool, device=device)

    for _ in range(max_len):
        tgt_mask = make_tgt_mask(ys, pad_idx=tgt_pad)
        logits   = model.decode(memory, src_mask, ys, tgt_mask)   # [batch, t, V]
        next_tok = logits[:, -1, :].argmax(dim=-1)                # [batch]
        ys       = torch.cat([ys, next_tok.unsqueeze(1)], dim=1)
        done    |= (next_tok == tgt_eos)
        if done.all():
            break

    return ys


# ══════════════════════════════════════════════════════════════════════
#  BLEU EVALUATION
# ══════════════════════════════════════════════════════════════════════

def evaluate_bleu(
    model:      Transformer,
    loader:     DataLoader,
    tgt_vocab:  dict,
    tgt_pad:    int,
    tgt_sos:    int,
    tgt_eos:    int,
    device:     str,
    max_len:    int = 100,
) -> float:
    """
    Compute corpus-level BLEU score on a DataLoader.

    Returns:
        BLEU score in [0, 100].
    """
    import sacrebleu
    inv_vocab = {v: k for k, v in tgt_vocab.items()}
    model.eval()

    predictions, references = [], []

    for src_batch, tgt_batch in loader:
        src_batch = src_batch.to(device)
        src_mask  = make_src_mask(src_batch, pad_idx=tgt_pad)

        ys = greedy_decode(
            model, src_batch, src_mask,
            tgt_sos=tgt_sos, tgt_eos=tgt_eos, tgt_pad=tgt_pad,
            max_len=max_len, device=device,
        )

        # Decode predictions and references
        for pred_ids in ys:
            toks = [inv_vocab.get(i.item(), "<unk>") for i in pred_ids[1:]]
            if "<eos>" in toks:
                toks = toks[: toks.index("<eos>")]
            predictions.append(" ".join(toks))

        for ref_ids in tgt_batch:
            toks = [inv_vocab.get(i.item(), "<unk>") for i in ref_ids[1:]]
            if "<eos>" in toks:
                toks = toks[: toks.index("<eos>")]
            references.append([" ".join(toks)])

    bleu = sacrebleu.corpus_bleu(
    predictions,
    list(zip(*references))
    )

    return bleu.score


# ══════════════════════════════════════════════════════════════════════
#  ONE EPOCH
# ══════════════════════════════════════════════════════════════════════

def run_epoch(
    model:      Transformer,
    loader:     DataLoader,
    criterion:  LabelSmoothingLoss,
    optimizer,
    scheduler,
    device:     str,
    train:      bool = True,
    step:       int  = 0,
) -> tuple:
    """
    Run one pass (train or eval) over the DataLoader.

    Returns:
        (mean_loss, updated_global_step)
    """
    model.train() if train else model.eval()
    total_loss  = 0.0
    total_tokens = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for src_batch, tgt_batch in loader:
            src_batch = src_batch.to(device)   # [B, src_len]
            tgt_batch = tgt_batch.to(device)   # [B, tgt_len]

            # Masks
            src_mask = make_src_mask(src_batch, pad_idx=Multi30kDataset.PAD_IDX)
            tgt_mask = make_tgt_mask(tgt_batch[:, :-1], pad_idx=Multi30kDataset.PAD_IDX)

            # Forward: teacher forcing — feed all but last target token
            logits = model(
                src_batch,
                tgt_batch[:, :-1],   # decoder input:  <sos> t1 t2 … t_{n-1}
                src_mask,
                tgt_mask,
            )                        # [B, tgt_len-1, V]

            # Targets: shift by 1 → t1 t2 … t_n <eos>
            tgt_out = tgt_batch[:, 1:].contiguous()   # [B, tgt_len-1]

            # Compute loss
            B, T, V = logits.shape
            loss = criterion(logits.view(B * T, V), tgt_out.view(B * T))

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                step += 1

            # Accumulate (weight by non-pad tokens for fair averaging)
            n_tok = (tgt_out != Multi30kDataset.PAD_IDX).sum().item()
            total_loss   += loss.item() * n_tok
            total_tokens += n_tok

    mean_loss = total_loss / max(total_tokens, 1)
    return mean_loss, step


# ══════════════════════════════════════════════════════════════════════
#  MAIN TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ── Weights & Biases ──────────────────────────────────────────────
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        config=vars(args),
        name=args.run_name,
    )

    # ── Datasets ──────────────────────────────────────────────────────
    print("Loading datasets …")
    train_ds = Multi30kDataset(split="train")
    val_ds   = Multi30kDataset(split="validation")
    test_ds  = Multi30kDataset(split="test")

    print("Building vocabulary …")
    train_ds.build_vocab(min_freq=args.min_freq)
    val_ds.build_vocab(src_vocab=train_ds.src_vocab, tgt_vocab=train_ds.tgt_vocab)
    test_ds.build_vocab(src_vocab=train_ds.src_vocab, tgt_vocab=train_ds.tgt_vocab)

    print("Processing data …")
    train_ds.process_data()
    val_ds.process_data()
    test_ds.process_data()

    print(f"  src vocab size : {train_ds.src_vocab_size}")
    print(f"  tgt vocab size : {train_ds.tgt_vocab_size}")
    print(f"  train samples  : {len(train_ds)}")
    print(f"  val   samples  : {len(val_ds)}")

    _collate = partial(
        collate_fn,
        src_pad_idx=Multi30kDataset.PAD_IDX,
        tgt_pad_idx=Multi30kDataset.PAD_IDX,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=_collate, num_workers=2, pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=_collate, num_workers=2,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=_collate, num_workers=2,
    )

    # ── Model ─────────────────────────────────────────────────────────
    print("Building model …")
    model = Transformer(
        src_vocab_size=train_ds.src_vocab_size,
        tgt_vocab_size=train_ds.tgt_vocab_size,
        d_model=args.d_model,
        N=args.N,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        dropout=args.dropout,
        src_vocab=train_ds.src_vocab,
        tgt_vocab=train_ds.tgt_vocab,
        src_tokenizer=train_ds.spacy_de,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params:,}")
    wandb.config.update({"n_params": n_params})

    # ── Loss, Optimiser, Scheduler ────────────────────────────────────
    criterion = LabelSmoothingLoss(
        vocab_size=train_ds.tgt_vocab_size,
        pad_idx=Multi30kDataset.PAD_IDX,
        smoothing=0.1,
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9
    )
    scheduler = NoamScheduler(optimizer, d_model=args.d_model, warmup_steps=args.warmup)

    # ── Training ──────────────────────────────────────────────────────
    best_val_loss = float("inf")
    global_step   = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, global_step = run_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            device=device, train=True, step=global_step,
        )
        val_loss, _ = run_epoch(
            model, val_loader, criterion, optimizer, scheduler,
            device=device, train=False, step=global_step,
        )

        elapsed = time.time() - t0
        lr_now  = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | "
            f"lr {lr_now:.2e} | {elapsed:.1f}s"
        )

        wandb.log(
            {
                "epoch":      epoch,
                "train_loss": train_loss,
                "val_loss":   val_loss,
                "lr":         lr_now,
                "step":       global_step,
                "train_ppl":  math.exp(min(train_loss, 10)),
                "val_ppl":    math.exp(min(val_loss,   10)),
            }
        )

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "src_vocab": train_ds.src_vocab,
                    "tgt_vocab": train_ds.tgt_vocab,
                },
                args.checkpoint,
            )
            print(f"  ✓ New best model saved → {args.checkpoint}")
            wandb.run.summary["best_val_loss"] = best_val_loss
            wandb.run.summary["best_epoch"]    = epoch

    # ── Final evaluation on test set ─────────────────────────────────
    print("\nLoading best checkpoint for final BLEU evaluation …")
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    test_bleu = evaluate_bleu(
        model, test_loader,
        tgt_vocab=train_ds.tgt_vocab,
        tgt_pad=Multi30kDataset.PAD_IDX,
        tgt_sos=Multi30kDataset.SOS_IDX,
        tgt_eos=Multi30kDataset.EOS_IDX,
        device=device,
        max_len=args.max_len,
    )
    print(f"Test BLEU: {test_bleu:.2f}")
    wandb.run.summary["test_bleu"] = test_bleu
    wandb.log({"test_bleu": test_bleu})

    # ── Quick demo translations ───────────────────────────────────────
    demo_sentences = [
        "Zwei junge weiße Männer sind im Freien in der Nähe vieler Büsche.",
        "Ein Mann in einem blauen Hemd steht auf einer Leiter und reinigt ein Fenster.",
        "Eine Gruppe von Menschen steht vor einem Iglu.",
    ]
    print("\nSample translations:")
    for src_sent in demo_sentences:
        translation = model.infer(src_sent, max_len=args.max_len, device=device)
        print(f"  DE: {src_sent}")
        print(f"  EN: {translation}\n")

    wandb.finish()


# ══════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Transformer NMT (DE→EN)")

    # Model hyper-parameters
    parser.add_argument("--d_model",    type=int,   default=256,
                        help="Model dimensionality (paper: 512, small: 256)")
    parser.add_argument("--N",          type=int,   default=3,
                        help="Number of encoder/decoder layers (paper: 6, small: 3)")
    parser.add_argument("--num_heads",  type=int,   default=8,
                        help="Number of attention heads")
    parser.add_argument("--d_ff",       type=int,   default=512,
                        help="FFN inner dimensionality (paper: 2048, small: 512)")
    parser.add_argument("--dropout",    type=float, default=0.1,
                        help="Dropout probability")

    # Training hyper-parameters
    parser.add_argument("--warmup",     type=int,   default=4000,
                        help="Noam warm-up steps")
    parser.add_argument("--epochs",     type=int,   default=20,
                        help="Total training epochs")
    parser.add_argument("--batch_size", type=int,   default=128,
                        help="Batch size")
    parser.add_argument("--max_len",    type=int,   default=100,
                        help="Maximum generation length during inference")
    parser.add_argument("--min_freq",   type=int,   default=2,
                        help="Minimum token frequency for vocabulary inclusion")

    # I/O
    parser.add_argument("--checkpoint",    type=str, default="best_model.pth",
                        help="Path to save best model checkpoint")
    parser.add_argument("--wandb_project", type=str, default="DA6401_A3",
                        help="W&B project name")
    parser.add_argument("--wandb_entity",  type=str, default="arbindrapatel-iitmaana",
                        help="W&B entity (team / username)")
    parser.add_argument("--run_name",      type=str, default=None,
                        help="W&B run name (auto-generated if None)")

    args, _ = parser.parse_known_args()
    train(args)