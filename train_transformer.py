"""
Train the NextBallTransformer on IPL ball-by-ball data.

Run from /Users/sarveshs/Desktop/ID5030/:
    python3 train_transformer.py

Saves trained model to  artifacts/transformer_model.pt
Updates                 artifacts/som_artifacts.pkl  with transformer metadata.
Expected wall time      ~6 min on CPU (MPS/CUDA = ~1 min)
"""

import os, time, pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from transformer_model import (
    NextBallTransformer, NUM_OUTCOME_CLASSES, BOS_TOKEN, NUM_FEATURES
)

DEVICE = (
    "mps"  if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available()          else
    "cpu"
)

# ── Hyper-params ──────────────────────────────────────────────────────────────
D_MODEL    = 64
N_HEADS    = 4
N_LAYERS   = 3
DROPOUT    = 0.10
BATCH_SIZE = 48
EPOCHS     = 12
LR         = 2e-3
GRAD_CLIP  = 1.0
SEED       = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


# ─── 1. Data loading & feature engineering ───────────────────────────────────

def load_sequences(csv_path: str = "IPL_with_match_types.csv"):
    print("Loading IPL data …")
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"  {len(df):,} deliveries, {df['match_id'].nunique()} matches")

    # ── Basic flags ──────────────────────────────────────────────────────────
    df["is_wicket"] = df["wicket_kind"].notna().astype(int)
    df["wides"]   = df.get("wides",   pd.Series(0, index=df.index)).fillna(0)
    df["noballs"] = df.get("noballs", pd.Series(0, index=df.index)).fillna(0)
    df["valid_ball"] = ((df["wides"] == 0) & (df["noballs"] == 0)).astype(int)
    df["runs_batter"] = df["runs_batter"].fillna(0).astype(int)
    df["runs_total"]  = df["runs_total"].fillna(0).astype(int)

    # ── Outcome class ────────────────────────────────────────────────────────
    def _cls(row):
        if row["is_wicket"] == 1 and row["valid_ball"] == 1:
            return 6
        if row["valid_ball"] == 0:
            return 7
        r = int(row["runs_batter"])
        return min(r, 5)   # 0-4 as-is, 5-or-6 → class 5

    df["outcome_class"] = df.apply(_cls, axis=1).astype(np.int64)

    # ── Sort chronologically ────────────────────────────────────────────────
    df = df.sort_values(["match_id", "innings", "over", "ball"],
                        na_position="last").reset_index(drop=True)

    # ── Running stats (pre-ball, no leakage) ────────────────────────────────
    grp = ["match_id", "innings"]
    grp_ov = ["match_id", "innings", "over"]

    df["_cr"]  = (df.groupby(grp)["runs_total"]
                  .transform(lambda x: x.cumsum().shift(1).fillna(0)))
    df["_cw"]  = (df.groupby(grp)["is_wicket"]
                  .transform(lambda x: x.cumsum().shift(1).fillna(0)))
    df["_vb"]  = (df.groupby(grp)["valid_ball"]
                  .transform(lambda x: x.cumsum().shift(1).fillna(0)))
    df["_orns"]= (df.groupby(grp_ov)["runs_total"]
                  .transform(lambda x: x.cumsum().shift(1).fillna(0)))

    df["_rr"]   = df["_cr"] / (df["_vb"] / 6.0).clip(lower=0.5)
    df["_br"]   = (120 - df["_vb"]).clip(lower=0.0)
    df["_wr"]   = (10  - df["_cw"]).clip(lower=0.0)
    df["_bov"]  = df.groupby(grp_ov).cumcount()  # ball number within over

    # ── Build feature matrix ─────────────────────────────────────────────────
    F_arr = np.column_stack([
        (df["over"].values / 19.0),
        (df["_bov"].values / 5.0),
        np.clip(df["_rr"].values / 15.0, 0, 1),
        (df["_cw"].values / 10.0),
        (df["_br"].values / 120.0),
        (df["_wr"].values / 10.0),
        (df["over"].values < 6).astype(np.float32),
        (df["over"].values >= 15).astype(np.float32),
        np.clip(df["_orns"].values / 20.0, 0, 1),
        (df["innings"].values == 2).astype(np.float32),
    ]).astype(np.float32)

    Y_arr = df["outcome_class"].values.astype(np.int64)

    # ── Group into per-innings sequences ─────────────────────────────────────
    sequences = []
    for (mid, inn), idx in df.groupby(["match_id", "innings"]).groups.items():
        idx = list(idx)
        sequences.append((F_arr[idx], Y_arr[idx]))

    print(f"  {len(sequences):,} innings sequences | "
          f"avg {np.mean([len(s[0]) for s in sequences]):.0f} balls")

    # ── Class frequency ──────────────────────────────────────────────────────
    all_y = np.concatenate([s[1] for s in sequences])
    counts = np.bincount(all_y, minlength=NUM_OUTCOME_CLASSES).astype(float)
    freqs  = counts / counts.sum()
    print("  Outcome class frequencies:")
    labels = ["dot","1r","2r","3r","4(b)","six","wkt","extra"]
    for lb, fr in zip(labels, freqs):
        print(f"    {lb:6s} {fr:.3f}")

    return sequences, freqs


# ─── 2. Dataset ───────────────────────────────────────────────────────────────

class InningsDataset(Dataset):
    def __init__(self, sequences):
        self.data = sequences

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        feats, outcomes = self.data[idx]
        feats    = torch.tensor(feats,    dtype=torch.float32)    # (T, F)
        outcomes = torch.tensor(outcomes, dtype=torch.long)       # (T,)
        # Shifted outcomes: prepend BOS, drop last (input to transformer)
        bos     = torch.full((1,), BOS_TOKEN, dtype=torch.long)
        shifted = torch.cat([bos, outcomes[:-1]])                  # (T,)
        return feats, shifted, outcomes


def collate_fn(batch):
    """Pad variable-length sequences within the batch."""
    feats_list, shifted_list, targets_list = zip(*batch)
    feats_pad   = pad_sequence(feats_list,   batch_first=True, padding_value=0.0)
    shifted_pad = pad_sequence(shifted_list, batch_first=True, padding_value=BOS_TOKEN)
    targets_pad = pad_sequence(targets_list, batch_first=True, padding_value=-100)
    # Padding mask: True where padded
    lengths = torch.tensor([f.size(0) for f in feats_list])
    max_len = feats_pad.size(1)
    mask    = torch.arange(max_len).unsqueeze(0) >= lengths.unsqueeze(1)
    return feats_pad, shifted_pad, targets_pad, mask


# ─── 3. Training & evaluation ─────────────────────────────────────────────────

def cross_entropy_loss(logits, targets):
    """Flattened CE, ignoring -100 padding."""
    B, T, C = logits.shape
    return F.cross_entropy(
        logits.reshape(B * T, C),
        targets.reshape(B * T),
        ignore_index=-100,
    )


def perplexity(loss: float) -> float:
    return float(np.exp(loss))


def run_epoch(model, loader, optimizer=None, scheduler=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0; total_tokens = 0
    with torch.set_grad_enabled(training):
        for feats, shifted, targets, mask in loader:
            feats   = feats.to(DEVICE)
            shifted = shifted.to(DEVICE)
            targets = targets.to(DEVICE)
            mask    = mask.to(DEVICE)

            logits = model(shifted, feats, padding_mask=mask)
            loss   = cross_entropy_loss(logits, targets)
            n_tok  = (targets != -100).sum().item()

            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
                if scheduler:
                    scheduler.step()

            total_loss   += loss.item() * n_tok
            total_tokens += n_tok

    return total_loss / max(total_tokens, 1)


# ─── 4. Main ──────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()

    sequences, freqs = load_sequences()

    # Time-based split: use year info → last 20% by sequence index as val
    n_val  = max(1, int(len(sequences) * 0.15))
    n_tr   = len(sequences) - n_val
    tr_seq = sequences[:n_tr]
    va_seq = sequences[n_tr:]

    tr_ds = InningsDataset(tr_seq)
    va_ds = InningsDataset(va_seq)
    tr_ld = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,
                       collate_fn=collate_fn, num_workers=0)
    va_ld = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False,
                       collate_fn=collate_fn, num_workers=0)

    model = NextBallTransformer(D_MODEL, N_HEADS, N_LAYERS, DROPOUT).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: d={D_MODEL} heads={N_HEADS} layers={N_LAYERS}  "
          f"params={n_params:,}  device={DEVICE}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    # OneCycle scheduler for fast convergence
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR, steps_per_epoch=len(tr_ld),
        epochs=EPOCHS, pct_start=0.2,
    )

    # Baseline: log-loss of always predicting class frequencies
    baseline_ce = float(-np.sum(freqs * np.log(freqs + 1e-9)))
    print(f"Frequency-baseline CE: {baseline_ce:.4f}  ppl={perplexity(baseline_ce):.2f}")

    best_val = float("inf")
    best_state = None
    history = []

    print(f"\n{'Epoch':>5} {'Train CE':>10} {'Val CE':>10} {'Val PPL':>9} {'Time':>7}")
    print("─" * 50)

    for epoch in range(1, EPOCHS + 1):
        t_ep = time.time()
        tr_loss = run_epoch(model, tr_ld, optimizer, scheduler)
        va_loss = run_epoch(model, va_ld)
        elapsed = time.time() - t_ep
        ppl     = perplexity(va_loss)
        print(f"{epoch:>5} {tr_loss:>10.4f} {va_loss:>10.4f} {ppl:>9.2f} {elapsed:>6.1f}s")
        history.append({"epoch": epoch, "train_ce": tr_loss, "val_ce": va_loss, "ppl": ppl})
        if va_loss < best_val:
            best_val   = va_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    total_time = time.time() - t0
    print(f"\nBest val CE: {best_val:.4f}  ppl={perplexity(best_val):.2f}  "
          f"(baseline ppl={perplexity(baseline_ce):.2f})")
    print(f"Total training time: {total_time/60:.1f} min")

    # ── Save model ──────────────────────────────────────────────────────────
    os.makedirs("artifacts", exist_ok=True)
    save_path = "artifacts/transformer_model.pt"
    torch.save({
        "model_state":  best_state,
        "model_config": {
            "d_model":    D_MODEL,
            "nhead":      N_HEADS,
            "num_layers": N_LAYERS,
            "dropout":    DROPOUT,
        },
        "train_history": history,
        "class_freqs":   freqs.tolist(),
        "baseline_ce":   baseline_ce,
        "best_val_ce":   best_val,
    }, save_path)
    print(f"Model saved → {save_path}  "
          f"({os.path.getsize(save_path)/1e6:.2f} MB)")

    # ── Update som_artifacts.pkl with transformer metadata ──────────────────
    pkl_path = "artifacts/som_artifacts.pkl"
    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            arts = pickle.load(f)
        arts["transformer_trained"]    = True
        arts["transformer_val_ce"]     = best_val
        arts["transformer_baseline_ce"]= baseline_ce
        arts["transformer_history"]    = history
        arts["transformer_class_freqs"]= freqs.tolist()
        with open(pkl_path, "wb") as f:
            pickle.dump(arts, f, protocol=4)
        print(f"som_artifacts.pkl updated with transformer metadata")


if __name__ == "__main__":
    main()
