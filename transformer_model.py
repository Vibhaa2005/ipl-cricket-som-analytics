"""
Cricket Transformer — Next-Ball Outcome Predictor + Innings Encoder.

Architecture  : Causal (GPT-style) Transformer Encoder
Input per ball: [outcome_class_prev (embedding) | 10 numerical context features]
Output        : P(next_ball_outcome ∈ {dot,1,2,3,4,six,wicket,extra})
Secondary use : mean-pool hidden states → 64-dim innings embedding
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Vocabulary ─────────────────────────────────────────────────────────────────
NUM_OUTCOME_CLASSES = 8   # 0=dot 1=1r 2=2r 3=3r 4=4(bdry) 5=6 6=wicket 7=extra
BOS_TOKEN           = 8   # beginning-of-sequence token (never predicted, only input)
VOCAB_SIZE          = 9   # 0-7 real outcomes + BOS

# ── Feature layout (10 floats per ball) ───────────────────────────────────────
#  0  over_norm            over / 19
#  1  ball_in_over_norm    ball position in over / 5
#  2  cum_rr_norm          cumulative run-rate / 15  (clipped 0-1)
#  3  cum_wickets_norm     wickets fallen / 10
#  4  balls_remaining_norm balls left / 120
#  5  wickets_remaining    (10 - wkts) / 10
#  6  is_powerplay         1 if over < 6
#  7  is_death             1 if over >= 15
#  8  runs_this_over_norm  runs scored in current over / 20 (clipped 0-1)
#  9  innings2             1 if innings == 2
NUM_FEATURES = 10


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 160, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10_000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)])


class NextBallTransformer(nn.Module):
    """
    Causal transformer for next-ball outcome prediction.

    At each sequence position t the model sees:
      - Embedding of the outcome that happened at position t-1 (BOS at t=0)
      - 10 numerical context features describing the match state at position t

    It predicts the outcome class at position t (teacher-forcing during training).

    Secondary capability: encode_innings() → 64-dim innings embedding.
    """

    def __init__(
        self,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
        max_seq: int = 160,
    ):
        super().__init__()
        assert d_model % 2 == 0, "d_model must be even (split between embed and feat)"

        self.d_model = d_model
        half = d_model // 2

        self.outcome_embed = nn.Embedding(VOCAB_SIZE, half)
        self.feat_proj = nn.Sequential(
            nn.Linear(NUM_FEATURES, half),
            nn.LayerNorm(half),
        )
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_seq, dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,      # pre-LN — more stable
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers,
                                                  enable_nested_tensor=False)
        self.head = nn.Linear(d_model, NUM_OUTCOME_CLASSES)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.outcome_embed.weight, std=0.02)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @staticmethod
    def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        # bool upper-triangular mask (True = ignore) — same type as padding_mask
        return torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
            diagonal=1,
        )

    def forward(
        self,
        shifted_outcomes: torch.Tensor,   # (B, T) int  — outcomes shifted right (BOS at 0)
        features: torch.Tensor,           # (B, T, NUM_FEATURES) float
        padding_mask: torch.Tensor | None = None,  # (B, T) bool  True=ignore
    ) -> torch.Tensor:
        """Returns logits (B, T, NUM_OUTCOME_CLASSES)."""
        T = shifted_outcomes.size(1)
        emb_o = self.outcome_embed(shifted_outcomes)   # (B, T, d/2)
        emb_f = self.feat_proj(features)               # (B, T, d/2)
        x = torch.cat([emb_o, emb_f], dim=-1)         # (B, T, d)
        x = self.pos_enc(x)
        causal = self._causal_mask(T, x.device)
        x = self.transformer(
            x,
            mask=causal,
            src_key_padding_mask=padding_mask,
            is_causal=True,
        )
        return self.head(x)                            # (B, T, C)

    def encode_innings(
        self,
        shifted_outcomes: torch.Tensor,
        features: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Mean-pool hidden states → (B, d_model) innings embedding."""
        T = shifted_outcomes.size(1)
        emb_o = self.outcome_embed(shifted_outcomes)
        emb_f = self.feat_proj(features)
        x = torch.cat([emb_o, emb_f], dim=-1)
        x = self.pos_enc(x)
        causal = self._causal_mask(T, x.device)
        h = self.transformer(
            x,
            mask=causal,
            src_key_padding_mask=padding_mask,
            is_causal=True,
        )
        if padding_mask is not None:
            active = (~padding_mask).float().unsqueeze(-1)  # (B, T, 1)
            return (h * active).sum(1) / active.sum(1).clamp(min=1.0)
        return h.mean(dim=1)  # (B, d_model)

    @torch.no_grad()
    def predict_next(
        self,
        features: torch.Tensor,           # (1, T, NUM_FEATURES)
        past_outcomes: torch.Tensor,      # (1, T) int — actual outcomes so far
    ) -> torch.Tensor:
        """Returns next-ball probability vector (NUM_OUTCOME_CLASSES,)."""
        self.eval()
        B, T = past_outcomes.shape
        # Shift right: prepend BOS, drop last
        bos = torch.full((B, 1), BOS_TOKEN, dtype=torch.long, device=past_outcomes.device)
        shifted = torch.cat([bos, past_outcomes[:, :-1]], dim=1)
        logits = self.forward(shifted, features)     # (B, T, C)
        return F.softmax(logits[0, -1], dim=-1)      # (C,) — next ball probs


# ── Convenience: build a single-step context from game state ─────────────────

def state_to_features(
    over: float,
    cum_runs: float,
    cum_wickets: float,
    innings: int = 1,
    ball_in_over: int = 0,
    runs_this_over: float = 0.0,
) -> torch.Tensor:
    """
    Build a (1, 1, NUM_FEATURES) feature tensor from simple game-state numbers.
    Enough to call predict_next() with a single-position context.
    """
    balls_done       = over * 6 + ball_in_over
    balls_remaining  = max(0.0, 120.0 - balls_done)
    cum_rr           = cum_runs / max(balls_done / 6.0, 0.5)
    wickets_remaining = max(0.0, 10.0 - cum_wickets)

    feats = [
        over / 19.0,
        ball_in_over / 5.0,
        min(cum_rr / 15.0, 1.0),
        cum_wickets / 10.0,
        balls_remaining / 120.0,
        wickets_remaining / 10.0,
        float(over < 6),
        float(over >= 15),
        min(runs_this_over / 20.0, 1.0),
        float(innings == 2),
    ]
    return torch.tensor([[feats]], dtype=torch.float32)   # (1, 1, 10)


OUTCOME_LABELS = ["Dot", "1 Run", "2 Runs", "3 Runs", "4 (Bdry)", "Six", "Wicket", "Extra"]
OUTCOME_COLORS = ["#95a5a6", "#3498db", "#2ecc71", "#f39c12",
                  "#e67e22", "#9b59b6", "#e74c3c", "#bdc3c7"]
