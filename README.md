# Multi-Dimensional SOM Framework for IPL Match Analytics

A research framework applying unsupervised learning and deep sequence modelling to ball-by-ball IPL data. Trains Self-Organising Maps across three analytical dimensions — match momentum, player archetypes, and delivery pressure — and integrates a GPT-style causal Transformer for next-ball outcome prediction.

**Live Demo:** [ipl-cricket-som-analytics.onrender.com](https://ipl-cricket-som-analytics.onrender.com)

---

## Overview

The system is structured across five phases:

| Phase | Description |
|-------|-------------|
| 1 | Data architecture & feature engineering (ball-by-ball, over-level, innings-level) |
| 2 | **Match Momentum SOM** — game state topology, winning vs losing trajectories |
| 3 | **Player Evolution SOM** — 15-match rolling archetypes, career migration |
| 4 | **Delivery Pressure SOM** — clutch metrics, batter × bowler matchup under pressure |
| 5 | Integration — enhanced win probability model, publication-ready visualisations |

Trained on **240,000+ ball-by-ball IPL deliveries** across 950+ matches.

---

## Models

### Self-Organising Maps
- **Match Momentum SOM** (10×10): maps over-by-over game state vectors; trajectory analytics (velocity, curvature, path length, state quality) confirmed significant difference between winning and losing paths (p < 0.001, Mann-Whitney U across all 4 metrics)
- **Player Archetype SOM** (5×5, 12 features): strike rate, boundary %, phase-specific scoring rates, consistency metrics → 5 semantically validated batter clusters + 8 bowler archetypes
- **Delivery Pressure SOM**: per-delivery pressure vectors; clutch-window detection identifying 5-ball high-leverage periods (Δwᵣ ≥ 0.12)

### Win Probability Model
Gradient Boosting classifier trained on ball-by-ball match state features.
- Brier score: **0.1407** (innings 2)
- R² = **0.84**

### Next-Ball Transformer
Causal (GPT-style) Transformer encoder predicting next-ball outcome over 8 classes: dot, 1, 2, 3, boundary, six, wicket, extra.

| Parameter | Value |
|-----------|-------|
| Parameters | 151,072 |
| Input | outcome embedding + 10 numerical context features per ball |
| Output | P(next outcome ∈ {dot, 1r, 2r, 3r, 4, six, wicket, extra}) |
| Secondary output | 64-dim innings embedding (mean-pooled hidden states) |
| Positional encoding | Sinusoidal |

---

## Streamlit Dashboard

6-page interactive application:

1. **Live Match Prediction** — ball-by-ball win probability with rolling updates
2. **SOM Match Trajectory** — visualise match path on the momentum grid
3. **Player Archetype Map** — explore batter/bowler cluster assignments
4. **Venue Analytics** — phase-conditioned batting statistics by ground
5. **Clutch Window Analysis** — high-leverage period detection per match
6. **Next-Ball Predictor** — Transformer inference over 8 outcome classes

---

## Project Structure

```
├── SOM_Research_Framework.py   # Core SOM training pipeline (all 5 phases)
├── SOM_Cricket_Analysis.ipynb  # Research notebook with analysis and visualisations
├── transformer_model.py        # Causal Transformer architecture
├── train_transformer.py        # Transformer training pipeline
├── streamlit_app.py            # 6-page Streamlit dashboard
├── artifacts/
│   ├── som_artifacts.pkl       # Serialised SOM models + metadata (11 MB bundle)
│   └── transformer_model.pt    # Trained Transformer checkpoint
├── Dockerfile
├── render.yaml
└── requirements.txt
```

---

## Running Locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

The dashboard loads pre-trained artifacts from `artifacts/`. To retrain:

```bash
# Retrain Transformer (requires IPL ball-by-ball dataset)
python train_transformer.py

# SOM retraining is done via SOM_Research_Framework.py or the notebook
```

---

## Dependencies

Python 3.10+ · PyTorch · Streamlit · MiniSOM · XGBoost · Scikit-learn · Plotly · Pandas · NumPy · SciPy
