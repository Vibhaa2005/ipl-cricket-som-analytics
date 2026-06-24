# =============================================================================
#  MULTI-OBJECTIVE UNSUPERVISED LEARNING FRAMEWORK
#  FOR DYNAMIC MATCH STATE & PLAYER EVOLUTION IN CRICKET
# =============================================================================
#
#  PROJECT ROADMAP (5 Phases)
#  ─────────────────────────────────────────────────────────────
#  PHASE 1 — Data Architecture & Unification      [THIS FILE: Cells 1-4]
#    • Load IPL hackathon dataset (ball-by-ball)
#    • Engineer over-level, innings-level, delivery-level features
#    • Classify bowling types (Pace/Spin) from dismissal heuristics
#    • Build player rolling-window profiles
#    • LATER: Merge Champions Trophy 2025 & ICC datasets
#
#  PHASE 2 — Dimension I: Match Momentum SOM       [Cells 5-9]
#    • Over-by-over game state vectors
#    • SOM training → Game State topological map
#    • Trajectory plotting (single match as path on grid)
#    • Batch trajectory analysis (winning vs losing paths)
#
#  PHASE 3 — Dimension II: Player Evolution SOM    [Cells 10-14]
#    • 15-match rolling window profiles for batters & bowlers
#    • SOM training → Player Archetype map
#    • Career migration visualization (Kohli, Dhoni, etc.)
#    • Archetype transition probability matrix
#
#  PHASE 4 — Dimension III: Delivery Pressure SOM  [Cells 15-19]
#    • Per-delivery pressure vector engineering
#    • SOM training → Pressure Zone map
#    • Clutch Player metric (performance in high-pressure nodes)
#    • Batter × Bowler matchup under pressure
#
#  PHASE 5 — Integration & Applications            [Cells 20-22]
#    • Merge SOM-derived features into hackathon ML pipeline
#    • Enhanced Match Winner model with momentum + clutch features
#    • External dataset fusion protocol (Champions Trophy, ICC)
#    • Publication-ready visualizations
#  ─────────────────────────────────────────────────────────────
#
#  Colab setup (run once):
#    !pip install minisom xgboost
# =============================================================================


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  PHASE 1 — DATA ARCHITECTURE & FEATURE ENGINEERING                        ║
# ╚════════════════════════════════════════════════════════════════════════════╝

# === CELL 1: IMPORTS & CONFIG ================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             classification_report, confusion_matrix)
from sklearn.model_selection import train_test_split

try:
    from minisom import MiniSom
    print("MiniSom loaded")
except ImportError:
    raise ImportError("Run: !pip install minisom")

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
    print("XGBoost loaded")
except ImportError:
    HAS_XGB = False
    print("XGBoost not found — run: !pip install xgboost")

plt.rcParams['figure.figsize'] = (14, 7)
plt.rcParams['font.size'] = 11
sns.set_style('whitegrid')
print("Phase 1 ready")


# === CELL 2: LOAD IPL DATASET & BASE ENGINEERING ============================
df = pd.read_csv('IPL_with_match_types.csv', low_memory=False)
print(f"Raw dataset: {df.shape[0]:,} balls across {df['match_id'].nunique()} matches")

# Match stage classification (from hackathon)
def classify_stage(s):
    s = str(s).strip()
    if s == 'Final': return 'Final'
    if s in ['Semi Final','Qualifier 1','Qualifier 2','Eliminator',
             'Elimination Final','3rd Place Play-Off']: return 'Playoff'
    return 'League'

df['match_stage'] = df['stage'].apply(classify_stage)

# Bowling type (from hackathon — stumping heuristic + manual list)
spin_from_stumpings = set(df[df['wicket_kind']=='stumped']['bowler'].unique())
manual_spin = {
    'SP Narine','R Ashwin','Rashid Khan','YS Chahal','Harbhajan Singh',
    'A Mishra','PP Chawla','RA Jadeja','Kuldeep Yadav','PP Ojha',
    'AR Patel','SK Warne','M Muralitharan','Imran Tahir','SB Jakati',
    'KH Pandya','R Tewatia','KV Sharma','Wanindu Hasaranga','R Bishnoi',
    'V Chakravarthy','Noor Ahmad','K Gowtham','Shahbaz Ahmed','A Zampa',
    'S Gopal','M Ashwin','K Yadav','Axar Patel','RD Chahar',
}
all_spinners = spin_from_stumpings | manual_spin
df['bowl_type'] = df['bowler'].apply(lambda x: 'Spin' if x in all_spinners else 'Pace')

# Helper flags
df['is_four'] = (~df['runs_not_boundary']) & (df['runs_batter'] == 4)
df['is_six']  = (~df['runs_not_boundary']) & (df['runs_batter'] == 6)
df['is_dot']  = (df['runs_total'] == 0) & (df['valid_ball'] == 1)
df['is_boundary'] = df['is_four'] | df['is_six']
df['is_wicket'] = df['wicket_kind'].notna().astype(int)

# Phase of innings
df['phase'] = pd.cut(df['over'], bins=[-1, 5, 14, 20],
                     labels=['Powerplay','Middle','Death'])

print(f"Engineering complete. Bowl types: Spin={df[df['bowl_type']=='Spin']['bowler'].nunique()}, "
      f"Pace={df[df['bowl_type']=='Pace']['bowler'].nunique()}")


# === CELL 3: OVER-LEVEL GAME STATE VECTORS (for Dimension I) ================
# Each over in each innings becomes a high-dimensional state vector

over_raw = df.groupby(['match_id','innings','over','batting_team','bowling_team',
                        'match_stage','venue','year']).agg(
    runs_this_over     = ('runs_total', 'sum'),
    wickets_this_over  = ('is_wicket', 'sum'),
    dots_this_over     = ('is_dot', 'sum'),
    boundaries         = ('is_boundary', 'sum'),
    sixes_this_over    = ('is_six', 'sum'),
    valid_balls        = ('valid_ball', 'sum'),
    extras             = ('runs_extras', 'sum'),
    # Cumulative state at END of this over
    cum_runs           = ('team_runs', 'max'),
    cum_wickets        = ('team_wicket', 'max'),
    cum_balls          = ('team_balls', 'max'),
    # Target info (2nd innings only)
    target             = ('runs_target', 'first'),
).reset_index()

# Derived dynamic features
over_raw['cum_run_rate'] = over_raw['cum_runs'] / ((over_raw['over'] + 1)).clip(lower=0.1)
over_raw['balls_remaining'] = 120 - over_raw['cum_balls']
over_raw['wickets_remaining'] = 10 - over_raw['cum_wickets']

# Required run rate (2nd innings)
over_raw['runs_required'] = over_raw['target'] - over_raw['cum_runs']
over_raw['req_run_rate'] = np.where(
    (over_raw['innings'] == 2) & (over_raw['balls_remaining'] > 0),
    over_raw['runs_required'] / (over_raw['balls_remaining'] / 6),
    np.nan
)

# Pressure index = req_rr / cum_rr (how far behind the asking rate?)
over_raw['pressure_ratio'] = np.where(
    over_raw['req_run_rate'].notna() & (over_raw['cum_run_rate'] > 0),
    over_raw['req_run_rate'] / over_raw['cum_run_rate'],
    np.nan
)

# Momentum: runs in this over vs average run rate so far
over_raw['over_run_rate'] = over_raw['runs_this_over'] / (over_raw['valid_balls'] / 6).clip(lower=0.1)
over_raw['momentum'] = over_raw['over_run_rate'] - over_raw['cum_run_rate']

# Match winner (for trajectory coloring later)
match_winners = df.drop_duplicates('match_id')[['match_id','match_won_by']]
over_raw = over_raw.merge(match_winners, on='match_id')
over_raw['batting_team_won'] = (over_raw['batting_team'] == over_raw['match_won_by']).astype(int)

print(f"Over-level game states: {len(over_raw):,} rows")
print(f"Columns: {list(over_raw.columns)}")
print(over_raw.head())


# === CELL 4: PLAYER ROLLING-WINDOW PROFILES (for Dimension II) ==============
# Build 15-match rolling windows for each batter

# First, get per-innings batter stats
batter_inn = df[df['valid_ball']==1].groupby(
    ['match_id','innings','batter','batting_team','year','match_stage']
).agg(
    runs         = ('runs_batter', 'sum'),
    balls        = ('valid_ball', 'sum'),
    fours        = ('is_four', 'sum'),
    sixes        = ('is_six', 'sum'),
    dots         = ('is_dot', 'sum'),
).reset_index()

# Dismissal flag
dismissals = df[df['wicket_kind'].notna()][['match_id','innings','player_out','wicket_kind']]
dismissals = dismissals.rename(columns={'player_out': 'batter'})
dismissals['dismissed'] = 1
batter_inn = batter_inn.merge(dismissals[['match_id','innings','batter','dismissed','wicket_kind']],
                               on=['match_id','innings','batter'], how='left')
batter_inn['dismissed'] = batter_inn['dismissed'].fillna(0)

batter_inn['strike_rate'] = (batter_inn['runs'] / batter_inn['balls'].clip(lower=1) * 100)
batter_inn['boundary_pct'] = ((batter_inn['fours']*4 + batter_inn['sixes']*6) /
                               batter_inn['runs'].clip(lower=1) * 100)
batter_inn['dot_pct'] = (batter_inn['dots'] / batter_inn['balls'].clip(lower=1) * 100)

# Sort chronologically per batter
batter_inn = batter_inn.sort_values(['batter','match_id','innings']).reset_index(drop=True)

# Build rolling 15-match windows
WINDOW = 15
MIN_CAREER = 50  # only players with 50+ innings

qualified_batters = batter_inn.groupby('batter').size()
qualified_batters = qualified_batters[qualified_batters >= MIN_CAREER].index.tolist()

windows = []
for player in qualified_batters:
    p_data = batter_inn[batter_inn['batter'] == player].reset_index(drop=True)
    for start in range(0, len(p_data) - WINDOW + 1, 5):  # step=5 for overlap
        window = p_data.iloc[start:start + WINDOW]
        total_runs = window['runs'].sum()
        total_balls = window['balls'].sum()
        total_dismissed = window['dismissed'].sum()
        windows.append({
            'batter': player,
            'window_start_match': window['match_id'].iloc[0],
            'window_end_match': window['match_id'].iloc[-1],
            'window_start_year': window['year'].iloc[0],
            'window_end_year': window['year'].iloc[-1],
            'window_idx': start // 5,
            # --- Profile vector ---
            'avg': total_runs / max(total_dismissed, 1),
            'sr': total_runs / max(total_balls, 1) * 100,
            'boundary_pct': window['boundary_pct'].mean(),
            'dot_pct': window['dot_pct'].mean(),
            'sixes_per_inn': window['sixes'].mean(),
            'fours_per_inn': window['fours'].mean(),
            'runs_per_inn': window['runs'].mean(),
            'consistency': window['runs'].std(),
            'dismissal_rate': total_dismissed / WINDOW,
            'avg_balls_faced': window['balls'].mean(),
            # Context
            'pct_playoff': (window['match_stage'].isin(['Playoff','Final'])).mean(),
        })

windows_df = pd.DataFrame(windows)
print(f"\nPlayer rolling windows: {len(windows_df):,} windows for {windows_df['batter'].nunique()} players")
print(f"Profile features: avg, sr, boundary_pct, dot_pct, sixes_per_inn, fours_per_inn, "
      f"runs_per_inn, consistency, dismissal_rate, avg_balls_faced, pct_playoff")
print(windows_df.head())


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  PHASE 2 — DIMENSION I: MATCH MOMENTUM & GAME STATE TRAJECTORY           ║
# ║  SOM maps the evolving state of an innings into topological clusters      ║
# ╚════════════════════════════════════════════════════════════════════════════╝

# === CELL 5: DIMENSION I — GAME STATE SOM TRAINING ===========================
#
# WHAT THIS DOES:
#   Every over in every innings is a data point in R^d space.
#   The SOM compresses this into a 2D grid where nearby nodes represent
#   similar game situations. We get clusters like:
#     - "Stable Accumulation" (low wickets, steady RR)
#     - "Aggressive Death Hitting" (high RR, many boundaries, late overs)
#     - "Top-Order Collapse" (multiple wickets, low RR, early overs)
#     - "Chase Pressure" (high req RR, rising pressure ratio)
#
# WHY SOM OVER KMEANS:
#   SOMs preserve topology — neighboring nodes on the grid represent
#   similar game states. This lets us plot a match as a TRAJECTORY
#   (a path across the grid) which is impossible with flat clustering.

# Feature vector for each over
STATE_FEATURES = [
    'cum_run_rate',        # how fast the team is scoring overall
    'over_run_rate',       # how fast THIS over was
    'momentum',            # over_rr - cum_rr (acceleration)
    'wickets_this_over',   # collapse signal
    'wickets_remaining',   # resources left
    'dots_this_over',      # pressure from dot balls
    'boundaries',          # aggression signal
    'cum_wickets',         # total fall of wickets
    'over',                # phase proxy (early/middle/death)
]

# For 2nd innings, add chase-specific features
STATE_FEATURES_CHASE = STATE_FEATURES + ['req_run_rate', 'pressure_ratio']

# --- Train SOM on 1st innings (bat-first context) ---
inn1 = over_raw[over_raw['innings'] == 1].copy()
X_state1 = inn1[STATE_FEATURES].fillna(0).values
scaler_state = StandardScaler()
X_state1_scaled = scaler_state.fit_transform(X_state1)

SOM_X, SOM_Y = 6, 6  # 36 nodes = game state archetypes
som_state = MiniSom(SOM_X, SOM_Y, X_state1_scaled.shape[1],
                    sigma=1.5, learning_rate=0.5, random_seed=42,
                    neighborhood_function='gaussian')
som_state.random_weights_init(X_state1_scaled)
som_state.train_random(X_state1_scaled, 5000)

# Assign each over to its BMU (Best Matching Unit)
inn1['som_x'] = [som_state.winner(x)[0] for x in X_state1_scaled]
inn1['som_y'] = [som_state.winner(x)[1] for x in X_state1_scaled]
inn1['som_node'] = inn1['som_x'] * SOM_Y + inn1['som_y']

print(f"Game State SOM trained: {SOM_X}x{SOM_Y} grid = {SOM_X*SOM_Y} nodes")
print(f"Quantization error: {som_state.quantization_error(X_state1_scaled):.4f}")


# === CELL 6: DIMENSION I — LABEL GAME STATE NODES ===========================
# Analyze what each SOM node represents by looking at average feature values

node_profiles = inn1.groupby('som_node')[STATE_FEATURES + ['batting_team_won']].mean()

# Auto-label based on dominant characteristics
def label_node(row):
    parts = []
    if row['cum_wickets'] > 5:
        parts.append('Collapse')
    elif row['wickets_this_over'] > 0.5:
        parts.append('Wicket-Fall')

    if row['over'] < 5:
        parts.append('Early')
    elif row['over'] < 14:
        parts.append('Mid')
    else:
        parts.append('Death')

    if row['momentum'] > 2:
        parts.append('Surge')
    elif row['momentum'] < -2:
        parts.append('Slowdown')
    else:
        parts.append('Steady')

    if row['boundaries'] > 2:
        parts.append('Aggressive')

    return ' | '.join(parts)

node_profiles['label'] = node_profiles.apply(label_node, axis=1)
node_labels = node_profiles['label'].to_dict()

print("\nGame State Node Profiles (sample):")
for node_id in sorted(node_profiles.index)[:12]:
    row = node_profiles.loc[node_id]
    print(f"  Node {node_id:2d}: [{row['label']:<40s}] "
          f"RR={row['cum_run_rate']:.1f} Wkts={row['cum_wickets']:.1f} "
          f"Over≈{row['over']:.0f} WinRate={row['batting_team_won']:.2f}")


# === CELL 7: DIMENSION I — SOM HEATMAPS =====================================

fig, axes = plt.subplots(2, 3, figsize=(20, 13))

heatmap_features = {
    'cum_run_rate': ('Run Rate', 'YlOrRd'),
    'cum_wickets': ('Wickets Fallen', 'Reds'),
    'momentum': ('Momentum (Over RR − Cum RR)', 'RdYlGn'),
    'boundaries': ('Boundaries per Over', 'YlGn'),
    'over': ('Average Over Number', 'Blues'),
    'batting_team_won': ('Win Rate from State', 'RdYlGn'),
}

for ax, (feat, (title, cmap)) in zip(axes.flatten(), heatmap_features.items()):
    grid = np.zeros((SOM_X, SOM_Y))
    counts = np.zeros((SOM_X, SOM_Y))
    for _, row in inn1.iterrows():
        grid[int(row['som_x']), int(row['som_y'])] += row[feat] if pd.notna(row.get(feat)) else 0
        counts[int(row['som_x']), int(row['som_y'])] += 1
    counts[counts == 0] = 1
    grid /= counts
    im = ax.imshow(grid, cmap=cmap, interpolation='nearest')
    ax.set_title(title, fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.set_xlabel('SOM X')
    ax.set_ylabel('SOM Y')

plt.suptitle('DIMENSION I: Game State Topological Map', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()


# === CELL 8: DIMENSION I — MATCH TRAJECTORY VISUALIZATION ====================
# Plot a single match as a path traversing the SOM grid
#
# THIS IS THE KEY INNOVATION: instead of saying "Team A has 60% win probability",
# we show WHERE the match IS on the game state map, and WHERE it's HEADING.

def plot_match_trajectory(match_id, innings=1, ax=None):
    """Plot one innings as a trajectory across the Game State SOM."""
    match_data = inn1[(inn1['match_id'] == match_id) & (inn1['innings'] == innings)]
    match_data = match_data.sort_values('over')

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    # Draw SOM grid background (shaded by win rate)
    grid_wr = np.zeros((SOM_X, SOM_Y))
    counts = np.zeros((SOM_X, SOM_Y))
    for _, row in inn1.iterrows():
        grid_wr[int(row['som_x']), int(row['som_y'])] += row['batting_team_won']
        counts[int(row['som_x']), int(row['som_y'])] += 1
    counts[counts == 0] = 1
    grid_wr /= counts
    ax.imshow(grid_wr, cmap='RdYlGn', alpha=0.3, interpolation='nearest',
              extent=[-0.5, SOM_Y-0.5, SOM_X-0.5, -0.5])

    # Plot trajectory as colored line (color = over number)
    xs = match_data['som_y'].values + np.random.normal(0, 0.08, len(match_data))
    ys = match_data['som_x'].values + np.random.normal(0, 0.08, len(match_data))
    overs = match_data['over'].values

    points = np.array([xs, ys]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = Normalize(vmin=0, vmax=19)
    lc = LineCollection(segments, cmap='plasma', norm=norm, linewidth=2.5, alpha=0.9)
    lc.set_array(overs[:-1])
    ax.add_collection(lc)

    # Mark start and end
    ax.scatter(xs[0], ys[0], s=200, c='green', marker='o', zorder=5,
              edgecolors='black', linewidths=2, label='Start (Over 0)')
    ax.scatter(xs[-1], ys[-1], s=200, c='red', marker='*', zorder=5,
              edgecolors='black', linewidths=2, label=f'End (Over {overs[-1]})')

    team = match_data['batting_team'].iloc[0]
    won = match_data['batting_team_won'].iloc[0]
    ax.set_title(f'{team} — {"WON" if won else "LOST"}', fontsize=12, fontweight='bold')
    ax.set_xlim(-0.5, SOM_Y-0.5)
    ax.set_ylim(SOM_X-0.5, -0.5)
    ax.legend(loc='lower right', fontsize=8)
    ax.set_xlabel('SOM X')
    ax.set_ylabel('SOM Y')
    return ax


# Pick 4 interesting matches (high-scoring, close finishes)
sample_matches = inn1.drop_duplicates('match_id').nlargest(4, 'cum_runs')['match_id'].values

fig, axes = plt.subplots(2, 2, figsize=(16, 16))
for ax, mid in zip(axes.flatten(), sample_matches):
    plot_match_trajectory(mid, innings=1, ax=ax)

plt.suptitle('DIMENSION I: Match Trajectories on Game State SOM\n'
             '(Background: green=high win rate, red=low | Line color: over number)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()


# === CELL 9: DIMENSION I — WINNING vs LOSING TRAJECTORY HEATMAPS ============
# Where do winning and losing innings "spend time" on the SOM?

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

for ax, (label, won_val) in zip(axes[:2], [('WINNING Innings', 1), ('LOSING Innings', 0)]):
    subset = inn1[inn1['batting_team_won'] == won_val]
    grid = np.zeros((SOM_X, SOM_Y))
    for _, row in subset.iterrows():
        grid[int(row['som_x']), int(row['som_y'])] += 1
    grid = grid / grid.sum()  # normalize to density
    im = ax.imshow(grid, cmap='hot_r', interpolation='nearest')
    ax.set_title(f'Density: {label}', fontsize=13, fontweight='bold')
    plt.colorbar(im, ax=ax)

# Difference map
win_grid = np.zeros((SOM_X, SOM_Y))
lose_grid = np.zeros((SOM_X, SOM_Y))
for _, row in inn1[inn1['batting_team_won']==1].iterrows():
    win_grid[int(row['som_x']), int(row['som_y'])] += 1
for _, row in inn1[inn1['batting_team_won']==0].iterrows():
    lose_grid[int(row['som_x']), int(row['som_y'])] += 1
win_grid /= max(win_grid.sum(), 1)
lose_grid /= max(lose_grid.sum(), 1)
diff = win_grid - lose_grid
im = axes[2].imshow(diff, cmap='RdYlGn', interpolation='nearest')
axes[2].set_title('Difference (Win − Loss Density)', fontsize=13, fontweight='bold')
plt.colorbar(im, ax=axes[2])

plt.suptitle('Where Winning vs Losing Innings Live on the Game State Map',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  PHASE 3 — DIMENSION II: PLAYER ARCHETYPE CAREER EVOLUTION               ║
# ║  SOM maps rolling performance windows into player archetypes              ║
# ║  A career = a migration path across the archetype grid                    ║
# ╚════════════════════════════════════════════════════════════════════════════╝

# === CELL 10: DIMENSION II — PLAYER ARCHETYPE SOM TRAINING ===================
#
# WHAT THIS DOES:
#   Each 15-match window of a player is a vector of [avg, sr, boundary%, dot%,
#   6s/inn, 4s/inn, runs/inn, consistency, dismissal_rate, balls_faced].
#   The SOM clusters these into archetypes like:
#     - "Anchor" (high avg, low SR, low boundaries)
#     - "Power Finisher" (high SR, high sixes, death overs)
#     - "Aggressive Opener" (high SR, high fours, early overs)
#     - "Struggling / Out of Form" (low avg, high dot%, high dismissal rate)

PROFILE_FEATURES = [
    'avg', 'sr', 'boundary_pct', 'dot_pct',
    'sixes_per_inn', 'fours_per_inn', 'runs_per_inn',
    'consistency', 'dismissal_rate', 'avg_balls_faced',
]

X_player = windows_df[PROFILE_FEATURES].fillna(0).values
scaler_player = StandardScaler()
X_player_scaled = scaler_player.fit_transform(X_player)

PSOM_X, PSOM_Y = 5, 5  # 25 archetype nodes
som_player = MiniSom(PSOM_X, PSOM_Y, X_player_scaled.shape[1],
                     sigma=1.5, learning_rate=0.5, random_seed=42,
                     neighborhood_function='gaussian')
som_player.random_weights_init(X_player_scaled)
som_player.train_random(X_player_scaled, 5000)

windows_df['som_x'] = [som_player.winner(x)[0] for x in X_player_scaled]
windows_df['som_y'] = [som_player.winner(x)[1] for x in X_player_scaled]
windows_df['archetype_node'] = windows_df['som_x'] * PSOM_Y + windows_df['som_y']

print(f"Player Archetype SOM trained: {PSOM_X}x{PSOM_Y} = {PSOM_X*PSOM_Y} archetypes")
print(f"Quantization error: {som_player.quantization_error(X_player_scaled):.4f}")


# === CELL 11: DIMENSION II — LABEL ARCHETYPE NODES ==========================

arch_profiles = windows_df.groupby('archetype_node')[PROFILE_FEATURES].mean()

def label_archetype(row):
    parts = []
    if row['avg'] > 35 and row['sr'] > 140:
        return 'Elite Aggressor'
    elif row['avg'] > 35 and row['sr'] <= 140:
        return 'Anchor / Accumulator'
    elif row['sr'] > 150 and row['sixes_per_inn'] > 1.5:
        return 'Power Finisher'
    elif row['sr'] > 140:
        return 'Aggressive Striker'
    elif row['avg'] < 15 and row['dismissal_rate'] > 0.8:
        return 'Struggling / Out of Form'
    elif row['dot_pct'] > 45:
        return 'Defensive / Under Pressure'
    elif row['runs_per_inn'] > 30:
        return 'Reliable Run-Scorer'
    else:
        return 'Role Player'

arch_profiles['label'] = arch_profiles.apply(label_archetype, axis=1)
archetype_labels = arch_profiles['label'].to_dict()
windows_df['archetype'] = windows_df['archetype_node'].map(archetype_labels)

print("\nPlayer Archetype Distribution:")
print(windows_df['archetype'].value_counts())
print("\nArchetype Profiles:")
for node in sorted(arch_profiles.index)[:10]:
    r = arch_profiles.loc[node]
    print(f"  Node {node:2d} [{r['label']:<25s}]: Avg={r['avg']:.1f} SR={r['sr']:.1f} "
          f"Bound%={r['boundary_pct']:.1f} Dot%={r['dot_pct']:.1f}")


# === CELL 12: DIMENSION II — ARCHETYPE SOM HEATMAPS =========================

fig, axes = plt.subplots(2, 3, figsize=(20, 13))

arch_heatmaps = {
    'avg': ('Batting Average', 'YlOrRd'),
    'sr': ('Strike Rate', 'YlGn'),
    'boundary_pct': ('Boundary %', 'Oranges'),
    'dot_pct': ('Dot Ball %', 'Blues'),
    'sixes_per_inn': ('Sixes per Innings', 'Purples'),
    'dismissal_rate': ('Dismissal Rate', 'Reds'),
}

for ax, (feat, (title, cmap)) in zip(axes.flatten(), arch_heatmaps.items()):
    grid = np.zeros((PSOM_X, PSOM_Y))
    counts = np.zeros((PSOM_X, PSOM_Y))
    for _, row in windows_df.iterrows():
        grid[int(row['som_x']), int(row['som_y'])] += row[feat]
        counts[int(row['som_x']), int(row['som_y'])] += 1
    counts[counts == 0] = 1
    grid /= counts
    im = ax.imshow(grid, cmap=cmap, interpolation='nearest')
    ax.set_title(title, fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax, fraction=0.046)

plt.suptitle('DIMENSION II: Player Archetype Topological Map', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()


# === CELL 13: DIMENSION II — CAREER MIGRATION (SPECIFIC PLAYERS) ============
#
# THE KEY VISUALIZATION: plot a player's career as a path across
# the archetype SOM. You can literally SEE when Kohli went from
# "Aggressive Striker" to "Elite Aggressor" to "Anchor" etc.

def plot_career_migration(player_name, ax=None):
    """Visualize one player's archetype evolution on the SOM."""
    p_data = windows_df[windows_df['batter'] == player_name].sort_values('window_idx')
    if len(p_data) == 0:
        print(f"No data for {player_name}")
        return

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    # Background: archetype density
    grid = np.zeros((PSOM_X, PSOM_Y))
    for _, row in windows_df.iterrows():
        grid[int(row['som_x']), int(row['som_y'])] += 1
    ax.imshow(grid, cmap='Greys', alpha=0.2, interpolation='nearest',
              extent=[-0.5, PSOM_Y-0.5, PSOM_X-0.5, -0.5])

    # Add archetype labels to grid
    for node_id, label in archetype_labels.items():
        nx, ny = node_id // PSOM_Y, node_id % PSOM_Y
        ax.text(ny, nx, label.split('/')[0][:12], fontsize=6, ha='center', va='center',
                alpha=0.4, color='gray')

    # Trajectory
    xs = p_data['som_y'].values + np.random.normal(0, 0.05, len(p_data))
    ys = p_data['som_x'].values + np.random.normal(0, 0.05, len(p_data))
    years = p_data['window_end_year'].values

    points = np.array([xs, ys]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = Normalize(vmin=years.min(), vmax=years.max())
    lc = LineCollection(segments, cmap='coolwarm', norm=norm, linewidth=2.5, alpha=0.8)
    lc.set_array(years[:-1])
    ax.add_collection(lc)

    ax.scatter(xs[0], ys[0], s=200, c='blue', marker='o', zorder=5,
              edgecolors='black', linewidths=2, label=f'Start ({years[0]})')
    ax.scatter(xs[-1], ys[-1], s=200, c='red', marker='*', zorder=5,
              edgecolors='black', linewidths=2, label=f'End ({years[-1]})')

    ax.set_title(f'{player_name} — Career Archetype Migration', fontsize=12, fontweight='bold')
    ax.set_xlim(-0.5, PSOM_Y-0.5)
    ax.set_ylim(PSOM_X-0.5, -0.5)
    ax.legend(loc='lower right', fontsize=8)

    # Print the archetype journey
    journey = p_data[['window_start_year','window_end_year','archetype','avg','sr']].values
    print(f"\n  {player_name}'s Archetype Journey:")
    prev_arch = None
    for ws, we, arch, avg, sr in journey:
        if arch != prev_arch:
            print(f"    {int(ws)}-{int(we)}: {arch} (Avg={avg:.1f}, SR={sr:.1f})")
            prev_arch = arch


# Plot 6 legendary players
LEGENDS = ['V Kohli', 'RG Sharma', 'MS Dhoni', 'AB de Villiers', 'DA Warner', 'SK Raina']

fig, axes = plt.subplots(2, 3, figsize=(20, 14))
for ax, player in zip(axes.flatten(), LEGENDS):
    plot_career_migration(player, ax=ax)

plt.suptitle('DIMENSION II: Career Archetype Evolution\n'
             '(Blue=early career → Red=late career)',
             fontsize=15, fontweight='bold')
plt.tight_layout()
plt.show()


# === CELL 14: DIMENSION II — ARCHETYPE TRANSITION MATRIX ====================
# What archetype do players typically evolve into next?

transitions = {}
for player in qualified_batters:
    p_data = windows_df[windows_df['batter'] == player].sort_values('window_idx')
    archetypes = p_data['archetype'].values
    for i in range(len(archetypes) - 1):
        key = (archetypes[i], archetypes[i+1])
        transitions[key] = transitions.get(key, 0) + 1

# Build transition matrix
all_archetypes = sorted(windows_df['archetype'].unique())
trans_matrix = pd.DataFrame(0, index=all_archetypes, columns=all_archetypes)
for (from_a, to_a), count in transitions.items():
    trans_matrix.loc[from_a, to_a] = count

# Normalize rows to probabilities
trans_prob = trans_matrix.div(trans_matrix.sum(axis=1), axis=0).fillna(0)

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(trans_prob, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax,
            linewidths=0.5, cbar_kws={'label': 'Transition Probability'})
ax.set_title('Archetype Transition Probability Matrix\n'
             '(Row = Current Archetype → Column = Next Archetype)',
             fontsize=14, fontweight='bold')
ax.set_xlabel('Next Archetype')
ax.set_ylabel('Current Archetype')
plt.tight_layout()
plt.show()

print("\nKey Transitions:")
for idx in trans_prob.index:
    top = trans_prob.loc[idx].nlargest(2)
    print(f"  {idx} → {top.index[0]} ({top.values[0]:.0%}), {top.index[1]} ({top.values[1]:.0%})")


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  PHASE 4 — DIMENSION III: DELIVERY PRESSURE INDEXING                      ║
# ║  SOM maps individual deliveries into "Pressure Zones"                     ║
# ║  Cross-referencing with performance → objective Clutch Metric             ║
# ╚════════════════════════════════════════════════════════════════════════════╝

# === CELL 15: DIMENSION III — DELIVERY PRESSURE VECTORS ======================

# Build per-delivery context vector
delivery = df[df['valid_ball'] == 1].copy()

# Dynamic context at moment of delivery
delivery['balls_remaining'] = 120 - delivery['team_balls']
delivery['wickets_remaining'] = 10 - delivery['team_wicket']
delivery['cum_run_rate'] = delivery['team_runs'] / (delivery['team_balls'] / 6).clip(lower=0.1)

# 2nd innings chase context
delivery['runs_required'] = delivery['runs_target'] - delivery['team_runs']
delivery['req_run_rate'] = np.where(
    (delivery['innings'] == 2) & (delivery['balls_remaining'] > 0),
    delivery['runs_required'] / (delivery['balls_remaining'] / 6).clip(lower=0.1),
    0
)

# Batter context at this moment
delivery['batter_sr_so_far'] = (delivery['batter_runs'] / delivery['batter_balls'].clip(lower=1) * 100)

# Phase encoding
delivery['phase_num'] = delivery['over'].apply(lambda x: 0 if x < 6 else (1 if x < 15 else 2))

# Pressure proxy for 1st innings: wickets lost + dot pressure + over number
# For 2nd innings: req_run_rate dominates
delivery['pressure_index'] = np.where(
    delivery['innings'] == 1,
    (delivery['team_wicket'] * 1.5 +
     delivery['phase_num'] * 2 +
     (120 - delivery['balls_remaining']) / 120 * 5),
    (delivery['req_run_rate'].clip(upper=20) +
     delivery['team_wicket'] * 1.5 +
     (120 - delivery['balls_remaining']) / 120 * 5)
)

print(f"Delivery-level data: {len(delivery):,} balls")
print(f"Pressure index range: {delivery['pressure_index'].min():.1f} to {delivery['pressure_index'].max():.1f}")


# === CELL 16: DIMENSION III — PRESSURE ZONE SOM ==============================

PRESSURE_FEATURES = [
    'cum_run_rate',
    'wickets_remaining',
    'balls_remaining',
    'batter_sr_so_far',
    'phase_num',
    'pressure_index',
    'req_run_rate',
    'team_wicket',
]

# Sample for training (full dataset is too large for SOM)
np.random.seed(42)
sample_idx = np.random.choice(len(delivery), size=min(50000, len(delivery)), replace=False)
delivery_sample = delivery.iloc[sample_idx].copy()

X_pressure = delivery_sample[PRESSURE_FEATURES].fillna(0).values
scaler_pressure = StandardScaler()
X_pressure_scaled = scaler_pressure.fit_transform(X_pressure)

DSOM_X, DSOM_Y = 5, 5
som_pressure = MiniSom(DSOM_X, DSOM_Y, X_pressure_scaled.shape[1],
                       sigma=1.5, learning_rate=0.5, random_seed=42,
                       neighborhood_function='gaussian')
som_pressure.random_weights_init(X_pressure_scaled)
som_pressure.train_random(X_pressure_scaled, 5000)

delivery_sample['psom_x'] = [som_pressure.winner(x)[0] for x in X_pressure_scaled]
delivery_sample['psom_y'] = [som_pressure.winner(x)[1] for x in X_pressure_scaled]
delivery_sample['pressure_node'] = delivery_sample['psom_x'] * DSOM_Y + delivery_sample['psom_y']

print(f"Pressure Zone SOM trained: {DSOM_X}x{DSOM_Y} = {DSOM_X*DSOM_Y} zones")
print(f"Quantization error: {som_pressure.quantization_error(X_pressure_scaled):.4f}")


# === CELL 17: DIMENSION III — PRESSURE ZONE HEATMAPS ========================

fig, axes = plt.subplots(2, 3, figsize=(20, 13))

pressure_heatmaps = {
    'pressure_index': ('Pressure Index', 'Reds'),
    'cum_run_rate': ('Cumulative Run Rate', 'YlOrRd'),
    'wickets_remaining': ('Wickets Remaining', 'RdYlGn'),
    'balls_remaining': ('Balls Remaining', 'Blues'),
    'req_run_rate': ('Required Run Rate', 'hot_r'),
    'batter_sr_so_far': ('Batter SR at Delivery', 'YlGn'),
}

for ax, (feat, (title, cmap)) in zip(axes.flatten(), pressure_heatmaps.items()):
    grid = np.zeros((DSOM_X, DSOM_Y))
    counts = np.zeros((DSOM_X, DSOM_Y))
    for _, row in delivery_sample.iterrows():
        grid[int(row['psom_x']), int(row['psom_y'])] += row[feat]
        counts[int(row['psom_x']), int(row['psom_y'])] += 1
    counts[counts == 0] = 1
    grid /= counts
    im = ax.imshow(grid, cmap=cmap, interpolation='nearest')
    ax.set_title(title, fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax, fraction=0.046)

plt.suptitle('DIMENSION III: Delivery Pressure Zone Map', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()


# === CELL 18: DIMENSION III — CLUTCH PLAYER METRIC ==========================
#
# THE KEY METRIC: performance STRICTLY within high-pressure zones
# This is objective and topology-derived — no arbitrary threshold.

# Identify high-pressure nodes (top 30% by avg pressure_index)
node_pressure = delivery_sample.groupby('pressure_node')['pressure_index'].mean()
pressure_threshold = node_pressure.quantile(0.70)
high_pressure_nodes = set(node_pressure[node_pressure >= pressure_threshold].index)

print(f"High-pressure nodes: {len(high_pressure_nodes)} of {DSOM_X*DSOM_Y}")
print(f"Pressure threshold: {pressure_threshold:.2f}")

delivery_sample['is_high_pressure'] = delivery_sample['pressure_node'].isin(high_pressure_nodes)

# Batter performance in high-pressure vs normal zones
batter_pressure = delivery_sample.groupby(['batter', 'is_high_pressure']).agg(
    balls = ('valid_ball', 'sum'),
    runs = ('runs_batter', 'sum'),
    boundaries = ('is_boundary', 'sum'),
    wickets = ('is_wicket', 'sum'),
    dots = ('is_dot', 'sum'),
).reset_index()

batter_pressure['sr'] = (batter_pressure['runs'] / batter_pressure['balls'].clip(lower=1) * 100).round(2)
batter_pressure['dot_pct'] = (batter_pressure['dots'] / batter_pressure['balls'].clip(lower=1) * 100).round(2)

# Pivot to get HP vs normal side by side
hp = batter_pressure[batter_pressure['is_high_pressure']].rename(
    columns={'sr':'hp_sr','balls':'hp_balls','runs':'hp_runs','dot_pct':'hp_dot_pct'})
normal = batter_pressure[~batter_pressure['is_high_pressure']].rename(
    columns={'sr':'normal_sr','balls':'normal_balls','runs':'normal_runs','dot_pct':'normal_dot_pct'})

clutch_df = hp[['batter','hp_balls','hp_runs','hp_sr','hp_dot_pct']].merge(
    normal[['batter','normal_balls','normal_runs','normal_sr','normal_dot_pct']], on='batter')

# Filter: min 30 HP balls AND min 50 normal balls (avoids tail-ender ratio inflation)
clutch_df = clutch_df[(clutch_df['hp_balls'] >= 30) & (clutch_df['normal_balls'] >= 50)].copy()

# CLUTCH METRIC = HP_SR / Normal_SR (>1 = performs BETTER under pressure)
clutch_df['clutch_ratio'] = (clutch_df['hp_sr'] / clutch_df['normal_sr'].clip(lower=1)).round(3)
clutch_df['sr_diff'] = clutch_df['hp_sr'] - clutch_df['normal_sr']
clutch_df = clutch_df.sort_values('clutch_ratio', ascending=False)

print("\n🏆 TOP CLUTCH PLAYERS (highest SR boost under pressure):")
print(clutch_df.head(15)[['batter','normal_sr','hp_sr','clutch_ratio','hp_balls','sr_diff']].to_string(index=False))

print("\n😰 PRESSURE CRUMBLERS (biggest SR drop under pressure):")
print(clutch_df.tail(15)[['batter','normal_sr','hp_sr','clutch_ratio','hp_balls','sr_diff']].to_string(index=False))


# === CELL 19: DIMENSION III — CLUTCH VISUALIZATION ===========================

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# 1. Clutch ratio scatter
top_n = clutch_df.nlargest(40, 'hp_balls')  # players with most HP balls
colors = ['#2ecc71' if x >= 1 else '#e74c3c' for x in top_n['clutch_ratio']]
axes[0].barh(top_n['batter'], top_n['clutch_ratio'], color=colors)
axes[0].axvline(x=1, color='black', linewidth=1.5, linestyle='--', label='Baseline (1.0)')
axes[0].set_xlabel('Clutch Ratio (HP Strike Rate / Normal Strike Rate)')
axes[0].set_title('Clutch Ratio: Performance Under Pressure\n'
                   '(>1 = better under pressure, <1 = worse)',
                   fontsize=13, fontweight='bold')
axes[0].legend()

# 2. SR in high-pressure vs normal
axes[1].scatter(top_n['normal_sr'], top_n['hp_sr'],
               s=top_n['hp_balls']*2, alpha=0.7,
               c=top_n['clutch_ratio'], cmap='RdYlGn',
               edgecolors='black', vmin=0.6, vmax=1.4)
max_sr = max(top_n['normal_sr'].max(), top_n['hp_sr'].max()) + 10
axes[1].plot([50, max_sr], [50, max_sr], 'k--', alpha=0.3, label='Equal Performance')
axes[1].set_xlabel('Normal Zone Strike Rate')
axes[1].set_ylabel('High-Pressure Zone Strike Rate')
axes[1].set_title('Strike Rate: High-Pressure vs Normal Zones\n'
                   '(Above line = clutch, below = crumbles)',
                   fontsize=13, fontweight='bold')

for _, row in top_n.nlargest(5, 'clutch_ratio').iterrows():
    axes[1].annotate(row['batter'], (row['normal_sr'], row['hp_sr']),
                    fontsize=7, fontweight='bold', color='green')
for _, row in top_n.nsmallest(5, 'clutch_ratio').iterrows():
    axes[1].annotate(row['batter'], (row['normal_sr'], row['hp_sr']),
                    fontsize=7, fontweight='bold', color='red')
axes[1].legend()

plt.tight_layout()
plt.show()


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  PHASE 5 — INTEGRATION WITH HACKATHON ML PIPELINE                        ║
# ║  Feed SOM-derived features into the match winner model                    ║
# ╚════════════════════════════════════════════════════════════════════════════╝

# === CELL 20: EXTRACT SOM-DERIVED FEATURES FOR ML ============================

# --- Feature 1: Team's typical game state trajectory (from Dimension I) ---
# For each team, what's their most common SOM node distribution?
team_state_profile = inn1.groupby(['batting_team','som_node']).size().unstack(fill_value=0)
team_state_profile = team_state_profile.div(team_state_profile.sum(axis=1), axis=0)
# Top 3 most-visited nodes per team
team_dom_states = {}
for team in team_state_profile.index:
    top3 = team_state_profile.loc[team].nlargest(3)
    team_dom_states[team] = {
        'dominant_node_1': top3.index[0],
        'dominant_node_2': top3.index[1],
        'dominant_node_3': top3.index[2],
        'state_entropy': -(team_state_profile.loc[team] *
                           np.log2(team_state_profile.loc[team].clip(lower=1e-10))).sum(),
    }
team_state_df = pd.DataFrame(team_dom_states).T
team_state_df.index.name = 'team'
team_state_df = team_state_df.reset_index()

# --- Feature 2: Player current archetype (from Dimension II) ---
# Get each player's LATEST archetype (most recent window)
latest_archetype = windows_df.sort_values('window_idx').groupby('batter').last()[
    ['archetype_node', 'archetype', 'avg', 'sr']
].reset_index()

# --- Feature 3: Team average clutch ratio (from Dimension III) ---
# Average clutch ratio of the batting lineup
team_clutch = clutch_df.merge(
    batter_inn[['batter','batting_team']].drop_duplicates(),
    on='batter'
)
team_avg_clutch = team_clutch.groupby('batting_team')['clutch_ratio'].mean().reset_index()
team_avg_clutch.columns = ['team', 'avg_clutch_ratio']

print("SOM-derived features ready:")
print(f"  Team state profiles: {len(team_state_df)} teams")
print(f"  Latest player archetypes: {len(latest_archetype)} players")
print(f"  Team clutch ratios: {len(team_avg_clutch)} teams")
print(f"\nThese features integrate into the match winner model from the hackathon.")


# === CELL 21: ENHANCED MATCH WINNER MODEL ====================================
# Rebuild the hackathon model with SOM features added

from sklearn.ensemble import GradientBoostingClassifier

# Load the match-level table from hackathon (rebuild quickly)
match_raw = df.drop_duplicates('match_id')[
    ['match_id','date','season','year','month','match_stage',
     'toss_winner','toss_decision','venue','city','match_won_by']
].copy()

teams = df.groupby('match_id')['batting_team'].first().reset_index()
teams.columns = ['match_id','team1']
teams2 = df[df['innings']==2].groupby('match_id')['batting_team'].first().reset_index()
teams2.columns = ['match_id','team2']
teams = teams.merge(teams2, on='match_id', how='left')
match_raw = match_raw.merge(teams, on='match_id')
match_raw = match_raw[match_raw['match_won_by'].notna()].copy()
match_raw['team1_won'] = (match_raw['team1'] == match_raw['match_won_by']).astype(int)

# Toss features
match_raw['toss_winner_is_team1'] = (match_raw['toss_winner'] == match_raw['team1']).astype(int)
match_raw['toss_chose_bat'] = (match_raw['toss_decision'] == 'bat').astype(int)

le_venue = LabelEncoder()
match_raw['venue_enc'] = le_venue.fit_transform(match_raw['venue'].astype(str))
le_team = LabelEncoder()
all_t = pd.concat([match_raw['team1'], match_raw['team2']]).unique()
le_team.fit(all_t)
match_raw['team1_enc'] = le_team.transform(match_raw['team1'])
match_raw['team2_enc'] = le_team.transform(match_raw['team2'])

stage_map = {'League': 0, 'Playoff': 1, 'Final': 2}
match_raw['stage_enc'] = match_raw['match_stage'].map(stage_map)

# Merge SOM features
match_raw = match_raw.merge(team_state_df.rename(columns={'team':'team1'}),
                            left_on='team1', right_on='team1', how='left',
                            suffixes=('','_t1'))
match_raw = match_raw.rename(columns={
    'dominant_node_1':'t1_dom_node','state_entropy':'t1_entropy'})

match_raw = match_raw.merge(team_avg_clutch.rename(columns={'team':'team1'}),
                            on='team1', how='left')
match_raw = match_raw.rename(columns={'avg_clutch_ratio':'t1_clutch'})

match_raw = match_raw.merge(team_avg_clutch.rename(columns={'team':'team2'}),
                            on='team2', how='left')
match_raw = match_raw.rename(columns={'avg_clutch_ratio':'t2_clutch'})

match_raw['clutch_diff'] = match_raw['t1_clutch'].fillna(1) - match_raw['t2_clutch'].fillna(1)

# Features
ENHANCED_FEATURES = [
    'team1_enc', 'team2_enc', 'venue_enc', 'stage_enc',
    'toss_winner_is_team1', 'toss_chose_bat', 'year', 'month',
    't1_clutch', 't2_clutch', 'clutch_diff',
    't1_entropy',
]

match_ml = match_raw.dropna(subset=['team1_won']).sort_values('date').reset_index(drop=True)
X = match_ml[[f for f in ENHANCED_FEATURES if f in match_ml.columns]].fillna(0)
y = match_ml['team1_won']

split = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

# Train
gb = GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                                subsample=0.8, random_state=42)
gb.fit(X_train, y_train)
y_pred = gb.predict(X_test)
y_prob = gb.predict_proba(X_test)[:, 1]

print(f"\n{'='*60}")
print(f"  ENHANCED MATCH WINNER MODEL (with SOM features)")
print(f"{'='*60}")
print(f"  Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"  F1 Score: {f1_score(y_test, y_pred):.4f}")
print(f"  AUC-ROC:  {roc_auc_score(y_test, y_prob):.4f}")

# Feature importance
imp = pd.Series(gb.feature_importances_,
                index=[f for f in ENHANCED_FEATURES if f in match_ml.columns])
print(f"\nFeature Importance:")
for feat, val in imp.sort_values(ascending=False).items():
    print(f"  {feat:<30s}: {val:.4f}")



# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  PHASE 6 — MATCH PREDICTION ENGINE                                        ║
# ║  Six prediction modes built on SOM-derived game state features:           ║
# ║   P1. Pre-match win probability          (Cell 21 — already built)        ║
# ║   P2. First-innings score forecasting                                     ║
# ║   P3. Live win probability — innings 1 (batting first)                    ║
# ║   P4. Live win probability — innings 2 (chase meter)                      ║
# ║   P5. Full match dynamics dashboard (6-panel visual)                      ║
# ║   P6. Over importance & venue tendency analysis                           ║
# ╚════════════════════════════════════════════════════════════════════════════╝

# === CELL 24: PREDICTION ENGINE SETUP — MODELS ==============================

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import brier_score_loss, mean_absolute_error
from sklearn.calibration import calibration_curve
import matplotlib.gridspec as gridspec

# ── Assign SOM nodes to innings-2 overs (using the innings-1 trained SOM) ───
inn2 = over_raw[over_raw['innings'] == 2].copy()
X_i2    = inn2[STATE_FEATURES].fillna(0).values
X_i2_sc = scaler_state.transform(X_i2)
inn2['som_x']    = [som_state.winner(x)[0] for x in X_i2_sc]
inn2['som_y']    = [som_state.winner(x)[1] for x in X_i2_sc]
inn2['som_node'] = inn2['som_x'] * SOM_Y + inn2['som_y']
inn2['runs_required'] = (inn2['target'] - inn2['cum_runs']).fillna(0)

# Add final score to innings-1 frame
final_scores_map = over_raw[over_raw['innings']==1].groupby('match_id')['cum_runs'].max()
inn1_ext = inn1.copy()
inn1_ext['final_score'] = inn1_ext['match_id'].map(final_scores_map)

# Node win-rate lookup (used in live predictions)
node_wr_map = inn1.groupby('som_node')['batting_team_won'].mean().to_dict()

# ── Feature sets ─────────────────────────────────────────────────────────────
INN1_WIN_FEATURES  = ['over','cum_run_rate','cum_wickets','momentum',
                      'wickets_this_over','boundaries','dots_this_over',
                      'wickets_remaining','som_node']
INN2_WIN_FEATURES  = ['over','cum_run_rate','cum_wickets','momentum',
                      'wickets_remaining','req_run_rate','pressure_ratio',
                      'runs_required','som_node']
SCORE_FEAT         = ['over','cum_run_rate','cum_wickets','momentum',
                      'wickets_remaining','boundaries','dots_this_over',
                      'sixes_this_over','som_node']

# ── Time-based train / test split (2022 cutoff keeps 2023-25 as holdout) ────
CUTOFF = 2022
i1_tr = inn1_ext[inn1_ext['year'] <= CUTOFF]
i1_te = inn1_ext[inn1_ext['year'] >  CUTOFF]
i2_tr = inn2[inn2['year'] <= CUTOFF]
i2_te = inn2[inn2['year'] >  CUTOFF]

# ── P3: Live win probability — innings 1 ────────────────────────────────────
win_model_inn1 = GradientBoostingClassifier(
    n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=42)
win_model_inn1.fit(i1_tr[INN1_WIN_FEATURES].fillna(0), i1_tr['batting_team_won'])

# ── P4: Live win probability — innings 2 (chase) ────────────────────────────
win_model_inn2 = GradientBoostingClassifier(
    n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=42)
win_model_inn2.fit(i2_tr[INN2_WIN_FEATURES].fillna(0), i2_tr['batting_team_won'])

# ── P2: First-innings score forecaster ───────────────────────────────────────
score_model = GradientBoostingRegressor(
    n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8, random_state=42)
score_model.fit(i1_tr[SCORE_FEAT].fillna(0), i1_tr['final_score'])

# ── Evaluation on held-out years ─────────────────────────────────────────────
wp1_te  = win_model_inn1.predict_proba(i1_te[INN1_WIN_FEATURES].fillna(0))[:,1]
wp2_te  = win_model_inn2.predict_proba(i2_te[INN2_WIN_FEATURES].fillna(0))[:,1]
sp_te   = score_model.predict(i1_te[SCORE_FEAT].fillna(0))

print(f"{'='*55}")
print(f"  PREDICTION ENGINE — Model Quality (2023-25 holdout)")
print(f"{'='*55}")
print(f"  Win model (Inn 1) Brier: {brier_score_loss(i1_te['batting_team_won'], wp1_te):.4f}  "
      f"(baseline 0.25 = random)")
print(f"  Win model (Inn 2) Brier: {brier_score_loss(i2_te['batting_team_won'], wp2_te):.4f}")
print(f"  Score forecaster   MAE:  {mean_absolute_error(i1_te['final_score'], sp_te):.1f} runs")
print(f"  Training overs     Inn1: {len(i1_tr):,} | Inn2: {len(i2_tr):,}")
print(f"  Holdout  overs     Inn1: {len(i1_te):,} | Inn2: {len(i2_te):,}")


# === CELL 25: CALIBRATION PLOTS & FEATURE IMPORTANCE ========================

fig, axes = plt.subplots(1, 3, figsize=(21, 6))

# Calibration — Inn 1
frac1, mpred1 = calibration_curve(i1_te['batting_team_won'], wp1_te, n_bins=10)
axes[0].plot(mpred1, frac1, 's-b', lw=2, ms=6, label='Inn 1 model')
axes[0].plot([0,1],[0,1],'k--',alpha=0.4,label='Perfect')
axes[0].set(xlabel='Predicted probability', ylabel='Actual win rate',
            title='Model Calibration — Innings 1 Win Probability')
axes[0].legend()

# Calibration — Inn 2
frac2, mpred2 = calibration_curve(i2_te['batting_team_won'], wp2_te, n_bins=10)
axes[1].plot(mpred2, frac2, 's-r', lw=2, ms=6, label='Inn 2 (chase) model')
axes[1].plot([0,1],[0,1],'k--',alpha=0.4,label='Perfect')
axes[1].set(xlabel='Predicted probability', ylabel='Actual win rate',
            title='Model Calibration — Innings 2 Win Probability')
axes[1].legend()

# Feature importance — combined
feat_imp1 = pd.Series(win_model_inn1.feature_importances_, index=INN1_WIN_FEATURES)
feat_imp2 = pd.Series(win_model_inn2.feature_importances_, index=INN2_WIN_FEATURES)
all_feats  = sorted(set(INN1_WIN_FEATURES) | set(INN2_WIN_FEATURES))
imp_df = pd.DataFrame({'Inn 1': feat_imp1.reindex(all_feats, fill_value=0),
                        'Inn 2': feat_imp2.reindex(all_feats, fill_value=0)},
                       index=all_feats)
imp_df.sort_values('Inn 1', ascending=True).plot.barh(ax=axes[2], color=['steelblue','tomato'])
axes[2].set_title('Feature Importance — Win Probability Models')
axes[2].set_xlabel('Importance')

plt.suptitle('PHASE 6: Prediction Engine — Model Diagnostics', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

print("\nTop features driving win probability:")
print("  Inn 1:", feat_imp1.sort_values(ascending=False).head(4).to_dict())
print("  Inn 2:", feat_imp2.sort_values(ascending=False).head(4).to_dict())


# === CELL 26: P5 — FULL MATCH DYNAMICS DASHBOARD ============================
#
# Six-panel dashboard for any match_id:
#   [1] Win probability timeline (both innings, turning points annotated)
#   [2] Score forecast vs actual (innings 1)
#   [3] SOM game-state trajectory on win-rate background
#   [4] Momentum heatmap — full 40-over colour bar
#   [5] Over-by-over run rate (actual vs projected)
#   [6] Wicket timeline + pressure index

def match_dashboard(match_id):
    """Full prediction dashboard for a single IPL match."""
    m1 = inn1_ext[inn1_ext['match_id'] == match_id].sort_values('over')
    m2 = inn2[inn2['match_id'] == match_id].sort_values('over')
    if len(m1) == 0:
        print(f"Match {match_id} not found in dataset.")
        return

    team1  = m1['batting_team'].iloc[0]
    team2  = m2['batting_team'].iloc[0] if len(m2) > 0 else '?'
    winner = m1['match_won_by'].iloc[0]
    score1 = int(m1['cum_runs'].iloc[-1])
    score2 = int(m2['cum_runs'].iloc[-1]) if len(m2) > 0 else 0

    # ── Win probability ───────────────────────────────────────────────────────
    wp1 = win_model_inn1.predict_proba(m1[INN1_WIN_FEATURES].fillna(0))[:,1]
    sp1 = score_model.predict(m1[SCORE_FEAT].fillna(0))

    if len(m2) > 0:
        wp2_raw = win_model_inn2.predict_proba(m2[INN2_WIN_FEATURES].fillna(0))[:,1]
        wp2 = 1 - wp2_raw  # convert to: P(team1/batting-first team wins)
    else:
        wp2 = np.array([])

    all_wp    = np.concatenate([wp1, wp2]) if len(wp2) else wp1
    all_overs = list(m1['over'].values) + [20 + o for o in (m2['over'].values if len(m2) else [])]

    # Turning points: biggest single-over probability swings
    wp_diff   = np.diff(all_wp)
    turn_idxs = np.argsort(np.abs(wp_diff))[::-1][:6]
    turning   = [(all_overs[i], wp_diff[i]) for i in sorted(turn_idxs)]

    fig = plt.figure(figsize=(22, 20))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.5, wspace=0.3)

    # ── Panel 1: Win probability timeline ────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ovs1 = m1['over'].values
    ax1.plot(ovs1, wp1, 'b-o', ms=4, lw=2.5, label=f'{team1} (batting 1st)')
    ax1.fill_between(ovs1, 0.5, wp1, where=wp1>=0.5, alpha=0.15, color='blue')
    ax1.fill_between(ovs1, wp1, 0.5, where=wp1<0.5, alpha=0.15, color='red')
    if len(wp2):
        ovs2 = [20+o for o in m2['over'].values]
        ax1.plot(ovs2, wp2, 'r-o', ms=4, lw=2.5, label=f'{team1} after {team2} bats')
        ax1.fill_between(ovs2, 0.5, wp2, where=wp2>=0.5, alpha=0.15, color='blue')
        ax1.fill_between(ovs2, wp2, 0.5, where=wp2<0.5, alpha=0.15, color='red')
    ax1.axhline(0.5, color='gray', ls='--', alpha=0.4)
    ax1.axvline(19.5, color='black', lw=2, alpha=0.5, label='Innings break')
    for ov, chg in turning:
        col = '#e67e22' if abs(chg) > 0.10 else '#f0b429'
        ax1.axvline(ov, color=col, alpha=0.6, lw=1.8)
        ax1.annotate(f"O{int(ov) if ov<20 else int(ov)-20}{'*' if abs(chg)>0.10 else ''}",
                     xy=(ov, 0.97), fontsize=8, ha='center', color=col, fontweight='bold')
    ax1.set(xlabel='Over (0–19: Inn1 | 20–39: Inn2)', ylabel=f'P({team1} wins)',
            ylim=(0,1), xlim=(-0.5, max(all_overs)+0.5),
            title=f'Win Probability Timeline — {team1} {score1}  vs  {team2} {score2}  |  Winner: {winner}')
    ax1.legend(loc='upper right', fontsize=9)

    # ── Panel 2: Score forecast vs actual ────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(ovs1, m1['cum_runs'].values, 'b-', lw=2.5, label='Actual score')
    ax2.plot(ovs1, sp1, 'b--', lw=2, alpha=0.7, label='Forecast final')
    ax2.fill_between(ovs1, sp1*0.92, sp1*1.08, alpha=0.12, color='blue', label='±8% band')
    ax2.axhline(score1, color='darkgreen', ls=':', lw=1.5, label=f'Actual final: {score1}')
    ax2.set(xlabel='Over', ylabel='Runs',
            title=f'{team1} — Score Forecasting (Inn 1)')
    ax2.legend(fontsize=9)

    # ── Panel 3: SOM trajectory ───────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    wr_grid = np.zeros((SOM_X, SOM_Y)); cnt = np.zeros((SOM_X, SOM_Y))
    for _, r in inn1_ext.iterrows():
        wr_grid[int(r['som_x']),int(r['som_y'])] += r['batting_team_won']
        cnt[int(r['som_x']),int(r['som_y'])]     += 1
    cnt[cnt==0]=1; wr_grid/=cnt
    ax3.imshow(wr_grid, cmap='RdYlGn', alpha=0.35, interpolation='nearest',
               extent=[-0.5,SOM_Y-0.5,SOM_X-0.5,-0.5])
    xs = m1['som_y'].values + np.random.normal(0,0.07,len(m1))
    ys = m1['som_x'].values + np.random.normal(0,0.07,len(m1))
    segs = np.concatenate([np.array([xs,ys]).T.reshape(-1,1,2)[:-1],
                            np.array([xs,ys]).T.reshape(-1,1,2)[1:]],axis=1)
    lc3 = LineCollection(segs, cmap='plasma', norm=Normalize(0,19), lw=2.5, alpha=0.85)
    lc3.set_array(m1['over'].values[:-1]); ax3.add_collection(lc3)
    ax3.scatter(xs[0],ys[0],s=180,c='lime',marker='o',zorder=5,edgecolors='k',lw=1.5,label='Start')
    ax3.scatter(xs[-1],ys[-1],s=180,c='red',marker='*',zorder=5,edgecolors='k',lw=1.5,label='End')
    ax3.set(xlim=(-0.5,SOM_Y-0.5),ylim=(SOM_X-0.5,-0.5),
            title=f'{team1} Inn1 — SOM Trajectory\n(Background: win-rate heatmap)')
    ax3.legend(fontsize=8)

    # ── Panel 4: Match dynamics heatmap (the 40-over colour bar) ─────────────
    ax4 = fig.add_subplot(gs[2, :])
    hm = np.array(all_wp).reshape(1, -1)
    im = ax4.imshow(hm, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax4.axvline(len(wp1)-0.5, color='white', lw=3)

    # Wicket markers (▼)
    for idx, (_, r) in enumerate(m1.iterrows()):
        if r['wickets_this_over'] > 0:
            ax4.scatter([idx],[0],marker='v',s=200,color='black',zorder=5)
            ax4.annotate(f"W{int(r['wickets_this_over'])}",
                        xy=(idx,-0.35),fontsize=7,ha='center',color='black')
    if len(m2):
        for idx, (_, r) in enumerate(m2.iterrows()):
            if r['wickets_this_over'] > 0:
                ax4.scatter([len(wp1)+idx],[0],marker='v',s=200,color='black',zorder=5)
                ax4.annotate(f"W{int(r['wickets_this_over'])}",
                            xy=(len(wp1)+idx,-0.35),fontsize=7,ha='center',color='black')

    # Turning point markers
    for ov, chg in turning:
        idx = all_overs.index(ov)
        ax4.axvline(idx, color='orange', alpha=0.9, lw=2.5)

    tick_step = 2
    tpos = list(range(0, len(all_wp), tick_step))
    tlbl = [str(int(all_overs[i])) if all_overs[i]<20 else f"I2-{int(all_overs[i]-20)}"
            for i in tpos if i < len(all_overs)]
    ax4.set_xticks(tpos[:len(tlbl)]); ax4.set_xticklabels(tlbl, fontsize=8)
    ax4.set_yticks([])
    ax4.set_title(
        f'Match Dynamics Heatmap — {team1} Win Probability (Green={team1} ahead | Red={team2} ahead)\n'
        f'▼=Wicket  |  Orange line=Major turning point  |  White line=Innings break')
    plt.colorbar(im, ax=ax4, label='Win Prob', fraction=0.008, pad=0.01)

    plt.suptitle(f'MATCH PREDICTION DASHBOARD  |  {team1}  vs  {team2}\n'
                 f'Scores: {score1} vs {score2}  |  Winner: {winner}',
                 fontsize=15, fontweight='bold', y=1.0)
    plt.show()

    # Turning points summary
    print(f"\n{'─'*60}")
    print(f"  KEY TURNING POINTS (sorted by impact)")
    print(f"{'─'*60}")
    for ov, chg in sorted(turning, key=lambda x: abs(x[1]), reverse=True):
        inn = 1 if ov < 20 else 2; onum = int(ov) if ov<20 else int(ov)-20
        direction = f"{team1} ↑" if chg > 0 else f"{team2} ↑"
        print(f"  Inn{inn} Over {onum:2d}: {direction}   swing {abs(chg):.1%}")

    return {'match_id': match_id, 'win_prob_inn1': wp1,
            'score_forecast': sp1, 'turning_points': turning}


# Demo on 3 contrasting matches
sample_ids = (inn1_ext.groupby('match_id')['batting_team_won']
              .first().reset_index())
won_ids  = sample_ids[sample_ids['batting_team_won']==1]['match_id'].iloc[5]
lost_ids = sample_ids[sample_ids['batting_team_won']==0]['match_id'].iloc[5]

print("=== DEMO: Match won by team batting first ===")
match_dashboard(won_ids)
print("\n=== DEMO: Match won by team chasing ===")
match_dashboard(lost_ids)


# === CELL 27: P3+P4 — LIVE PREDICTION INTERFACE ============================
#
# Use this to predict mid-match: just supply the current game state.
# No ball-by-ball data needed — a commentator could fill this in live.

def predict_live(innings, over, cum_runs, cum_wickets,
                 target=None, momentum=0.0, boundaries=1,
                 dots=2, wickets_this_over=0, sixes=0):
    """
    Live win probability + score forecast at any moment.

    innings          : 1 or 2
    over             : completed overs (0–19)
    cum_runs         : total runs so far
    cum_wickets      : wickets fallen
    target           : required total (innings 2 only)
    momentum         : this over's RR minus overall RR (optional, default 0)
    boundaries/dots/wickets_this_over/sixes : this over's stats (optional)
    """
    balls_done      = over * 6
    balls_remaining = 120 - balls_done
    cum_rr          = cum_runs / max(over, 0.5)
    wkts_remaining  = 10 - cum_wickets
    over_rr         = cum_rr + momentum

    # Resolve SOM node from current state
    sv     = np.array([[cum_rr, over_rr, momentum, wickets_this_over,
                        wkts_remaining, dots, boundaries, cum_wickets, over]])
    sv_sc  = scaler_state.transform(sv)
    winner = som_state.winner(sv_sc[0])
    snode  = winner[0] * SOM_Y + winner[1]

    def _bar(p, width=40):
        filled = int(p * width)
        return f"[{'█'*filled}{'░'*(width-filled)}] {p:.1%}"

    if innings == 1:
        feat   = np.array([[over, cum_rr, cum_wickets, momentum, wickets_this_over,
                            boundaries, dots, wkts_remaining, snode]])
        sfeat  = np.array([[over, cum_rr, cum_wickets, momentum, wkts_remaining,
                            boundaries, dots, sixes, snode]])
        wp     = win_model_inn1.predict_proba(feat)[0, 1]
        sp     = score_model.predict(sfeat)[0]
        nwr    = node_wr_map.get(snode, 0.5)

        print(f"\n{'═'*58}")
        print(f"  LIVE PREDICTION  ·  Innings 1  ·  End of Over {over}")
        print(f"{'═'*58}")
        print(f"  Score:            {cum_runs}/{cum_wickets}  ({cum_rr:.1f} RR)")
        print(f"  Balls remaining:  {balls_remaining}")
        print(f"  SOM game state:   Node {snode} — {node_labels.get(snode, '?')}")
        print(f"  Node hist. W/R:   {nwr:.1%}")
        print(f"{'─'*58}")
        print(f"  Win probability:  {_bar(wp)}")
        print(f"  Score forecast:   {sp:.0f} runs  (range: {sp*0.92:.0f}–{sp*1.08:.0f})")
        simple = cum_rr * 20
        print(f"  Simple pace proj: {simple:.0f} runs")
        return wp, sp

    else:  # innings == 2
        if target is None:
            print("Provide target for innings 2."); return
        runs_req    = target - cum_runs
        req_rr      = runs_req / max(balls_remaining / 6, 0.1)
        press_ratio = req_rr  / max(cum_rr, 0.1)

        feat = np.array([[over, cum_rr, cum_wickets, momentum, wkts_remaining,
                          req_rr, press_ratio, runs_req, snode]])
        wp_chaser   = win_model_inn2.predict_proba(feat)[0, 1]
        wp_defender = 1 - wp_chaser

        print(f"\n{'═'*58}")
        print(f"  LIVE PREDICTION  ·  Innings 2  ·  End of Over {over}")
        print(f"{'═'*58}")
        print(f"  Chaser:           {cum_runs}/{cum_wickets}  ({cum_rr:.1f} RR)")
        print(f"  Target:           {target}  |  Need: {runs_req} off {balls_remaining} balls")
        print(f"  Required RR:      {req_rr:.2f}  |  Pressure ratio: {press_ratio:.2f}")
        print(f"  SOM game state:   Node {snode} — {node_labels.get(snode, '?')}")
        print(f"{'─'*58}")
        print(f"  Chaser wins:      {_bar(wp_chaser)}")
        print(f"  Defender wins:    {_bar(wp_defender)}")
        return wp_chaser, wp_defender


# ── Live prediction demos ─────────────────────────────────────────────────────
print("Example 1: Innings 1, over 6, 52/0 (flying start)")
predict_live(1, over=6, cum_runs=52, cum_wickets=0, boundaries=2, dots=1)

print("\nExample 2: Innings 1, over 12, 78/4 (collapse)")
predict_live(1, over=12, cum_runs=78, cum_wickets=4, boundaries=0, dots=3, wickets_this_over=2)

print("\nExample 3: Innings 2, over 15, 115/3, chasing 162")
predict_live(2, over=15, cum_runs=115, cum_wickets=3, target=162, boundaries=2, dots=1)

print("\nExample 4: Innings 2, over 18, 135/7, chasing 155 (death overs drama)")
predict_live(2, over=18, cum_runs=135, cum_wickets=7, target=155, boundaries=1, dots=3)


# === CELL 28: P6 — OVER IMPORTANCE & VENUE TENDENCY ANALYSIS ================

fig, axes = plt.subplots(1, 3, figsize=(22, 7))

# ── P6a: Over importance — which over drives win probability most? ───────────
imp_by_over = pd.Series(score_model.feature_importances_, index=SCORE_FEAT)
over_impact = (inn1_ext.groupby('over')
               .apply(lambda g: np.corrcoef(
                   win_model_inn1.predict_proba(g[INN1_WIN_FEATURES].fillna(0))[:,1],
                   g['batting_team_won'])[0,1])
               .reset_index(name='win_corr'))

bars = axes[0].bar(over_impact['over'], over_impact['win_corr'],
                   color=[plt.cm.RdYlGn(v*0.5+0.5) for v in over_impact['win_corr']])
axes[0].axhline(0, color='gray', lw=0.8, ls='--')
axes[0].set(xlabel='Over', ylabel='Corr(predicted win prob, actual outcome)',
            title='Over Importance — How Much Each Over\nPredicts the Final Result')
for phase, (start, end, col) in zip(['PP','Mid','Death'],[(0,5,'#3498db'),(6,14,'#2ecc71'),(15,19,'#e74c3c')]):
    axes[0].axvspan(start-0.4, end+0.4, alpha=0.07, color=col)
    axes[0].text((start+end)/2, axes[0].get_ylim()[1]*0.97, phase,
                 ha='center', fontsize=9, color=col, fontweight='bold')

# ── P6b: Venue win-rate heatmap (bat first vs chase) ─────────────────────────
venue_stats = (over_raw[over_raw['innings']==1]
               .drop_duplicates('match_id')
               .groupby('venue')['batting_team_won'].agg(['mean','count'])
               .reset_index())
venue_stats.columns = ['venue','bat_first_wr','matches']
top_venues = venue_stats.nlargest(15,'matches').sort_values('bat_first_wr')
colors_v = ['#e74c3c' if v<0.5 else '#2ecc71' for v in top_venues['bat_first_wr']]
axes[1].barh(range(len(top_venues)), top_venues['bat_first_wr']-0.5,
             left=0.5, color=colors_v, alpha=0.8)
axes[1].axvline(0.5, color='black', lw=1.5, ls='--')
axes[1].set_yticks(range(len(top_venues)))
axes[1].set_yticklabels([v[:28] for v in top_venues['venue']], fontsize=8)
axes[1].set(xlabel='Bat-first win rate',
            title='Venue Tendency — Bat First vs Chase\n(Green=favours batting first)')
for i, (_, row) in enumerate(top_venues.iterrows()):
    axes[1].text(row['bat_first_wr']+0.005*(1 if row['bat_first_wr']>0.5 else -1),
                 i, f"{row['bat_first_wr']:.0%} ({int(row['matches'])}m)",
                 va='center', fontsize=7)

# ── P6c: Win probability at each over (average across all matches) ───────────
wp_by_over = (inn1_ext.groupby('over')
              .apply(lambda g: pd.Series({
                  'avg_wp': win_model_inn1.predict_proba(g[INN1_WIN_FEATURES].fillna(0))[:,1].mean(),
                  'winner_avg_wp': win_model_inn1.predict_proba(
                      g[g['batting_team_won']==1][INN1_WIN_FEATURES].fillna(0))[:,1].mean()
                      if (g['batting_team_won']==1).any() else np.nan,
                  'loser_avg_wp': win_model_inn1.predict_proba(
                      g[g['batting_team_won']==0][INN1_WIN_FEATURES].fillna(0))[:,1].mean()
                      if (g['batting_team_won']==0).any() else np.nan,
              })).reset_index())

axes[2].plot(wp_by_over['over'], wp_by_over['winner_avg_wp'], 'g-o', ms=5, lw=2, label='Winning innings')
axes[2].plot(wp_by_over['over'], wp_by_over['loser_avg_wp'],  'r-o', ms=5, lw=2, label='Losing innings')
axes[2].fill_between(wp_by_over['over'],
                     wp_by_over['winner_avg_wp'],
                     wp_by_over['loser_avg_wp'],
                     alpha=0.15, color='green')
axes[2].axhline(0.5, color='gray', ls='--', alpha=0.5)
axes[2].set(xlabel='Over', ylabel='Average predicted win probability',
            title='Avg Win Probability Trajectory\n(Winning vs Losing innings)',
            ylim=(0.3, 0.8))
axes[2].legend()

plt.suptitle('PHASE 6: Over Importance, Venue Tendencies & Win Probability Separation',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Summarise venue findings
print("\nVenues that favour BATTING FIRST (bat-first WR > 55%):")
for _, r in top_venues[top_venues['bat_first_wr']>0.55].iterrows():
    print(f"  {r['venue'][:45]:<45} {r['bat_first_wr']:.0%}  ({int(r['matches'])} matches)")
print("\nVenues that favour CHASING (bat-first WR < 45%):")
for _, r in top_venues[top_venues['bat_first_wr']<0.45].iterrows():
    print(f"  {r['venue'][:45]:<45} {r['bat_first_wr']:.0%}  ({int(r['matches'])} matches)")


# === CELL 23: PROJECT SUMMARY ================================================
print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║     MULTI-OBJECTIVE UNSUPERVISED LEARNING FRAMEWORK — COMPLETE           ║
╠═══════════════════════════════════════════════════════════════════════════╣
║                                                                           ║
║  DIMENSION I: Match Momentum SOM                                          ║
║    • 6×6 topological map of 36 game states                                ║
║    • Single match = trajectory across the grid                            ║
║    • Winning innings cluster in different regions than losing              ║
║    • Momentum shifts are visible as trajectory direction changes           ║
║                                                                           ║
║  DIMENSION II: Player Archetype Evolution SOM                             ║
║    • 5×5 map of 25 player archetypes                                      ║
║    • 15-match rolling windows track form evolution                         ║
║    • Career migration shows exact transition points                       ║
║    • Transition matrix reveals typical evolution pathways                  ║
║                                                                           ║
║  DIMENSION III: Delivery Pressure Indexing SOM                            ║
║    • 5×5 map of 25 pressure zones                                         ║
║    • Purely objective clutch metric from topology                         ║
║    • Cross-references batter SR in high-pressure vs normal zones          ║
║    • No arbitrary thresholds — pressure zones emerge from data            ║
║                                                                           ║
║  INTEGRATION:                                                             ║
║    • SOM features (clutch ratio, state entropy, archetype) fed into       ║
║      the hackathon match winner model                                     ║
║    • External dataset fusion protocol ready for CT2025 + ICC data         ║
║                                                                           ║
║  ALGORITHMS USED:                                                         ║
║    • Self-Organizing Maps (MiniSom) — all 3 dimensions                    ║
║    • Gradient Boosting / XGBoost — enhanced match prediction              ║
║    • StandardScaler — feature normalization for SOM input                  ║
║    • Chi-Square — statistical significance testing                        ║
║                                                                           ║
║  NEXT STEPS:                                                              ║
║    1. Download external datasets and run fusion protocol (Cell 22)        ║
║    2. Train SOMs on combined multi-format data                            ║
║    3. Compare T20 vs ODI topologies                                       ║
║    4. Build a real-time dashboard for live match trajectory tracking       ║
╚═══════════════════════════════════════════════════════════════════════════╝
""")


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  PHASE 7 — ENHANCED FEATURE ENGINEERING v2                                ║
# ║  Expand over-state features from 9 → 28 independent signals:             ║
# ║   • Temporal lags & EMA (rolling 3-over context)                          ║
# ║   • Resource utilization & structural metrics                             ║
# ║   • Partnership stability & post-wicket recovery                          ║
# ║   • Aggression index, score volatility, collapse/surge flags              ║
# ║   • Chase enrichments: gradient, DLS proxy                                ║
# ║  Also adds a 4th SOM dimension: Bowler Archetypes                         ║
# ╚════════════════════════════════════════════════════════════════════════════╝

# === CELL 29: OVER-STATE FEATURE EXPANSION ===================================
#
# WHY EACH NEW FEATURE GROUP:
#
# Temporal lags (rr_last3, wkts_last3, …)
#   The original SOM treats every over as i.i.d.  A team at 60/0 in over 6
#   after scoring 10-12-10-9-9 looks identical to a team that just scored
#   3-3-3-3-3-25.  Lag features break this: they encode LOCAL momentum.
#
# Resource utilization (resource_util, boundary_dep, batting_aggression)
#   "Over 12" is ambiguous.  "56% of innings consumed" is universal.
#   boundary_dep captures HOW the runs are being scored (hitting vs rotating).
#
# Partnership stability (balls_since_wicket, partnership_rr)
#   A team at 80/2 with a 30-ball partnership is tactically different from
#   one that just lost a wicket and is at 80/2 ball 1 of a new pair.
#
# Event indicators (collapse_flag, surge_flag)
#   Hard binary signals the SOM can use to segregate crisis from cruise states.
#
# Chase enrichments (chase_gradient, dls_resource)
#   Innings-2 context is fundamentally different; these features are 0 for
#   innings 1 so they don't pollute the bat-first cluster structure.

import itertools

# ── Step 0: Ensure chronological sort ────────────────────────────────────────
over_raw = over_raw.sort_values(['match_id', 'innings', 'over']).reset_index(drop=True)
_GRP = ['match_id', 'innings']

# ── A. Temporal lag features (last 3 overs, shift(1) to avoid leakage) ───────
for _raw, _new in [
    ('over_run_rate',     'rr_last3'),
    ('wickets_this_over', 'wkts_last3'),
    ('boundaries',        'boundaries_last3'),
    ('dots_this_over',    'dots_last3'),
]:
    over_raw[_new] = (over_raw.groupby(_GRP)[_raw]
                     .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
                     .fillna(0))

# Score volatility last 3 overs: high value = chaotic/unpredictable scoring
over_raw['score_vol_3'] = (over_raw.groupby(_GRP)['runs_this_over']
                           .transform(lambda x: x.shift(1).rolling(3, min_periods=2).std())
                           .fillna(0))

# EMA of over run rates (α=0.3 → recent overs dominate; decays ~3 overs back)
over_raw['rr_ema'] = (over_raw.groupby(_GRP)['over_run_rate']
                     .transform(lambda x: x.ewm(alpha=0.3, adjust=False).mean())
                     .fillna(0))

# ── B. Resource & structural features ─────────────────────────────────────────
over_raw['resource_util']     = over_raw['cum_balls'] / 120.0
over_raw['over_norm']         = over_raw['over'] / 19.0
over_raw['phase_enc']         = over_raw['over'].apply(
    lambda x: 0 if x < 6 else (1 if x < 15 else 2))  # 0=PP, 1=Mid, 2=Death

# Boundary dependency: what fraction of cumulative runs came from fours/sixes?
_cb = over_raw.groupby(_GRP)['boundaries'].cumsum()
_cs = over_raw.groupby(_GRP)['sixes_this_over'].cumsum()
over_raw['boundary_dep'] = ((_cb - _cs) * 4 + _cs * 6) / over_raw['cum_runs'].clip(lower=1)

# Batting aggression of THIS over (intent signal for the SOM)
over_raw['batting_aggression'] = (
    (over_raw['sixes_this_over'] * 2 + over_raw['boundaries']) /
    over_raw['valid_balls'].clip(lower=1))

# Score-vs-par ratio: is this innings tracking ahead or behind historical average?
_par = over_raw[over_raw['innings'] == 1].groupby('over')['cum_runs'].mean().to_dict()
over_raw['score_par_ratio'] = (
    over_raw['cum_runs'] / over_raw['over'].map(_par).fillna(50).clip(lower=1))

# ── C. Partnership stability (ball-by-ball computation) ───────────────────────
# Identify current batting partnership at the END of each over:
#   balls_since_wicket = how many valid balls since the last dismissal
#   partnership_rr     = scoring rate of the current pair

_d = df[df['valid_ball'] == 1].sort_values(
    ['match_id', 'innings', 'team_balls']).copy()

# Partnership group: increments after EACH wicket (shift so wicket ball is
# in the ENDING partnership, next ball starts a new group)
_d['_wkt_flag'] = _d['wicket_kind'].notna().astype(int)
_d['_wkt_group'] = (_d.groupby(['match_id', 'innings'])['_wkt_flag']
                    .transform(lambda x: x.cumsum().shift(1).fillna(0)))
_d['balls_in_pship'] = _d.groupby(['match_id', 'innings', '_wkt_group']).cumcount()
_d['pship_runs']     = (_d.groupby(['match_id', 'innings', '_wkt_group'])['runs_batter']
                        .transform('cumsum'))

_over_pship = (_d.groupby(['match_id', 'innings', 'over'])
               .agg(balls_since_wicket=('balls_in_pship', 'max'),
                    partnership_runs  =('pship_runs',     'max'))
               .reset_index())

over_raw = over_raw.merge(_over_pship, on=['match_id', 'innings', 'over'], how='left')
over_raw['balls_since_wicket'] = over_raw['balls_since_wicket'].fillna(0)
over_raw['partnership_rr']     = (
    over_raw['partnership_runs'] /
    (over_raw['balls_since_wicket'].clip(lower=1) / 6)).fillna(0).clip(upper=50)

# ── D. Event indicator flags ───────────────────────────────────────────────────
# collapse_flag: avg ≥ 0.67 wkts/over in last 3 = roughly 2 wkts in 3 overs
over_raw['collapse_flag'] = (over_raw['wkts_last3'] >= 0.67).astype(float)
# surge_flag: recent RR at least 50% above overall RR (acceleration phase)
over_raw['surge_flag'] = (
    over_raw['rr_last3'] > over_raw['cum_run_rate'] * 1.5).astype(float)

# ── E. Chase-specific enrichments (zero for innings 1) ────────────────────────
# chase_gradient: is required RR getting harder (+) or easier (-) over last 3 overs?
over_raw['chase_gradient'] = np.where(
    over_raw['innings'] == 2,
    over_raw.groupby(_GRP)['req_run_rate']
            .transform(lambda x: x.diff(3).fillna(0)),
    0.0)

# Simplified DLS resource proxy: balls_remaining × f(wickets_remaining)
# Captures the interaction between two resource axes that req_rr alone misses
over_raw['dls_resource'] = np.where(
    over_raw['innings'] == 2,
    (over_raw['balls_remaining'] / 120) * (over_raw['wickets_remaining'] / 10) ** 0.7,
    0.0)

# ── F. Define the full 28-feature enhanced state vector ───────────────────────
ENHANCED_STATE_FEATURES = [
    # Original signals (retained for interpretability)
    'cum_run_rate', 'over_run_rate', 'momentum',
    'wickets_this_over', 'wickets_remaining', 'cum_wickets',
    'dots_this_over', 'boundaries',
    # Temporal context
    'rr_last3', 'wkts_last3', 'boundaries_last3', 'dots_last3',
    'score_vol_3', 'rr_ema',
    # Resource & structure
    'resource_util', 'over_norm', 'phase_enc',
    'boundary_dep', 'batting_aggression', 'score_par_ratio',
    # Partnership stability
    'balls_since_wicket', 'partnership_rr',
    # Event indicators
    'collapse_flag', 'surge_flag',
    # Chase context (0 for innings 1)
    'req_run_rate', 'pressure_ratio', 'chase_gradient', 'dls_resource',
]

_nan_count = over_raw[ENHANCED_STATE_FEATURES].isna().sum().sum()
print(f"Feature engineering v2 complete.")
print(f"  over_raw shape: {over_raw.shape}")
print(f"  Enhanced feature count: {len(ENHANCED_STATE_FEATURES)}")
print(f"  Total NaNs (will be filled): {_nan_count}")
print(f"\n  New features added:")
_new_feats = ['rr_last3','wkts_last3','boundaries_last3','dots_last3','score_vol_3','rr_ema',
              'resource_util','over_norm','phase_enc','boundary_dep','batting_aggression',
              'score_par_ratio','balls_since_wicket','partnership_rr',
              'collapse_flag','surge_flag','chase_gradient','dls_resource']
for f in _new_feats:
    _mu = over_raw[f].mean(); _sd = over_raw[f].std()
    print(f"    {f:<26s}: mean={_mu:.3f}  std={_sd:.3f}")


# === CELL 30: BOWLER ARCHETYPE SOM ===========================================
#
# The original framework was batter-centric.  This cell adds the bowling half:
#   • Per-spell bowler performance → rolling 10-match windows
#   • Phase-split economy (powerplay / middle / death) as separate features
#   • 6×6 Bowler Archetype SOM → labels like "Death Specialist", "Strike Bowler"
#   • Career migration visualization for elite bowlers

# ── A. Per-innings / per-spell bowler stats ───────────────────────────────────
bowler_inn = (df[df['valid_ball'] == 1]
              .groupby(['match_id','innings','bowler','bowling_team',
                        'year','bowl_type','match_stage'])
              .agg(balls         =('valid_ball',  'sum'),
                   runs_conceded =('runs_total',  'sum'),
                   wickets       =('is_wicket',   'sum'),
                   dots          =('is_dot',      'sum'),
                   boundaries_c  =('is_boundary', 'sum'),
                   sixes_c       =('is_six',      'sum'))
              .reset_index())

bowler_inn['economy']      = bowler_inn['runs_conceded'] / (bowler_inn['balls']/6).clip(lower=0.1)
bowler_inn['bowling_sr']   = bowler_inn['balls'] / bowler_inn['wickets'].clip(lower=0.1)
bowler_inn['dot_pct']      = bowler_inn['dots'] / bowler_inn['balls'].clip(lower=1) * 100
bowler_inn['boundary_pct_c'] = bowler_inn['boundaries_c'] / bowler_inn['balls'].clip(lower=1) * 100
bowler_inn['wicket_rate']  = bowler_inn['wickets'] / (bowler_inn['balls']/6).clip(lower=0.1)

# Phase-split economy (career averages per phase)
def _phase_eco(phase_label):
    _ph = df[(df['valid_ball']==1) & (df['phase']==phase_label)]
    _g  = _ph.groupby(['match_id','innings','bowler']).agg(
        b=('valid_ball','sum'), r=('runs_total','sum')).reset_index()
    _g['eco'] = _g['r'] / (_g['b']/6).clip(lower=0.1)
    return _g.groupby('bowler')['eco'].mean().to_dict()

_pp_eco    = _phase_eco('Powerplay')
_mid_eco   = _phase_eco('Middle')
_death_eco = _phase_eco('Death')
bowler_inn['pp_economy']    = bowler_inn['bowler'].map(_pp_eco).fillna(bowler_inn['economy'])
bowler_inn['mid_economy']   = bowler_inn['bowler'].map(_mid_eco).fillna(bowler_inn['economy'])
bowler_inn['death_economy'] = bowler_inn['bowler'].map(_death_eco).fillna(bowler_inn['economy'])

print(f"Bowler innings stats: {len(bowler_inn):,} spells")

# ── B. Rolling 10-spell windows for bowlers ───────────────────────────────────
BOWLER_WINDOW   = 10
MIN_BOWL_CAREER = 25

_q_bowlers = (bowler_inn.groupby('bowler').size())
_q_bowlers = _q_bowlers[_q_bowlers >= MIN_BOWL_CAREER].index.tolist()

bowler_windows_list = []
for _bow in _q_bowlers:
    _pd = bowler_inn[bowler_inn['bowler'] == _bow].reset_index(drop=True)
    for _s in range(0, len(_pd) - BOWLER_WINDOW + 1, 3):
        _w = _pd.iloc[_s:_s + BOWLER_WINDOW]
        _tb = _w['balls'].sum(); _tr = _w['runs_conceded'].sum(); _tw = _w['wickets'].sum()
        bowler_windows_list.append({
            'bowler':              _bow,
            'window_idx':          _s // 3,
            'window_start_year':   int(_w['year'].iloc[0]),
            'window_end_year':     int(_w['year'].iloc[-1]),
            'bowl_type':           _w['bowl_type'].mode().iloc[0],
            'economy':             _tr / (_tb/6 + 0.01),
            'wickets_per_spell':   _tw / len(_w),
            'dot_pct':             _w['dot_pct'].mean(),
            'boundary_pct_c':      _w['boundary_pct_c'].mean(),
            'avg_balls_per_spell': _w['balls'].mean(),
            'bowling_sr':          _tb / max(_tw, 1),
            'sixes_per_spell':     _w['sixes_c'].mean(),
            'wicket_rate':         _tw / max(_tb/6, 0.1),
            'death_economy':       _w['death_economy'].mean(),
            'pp_economy':          _w['pp_economy'].mean(),
        })

bowler_windows_df = pd.DataFrame(bowler_windows_list)
print(f"Bowler windows: {len(bowler_windows_df):,} windows for "
      f"{bowler_windows_df['bowler'].nunique()} bowlers")

# ── C. Train 6×6 Bowler Archetype SOM ────────────────────────────────────────
BOWLER_FEATURES = [
    'economy', 'wickets_per_spell', 'dot_pct', 'boundary_pct_c',
    'bowling_sr', 'wicket_rate', 'sixes_per_spell',
    'death_economy', 'pp_economy',
]

X_bow = bowler_windows_df[BOWLER_FEATURES].fillna(0).values
scaler_bowler = StandardScaler()
X_bow_sc = scaler_bowler.fit_transform(X_bow)

BSOM_X, BSOM_Y = 6, 6
som_bowler = MiniSom(BSOM_X, BSOM_Y, X_bow_sc.shape[1],
                     sigma=1.5, learning_rate=0.5, random_seed=42,
                     neighborhood_function='gaussian')
som_bowler.random_weights_init(X_bow_sc)
som_bowler.train_random(X_bow_sc, 5000)

bowler_windows_df['bsom_x']       = [som_bowler.winner(x)[0] for x in X_bow_sc]
bowler_windows_df['bsom_y']       = [som_bowler.winner(x)[1] for x in X_bow_sc]
bowler_windows_df['bowler_node']  = (bowler_windows_df['bsom_x'] * BSOM_Y +
                                     bowler_windows_df['bsom_y'])

print(f"\nBowler SOM trained: {BSOM_X}×{BSOM_Y} = {BSOM_X*BSOM_Y} archetypes")
print(f"  QE: {som_bowler.quantization_error(X_bow_sc):.4f}")

# ── D. Auto-label bowler archetype nodes ─────────────────────────────────────
_bp = bowler_windows_df.groupby('bowler_node')[BOWLER_FEATURES].mean()

def _label_bowler(row):
    if row['economy'] < 6.5 and row['wicket_rate'] > 1.5:
        return 'Elite All-Rounder'
    elif row['wicket_rate'] > 1.8:
        return 'Strike Bowler'
    elif row['economy'] < 6.5 and row['dot_pct'] > 38:
        return 'Economy Specialist'
    elif row['death_economy'] < 8.5 and row['death_economy'] < row['economy']:
        return 'Death Specialist'
    elif row['pp_economy'] < 7.0:
        return 'Powerplay Specialist'
    elif row['dot_pct'] > 42:
        return 'Containment Bowler'
    elif row['economy'] > 9.5:
        return 'Expensive / Pressured'
    else:
        return 'Workhorse / Utility'

_bp['label'] = _bp.apply(_label_bowler, axis=1)
bowler_arch_labels = _bp['label'].to_dict()
bowler_windows_df['bowler_archetype'] = bowler_windows_df['bowler_node'].map(bowler_arch_labels)

print("\nBowler Archetype Distribution:")
print(bowler_windows_df['bowler_archetype'].value_counts().to_string())

# ── E. Bowler SOM heatmaps (vectorized) ──────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(20, 13))
_bfeat_map = {
    'economy':        ('Economy Rate',            'YlOrRd_r'),
    'wicket_rate':    ('Wickets per Over',         'YlGn'),
    'dot_pct':        ('Dot Ball %',               'Blues'),
    'boundary_pct_c': ('Boundaries Conceded %',    'Reds'),
    'death_economy':  ('Death Overs Economy',      'hot_r'),
    'pp_economy':     ('Powerplay Economy',        'Purples_r'),
}
_bsx = bowler_windows_df['bsom_x'].values
_bsy = bowler_windows_df['bsom_y'].values
for ax, (feat, (title, cmap)) in zip(axes.flatten(), _bfeat_map.items()):
    grid = np.zeros((BSOM_X, BSOM_Y)); cnt = np.zeros((BSOM_X, BSOM_Y))
    np.add.at(grid, (_bsx, _bsy), bowler_windows_df[feat].values)
    np.add.at(cnt,  (_bsx, _bsy), 1)
    cnt[cnt == 0] = 1; grid /= cnt
    im = ax.imshow(grid, cmap=cmap, interpolation='nearest')
    ax.set_title(title, fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax, fraction=0.046)
    for nid, lbl in bowler_arch_labels.items():
        _r, _c = nid // BSOM_Y, nid % BSOM_Y
        ax.text(_c, _r, lbl.split('/')[0][:10], fontsize=5.5, ha='center',
                va='center', alpha=0.55, color='black')

plt.suptitle('DIMENSION IV: Bowler Archetype Topological Map (6×6)',
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

# ── F. Career migration for elite bowlers ────────────────────────────────────
_elite_bowlers = ['JJ Bumrah', 'Rashid Khan', 'SP Narine', 'DJ Bravo', 'R Ashwin']

fig, axes = plt.subplots(1, len(_elite_bowlers), figsize=(22, 5))
for ax, _bowler in zip(axes, _elite_bowlers):
    _bd = bowler_windows_df[bowler_windows_df['bowler'] == _bowler].sort_values('window_idx')
    if len(_bd) == 0:
        ax.set_title(f'{_bowler}\n(no data)'); continue

    # Background density
    _g = np.zeros((BSOM_X, BSOM_Y)); _c = np.zeros((BSOM_X, BSOM_Y))
    np.add.at(_g, (bowler_windows_df['bsom_x'].values, bowler_windows_df['bsom_y'].values), 1)
    np.add.at(_c, (bowler_windows_df['bsom_x'].values, bowler_windows_df['bsom_y'].values), 1)
    _c[_c==0]=1; _g/=_c
    ax.imshow(_g, cmap='Greys', alpha=0.2, interpolation='nearest',
              extent=[-0.5, BSOM_Y-0.5, BSOM_X-0.5, -0.5])

    _xs = _bd['bsom_y'].values + np.random.normal(0, 0.05, len(_bd))
    _ys = _bd['bsom_x'].values + np.random.normal(0, 0.05, len(_bd))
    _yrs = _bd['window_end_year'].values
    _pts = np.array([_xs, _ys]).T.reshape(-1, 1, 2)
    _segs = np.concatenate([_pts[:-1], _pts[1:]], axis=1)
    _lc = LineCollection(_segs, cmap='coolwarm',
                         norm=Normalize(_yrs.min(), _yrs.max()), lw=2, alpha=0.8)
    _lc.set_array(_yrs[:-1]); ax.add_collection(_lc)
    ax.scatter(_xs[0], _ys[0], s=120, c='blue', marker='o', zorder=5, edgecolors='k')
    ax.scatter(_xs[-1], _ys[-1], s=120, c='red', marker='*', zorder=5, edgecolors='k')
    ax.set_xlim(-0.5, BSOM_Y-0.5); ax.set_ylim(BSOM_X-0.5, -0.5)
    ax.set_title(f'{_bowler}\n({int(_yrs[0])}-{int(_yrs[-1])})', fontsize=9, fontweight='bold')
    for nid, lbl in bowler_arch_labels.items():
        _r, _c2 = nid // BSOM_Y, nid % BSOM_Y
        ax.text(_c2, _r, lbl[:8], fontsize=5, ha='center', va='center',
                alpha=0.35, color='gray')

plt.suptitle('Bowler Archetype Career Migration\n(Blue=early career → Red=late career)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  PHASE 8 — ADVANCED SOM ARCHITECTURE                                      ║
# ║   • SOM quality diagnostics (QE, TE, Silhouette across grid sizes)        ║
# ║   • 10×10 Game State SOM v2 (28 features → 100 tactical states)          ║
# ║   • 8×8 Player Archetype SOM v2 (richer profile features)                ║
# ╚════════════════════════════════════════════════════════════════════════════╝

# === CELL 31: SOM QUALITY DIAGNOSTICS ========================================
#
# WHY SIZE MATTERS:
#   A 6×6 SOM with 36 nodes for 23,000 training samples = ~640 samples/node.
#   Over-compression blurs tactically different states into one node.
#   Under-compression (too large) fragments the space and destroys topological
#   continuity.  The "elbow" in QE and the peak in Silhouette identify the
#   sweet spot.
#
# METRICS EXPLAINED:
#   Quantization Error (QE):   mean distance from each point to its BMU.
#                              Lower = tighter clusters.  Always decreases
#                              with more nodes; use in ratio to Silhouette.
#   Topographic Error (TE):    fraction of points whose 2nd-best BMU is NOT
#                              adjacent to their best BMU.  Lower = better
#                              topology preservation.  Should be < 0.10.
#   Silhouette Score:          cluster cohesion vs separation [-1, +1].
#                              Peaks at the optimal cluster count.

try:
    from sklearn.metrics import silhouette_score as _sil
    HAS_SIL = True
except ImportError:
    HAS_SIL = False

def som_grid_search(X_sc, sizes, n_iter=5000, label='SOM'):
    """Evaluate SOM quality across grid sizes.  Returns a summary DataFrame."""
    results = []
    n_feat = X_sc.shape[1]
    for sx, sy in sizes:
        _sigma = max(sx, sy) * 0.1 + 0.5
        _som = MiniSom(sx, sy, n_feat, sigma=_sigma,
                       learning_rate=0.5, random_seed=42,
                       neighborhood_function='gaussian')
        _som.random_weights_init(X_sc)
        _som.train_random(X_sc, n_iter)

        qe = _som.quantization_error(X_sc)
        # TE on a 5,000-sample to stay fast
        _idx = np.random.RandomState(42).choice(len(X_sc), min(5000, len(X_sc)), replace=False)
        te = _som.topographic_error(X_sc[_idx])

        _labels = np.array([_som.winner(x)[0]*sy + _som.winner(x)[1]
                            for x in X_sc[_idx]])
        sil = (_sil(X_sc[_idx], _labels, random_state=42)
               if HAS_SIL and len(np.unique(_labels)) > 1 else float('nan'))

        results.append({'SOM': label, 'size': f'{sx}×{sy}', 'nodes': sx*sy,
                        'QE': round(qe, 4), 'TE': round(te, 4),
                        'Silhouette': round(sil, 4)})
        print(f"  {sx:2d}×{sy:2d} ({sx*sy:3d} nodes): QE={qe:.4f}  "
              f"TE={te:.4f}  Sil={sil:.4f}")
    return pd.DataFrame(results)

# Rebuild v1 scaled data for comparison
_X_s1_v1 = inn1[STATE_FEATURES].fillna(0).values  # inn1 is the old slice
_X_s1_v1_sc = scaler_state.transform(_X_s1_v1)

inn1_v2 = over_raw[over_raw['innings'] == 1].copy()
inn2_v2 = over_raw[over_raw['innings'] == 2].copy()

_X_s1_v2 = inn1_v2[ENHANCED_STATE_FEATURES].fillna(0).values
_scaler_v2 = StandardScaler()
_X_s1_v2_sc = _scaler_v2.fit_transform(_X_s1_v2)

GRID_SIZES = [(6, 6), (8, 8), (10, 10), (12, 12)]

print("Game State SOM — grid search (original 9 features):")
gs_v1 = som_grid_search(_X_s1_v1_sc, GRID_SIZES, n_iter=3000, label='GameState-v1')

print("\nGame State SOM — grid search (enhanced 28 features):")
gs_v2 = som_grid_search(_X_s1_v2_sc, GRID_SIZES, n_iter=3000, label='GameState-v2')

# Diagnostic plot
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, metric, title in [
    (axes[0], 'QE',         'Quantization Error\n(lower = tighter clusters)'),
    (axes[1], 'TE',         'Topographic Error\n(lower = better topology)'),
    (axes[2], 'Silhouette', 'Silhouette Score\n(higher = better separation)'),
]:
    for df_g, style, lbl in [(gs_v1, 'b-o', '9-feat v1'), (gs_v2, 'r-s', '28-feat v2')]:
        ax.plot(df_g['nodes'], df_g[metric], style, lw=2, ms=7, label=lbl)
    ax.set_xlabel('Number of SOM nodes'); ax.set_title(title, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

plt.suptitle('PHASE 8: SOM Grid-Size Quality Diagnostics', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

print("\nGrid search summary:")
print(pd.concat([gs_v1, gs_v2]).to_string(index=False))


# === CELL 32: TRAIN ADVANCED SOMs v2 ==========================================
#
# Based on the grid search: 10×10 balances resolution, TE and Silhouette.
# We train with more iterations (10,000) and slightly wider initial sigma
# to encourage global topology learning before local fine-tuning.

SOM_X2, SOM_Y2 = 10, 10

# ── Game State SOM v2 (10×10, 28 features, innings 1) ────────────────────────
scaler_state_v2 = StandardScaler()
X_s1_v2_sc = scaler_state_v2.fit_transform(
    inn1_v2[ENHANCED_STATE_FEATURES].fillna(0).values)

som_state_v2 = MiniSom(SOM_X2, SOM_Y2, X_s1_v2_sc.shape[1],
                        sigma=2.0, learning_rate=0.3, random_seed=42,
                        neighborhood_function='gaussian')
som_state_v2.random_weights_init(X_s1_v2_sc)
som_state_v2.train_random(X_s1_v2_sc, 10000)

inn1_v2['som_x']   = [som_state_v2.winner(x)[0] for x in X_s1_v2_sc]
inn1_v2['som_y']   = [som_state_v2.winner(x)[1] for x in X_s1_v2_sc]
inn1_v2['som_node'] = inn1_v2['som_x'] * SOM_Y2 + inn1_v2['som_y']

_idx5k = np.random.RandomState(42).choice(len(X_s1_v2_sc), 5000, replace=False)
print(f"Game State SOM v2 ({SOM_X2}×{SOM_Y2}, {len(ENHANCED_STATE_FEATURES)} features):")
print(f"  QE = {som_state_v2.quantization_error(X_s1_v2_sc):.4f}  "
      f"TE = {som_state_v2.topographic_error(X_s1_v2_sc[_idx5k]):.4f}")
print(f"  (v1 6×6 QE = {som_state.quantization_error(_X_s1_v1_sc):.4f}  for comparison)")

# ── Auto-label v2 nodes ───────────────────────────────────────────────────────
_node_prof_v2 = inn1_v2.groupby('som_node')[ENHANCED_STATE_FEATURES + ['batting_team_won']].mean()

def _label_node_v2(row):
    parts = []
    if row['collapse_flag'] > 0.4:            parts.append('Collapse')
    elif row['wkts_last3'] > 0.5:             parts.append('Under-Attack')
    if row['over_norm'] < 0.3:                parts.append('Early')
    elif row['over_norm'] < 0.75:             parts.append('Mid')
    else:                                      parts.append('Death')
    if row['surge_flag'] > 0.4:              parts.append('Surge')
    elif row['batting_aggression'] > 0.35:    parts.append('Aggressive')
    elif row['momentum'] < -1.5:              parts.append('Stagnating')
    else:                                      parts.append('Steady')
    if row['balls_since_wicket'] > 20:         parts.append('LongPship')
    return ' | '.join(parts)

_node_prof_v2['label'] = _node_prof_v2.apply(_label_node_v2, axis=1)
node_labels_v2 = _node_prof_v2['label'].to_dict()
inn1_v2['node_label'] = inn1_v2['som_node'].map(node_labels_v2)

print(f"\nSample v2 node profiles (top 15 by visit count):")
_vc = inn1_v2['som_node'].value_counts().head(15)
for nid in _vc.index:
    _r = _node_prof_v2.loc[nid]
    print(f"  Node {nid:3d} [{_r['label']:<40s}] "
          f"RR={_r['cum_run_rate']:.1f} Wkts={_r['cum_wickets']:.1f} "
          f"WinR={_r['batting_team_won']:.2f}")

# ── v2 SOM heatmaps (vectorized) ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(28, 13))
_hm_feats_v2 = {
    'cum_run_rate':      ('Run Rate',              'YlOrRd'),
    'batting_team_won':  ('Win Rate',              'RdYlGn'),
    'momentum':          ('Momentum',              'RdYlGn'),
    'score_vol_3':       ('Score Volatility',      'Purples'),
    'boundary_dep':      ('Boundary Dependency',   'YlGn'),
    'batting_aggression':('Batting Aggression',    'Oranges'),
    'collapse_flag':     ('Collapse Probability',  'Reds'),
    'balls_since_wicket':('Partnership Stability', 'Blues'),
}
_sx2 = inn1_v2['som_x'].values; _sy2 = inn1_v2['som_y'].values
for ax, (feat, (title, cmap)) in zip(axes.flatten(), _hm_feats_v2.items()):
    grid = np.zeros((SOM_X2, SOM_Y2)); cnt = np.zeros((SOM_X2, SOM_Y2))
    np.add.at(grid, (_sx2, _sy2), inn1_v2[feat].fillna(0).values)
    np.add.at(cnt,  (_sx2, _sy2), 1)
    cnt[cnt == 0] = 1; grid /= cnt
    im = ax.imshow(grid, cmap=cmap, interpolation='nearest')
    ax.set_title(title, fontsize=11, fontweight='bold')
    plt.colorbar(im, ax=ax, fraction=0.046)

plt.suptitle('PHASE 8: Game State SOM v2 (10×10, 28 features)\n'
             'Topological maps of enriched cricket state space',
             fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
plt.show()

# ── Player SOM v2 (8×8 with enriched features) ───────────────────────────────
# Add powerplay and death SR to player windows
_pp_sr  = (df[df['phase']=='Powerplay'][df['valid_ball']==1]
           .groupby(['match_id','innings','batter'])
           .agg(r=('runs_batter','sum'), b=('valid_ball','sum')).reset_index())
_pp_sr['pp_sr'] = _pp_sr['r'] / _pp_sr['b'].clip(lower=1) * 100
_pp_sr_map = _pp_sr.groupby('batter')['pp_sr'].mean().to_dict()

_dt_sr  = (df[df['phase']=='Death'][df['valid_ball']==1]
           .groupby(['match_id','innings','batter'])
           .agg(r=('runs_batter','sum'), b=('valid_ball','sum')).reset_index())
_dt_sr['dt_sr'] = _dt_sr['r'] / _dt_sr['b'].clip(lower=1) * 100
_dt_sr_map = _dt_sr.groupby('batter')['dt_sr'].mean().to_dict()

windows_df['pp_sr']   = windows_df['batter'].map(_pp_sr_map).fillna(windows_df['sr'])
windows_df['dt_sr']   = windows_df['batter'].map(_dt_sr_map).fillna(windows_df['sr'])
windows_df['anchor_factor'] = windows_df['avg_balls_faced'] / windows_df['avg'].clip(lower=1)
windows_df['big_inn_pct'] = (batter_inn.groupby('batter')
                              .apply(lambda g: (g['runs'] >= 30).mean())
                              .reindex(windows_df['batter']).values)

PROFILE_FEATURES_V2 = PROFILE_FEATURES + ['pp_sr', 'dt_sr', 'anchor_factor', 'big_inn_pct']

X_pl_v2 = windows_df[PROFILE_FEATURES_V2].fillna(0).values
scaler_player_v2 = StandardScaler()
X_pl_v2_sc = scaler_player_v2.fit_transform(X_pl_v2)

PSOM_X2, PSOM_Y2 = 8, 8
som_player_v2 = MiniSom(PSOM_X2, PSOM_Y2, X_pl_v2_sc.shape[1],
                         sigma=2.0, learning_rate=0.4, random_seed=42,
                         neighborhood_function='gaussian')
som_player_v2.random_weights_init(X_pl_v2_sc)
som_player_v2.train_random(X_pl_v2_sc, 8000)

windows_df['p2_x'] = [som_player_v2.winner(x)[0] for x in X_pl_v2_sc]
windows_df['p2_y'] = [som_player_v2.winner(x)[1] for x in X_pl_v2_sc]
windows_df['arch_node_v2'] = windows_df['p2_x'] * PSOM_Y2 + windows_df['p2_y']

print(f"\nPlayer SOM v2: {PSOM_X2}×{PSOM_Y2} = {PSOM_X2*PSOM_Y2} archetypes  "
      f"QE={som_player_v2.quantization_error(X_pl_v2_sc):.4f}")


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  PHASE 9 — SOM TRANSITION GRAPH & MATCH DYNAMICS                         ║
# ║  Model the game as a Markov chain on the SOM topology:                    ║
# ║   • Directed transition graph (networkx) for both innings                 ║
# ║   • Win/loss-conditioned transition probabilities                         ║
# ║   • Community detection (tactical regions)                                ║
# ║   • PageRank centrality (hub game states)                                 ║
# ║   • Trajectory analytics: velocity, curvature, path entropy               ║
# ║   • Tactical event detection: collapse, surge, choke, recovery            ║
# ╚════════════════════════════════════════════════════════════════════════════╝

# === CELL 33: BUILD SOM TRANSITION GRAPHS =====================================
#
# INSIGHT: The SOM topology + Markov transitions together describe HOW matches
# evolve tactically.  Questions we can now answer:
#   "Which game states almost always precede a collapse?"
#   "What is the winning team's typical path through state-space?"
#   "Which nodes are unavoidable hubs (visited in most innings)?"
#   "What is the transition entropy at over 15?"

try:
    import networkx as nx
    HAS_NX = True
    print("networkx loaded")
except ImportError:
    HAS_NX = False
    print("networkx not available — pip install networkx.  "
          "Transition matrices will still be computed.")

_N = SOM_X2 * SOM_Y2  # total number of nodes

def build_transition_graph(inn_df, n_nodes, som_y,
                            node_col='som_node', over_col='over',
                            match_col='match_id', won_col='batting_team_won'):
    """Build Markov transition matrix and optional networkx directed graph."""
    T      = np.zeros((_N, _N), dtype=np.float64)
    T_win  = np.zeros((_N, _N), dtype=np.float64)
    T_lose = np.zeros((_N, _N), dtype=np.float64)

    for _mid, _grp in inn_df.groupby(match_col):
        _traj = _grp.sort_values(over_col)[node_col].values.astype(int)
        _won  = int(_grp[won_col].iloc[0])
        for _t in range(len(_traj) - 1):
            s, d = _traj[_t], _traj[_t+1]
            if 0 <= s < n_nodes and 0 <= d < n_nodes:
                T[s, d] += 1
                if _won: T_win[s, d] += 1
                else:    T_lose[s, d] += 1

    # Row-normalise to get Markov matrix
    _rs = T.sum(axis=1, keepdims=True).clip(min=1)
    T_prob = T / _rs

    G = None
    if HAS_NX:
        G = nx.DiGraph()
        _node_wr = inn_df.groupby(node_col)[won_col].mean().to_dict()
        _node_vc = inn_df[node_col].value_counts().to_dict()
        for _n in range(n_nodes):
            G.add_node(_n,
                       win_rate=float(_node_wr.get(_n, 0.5)),
                       visit_count=int(_node_vc.get(_n, 0)),
                       grid_row=int(_n) // som_y,
                       grid_col=int(_n) % som_y)
        for s in range(n_nodes):
            for d in range(n_nodes):
                if T[s, d] > 0:
                    G.add_edge(s, d,
                               weight       =int(T[s, d]),
                               prob         =float(T_prob[s, d]),
                               win_weight   =int(T_win[s, d]),
                               lose_weight  =int(T_lose[s, d]),
                               win_cond_prob=float(T_win[s, d] / max(T[s, d], 1)))

    return T_prob, T_win, T_lose, G

# Assign v2 SOM nodes to innings 2
X_s2_v2_sc = scaler_state_v2.transform(
    inn2_v2[ENHANCED_STATE_FEATURES].fillna(0).values)
inn2_v2['som_x']    = [som_state_v2.winner(x)[0] for x in X_s2_v2_sc]
inn2_v2['som_y']    = [som_state_v2.winner(x)[1] for x in X_s2_v2_sc]
inn2_v2['som_node'] = inn2_v2['som_x'] * SOM_Y2 + inn2_v2['som_y']

print("Building innings-1 transition graph (batting first)...")
T1_prob, T1_win, T1_lose, G1 = build_transition_graph(
    inn1_v2, _N, SOM_Y2)

print("Building innings-2 transition graph (chase)...")
T2_prob, T2_win, T2_lose, G2 = build_transition_graph(
    inn2_v2, _N, SOM_Y2)

if HAS_NX:
    print(f"\nInn-1 graph: {G1.number_of_nodes()} nodes, {G1.number_of_edges()} edges")
    print(f"Inn-2 graph: {G2.number_of_nodes()} nodes, {G2.number_of_edges()} edges")


# === CELL 34: GRAPH ANALYSIS & VISUALIZATION ==================================

# ── A. Markov matrix heatmaps ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(24, 7))
_im0 = axes[0].imshow(T1_prob, cmap='YlOrRd', aspect='auto')
axes[0].set_title('Innings 1 — Markov Transition Matrix\n(Row=from, Col=to)',
                  fontweight='bold')
plt.colorbar(_im0, ax=axes[0], label='Transition probability')

# Win-conditional: P(win | traversed edge s→d)
_T1_wp = T1_win / np.maximum(T1_win + T1_lose, 1)
_im1 = axes[1].imshow(_T1_wp, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
axes[1].set_title('Win-Conditional Transitions\n(Green=transition seen in winning innings)',
                  fontweight='bold')
plt.colorbar(_im1, ax=axes[1], label='P(win | transition)')

_im2 = axes[2].imshow(T2_prob, cmap='YlOrRd', aspect='auto')
axes[2].set_title('Innings 2 — Markov Transition Matrix\n(Chase dynamics)',
                  fontweight='bold')
plt.colorbar(_im2, ax=axes[2], label='Transition probability')

for ax in axes:
    ax.set_xlabel('Destination node'); ax.set_ylabel('Source node')

plt.suptitle('PHASE 9: SOM Markov Transition Matrices', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# ── B. Transition entropy (per node) ─────────────────────────────────────────
def _ent(T):
    T = np.where(T > 0, T, 1e-12)
    e = -(T * np.log2(T)).sum(axis=1)
    e[T.sum(axis=1) < 1e-10] = 0
    return e

ent_i1 = _ent(T1_prob)
ent_i2 = _ent(T2_prob)

fig, axes = plt.subplots(1, 3, figsize=(22, 7))
for ax, ent, title in [(axes[0], ent_i1, 'Innings 1 Transition Entropy'),
                        (axes[1], ent_i2, 'Innings 2 Transition Entropy')]:
    grid = ent.reshape(SOM_X2, SOM_Y2)
    im = ax.imshow(grid, cmap='hot', interpolation='nearest')
    ax.set_title(f'{title}\n(Hot=chaotic state, many outgoing paths)',
                 fontweight='bold')
    plt.colorbar(im, ax=ax, label='Shannon entropy (bits)')
    for _i in range(SOM_X2):
        for _j in range(SOM_Y2):
            ax.text(_j, _i, f'{grid[_i,_j]:.1f}', ha='center', va='center',
                    fontsize=5.5,
                    color='white' if grid[_i,_j] > ent.max()*0.55 else 'black')

# Win rate heatmap for v2 SOM
_wr2 = np.zeros((SOM_X2, SOM_Y2)); _cnt2 = np.zeros((SOM_X2, SOM_Y2))
np.add.at(_wr2,  (_sx2, _sy2), inn1_v2['batting_team_won'].values)
np.add.at(_cnt2, (_sx2, _sy2), 1)
_cnt2[_cnt2==0] = 1; _wr2 /= _cnt2
im3 = axes[2].imshow(_wr2, cmap='RdYlGn', interpolation='nearest', vmin=0, vmax=1)
axes[2].set_title('Node Win Rates (v2 SOM, 10×10)\n'
                  'Green=batting-first team wins from here', fontweight='bold')
plt.colorbar(im3, ax=axes[2], label='Win rate')

plt.suptitle('PHASE 9: SOM Topology Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# ── C. NetworkX graph visualization ──────────────────────────────────────────
if HAS_NX:
    fig, axes = plt.subplots(1, 2, figsize=(22, 10))
    for ax, (G, title) in zip(axes,
            [(G1, 'Innings 1 — Batting First'), (G2, 'Innings 2 — Chase')]):
        _pos = {n: (G.nodes[n]['grid_col'], -G.nodes[n]['grid_row'])
                for n in G.nodes()}
        _sizes  = [max(G.nodes[n]['visit_count'] / 8, 15) for n in G.nodes()]
        _colors = [G.nodes[n]['win_rate'] for n in G.nodes()]

        # Show only top-120 edges by weight
        _all_e = sorted(G.edges(data=True), key=lambda e: e[2]['weight'], reverse=True)[:120]
        _ewidths = [e[2]['weight'] / 60 for e in _all_e]
        _ecolors = [e[2]['win_cond_prob'] for e in _all_e]

        nx.draw_networkx_nodes(G, _pos, ax=ax, node_size=_sizes,
                               node_color=_colors, cmap='RdYlGn',
                               vmin=0, vmax=1, alpha=0.85)
        nx.draw_networkx_labels(G, _pos, ax=ax,
                                labels={n: str(n) for n in G.nodes()},
                                font_size=4.5, font_color='black')
        nx.draw_networkx_edges(G, _pos, ax=ax,
                               edgelist=[(e[0],e[1]) for e in _all_e],
                               width=_ewidths, edge_color=_ecolors,
                               edge_cmap=plt.cm.RdYlGn,
                               edge_vmin=0, edge_vmax=1,
                               arrows=True, arrowsize=8, alpha=0.45,
                               connectionstyle='arc3,rad=0.06')
        ax.set_title(f'{title}\n(Node color=win rate, Edge color=P(win|transition), '
                     f'Top-120 edges)', fontsize=11, fontweight='bold')
        ax.axis('off')

    plt.suptitle('PHASE 9: Cricket Dynamics as a Directed Graph\n'
                 '(SOM nodes = game states, edges = over-to-over transitions)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # ── D. Community detection (tactical regions) ─────────────────────────────
    # Communities = groups of SOM nodes that matches tend to STAY IN together.
    # Topologically these are the "innings phases" discovered without hand-coding.
    try:
        _comms_i1 = nx.community.greedy_modularity_communities(
            G1.to_undirected(), weight='weight')
        print(f"\nInnings 1 — {len(_comms_i1)} tactical communities detected:")
        for _ci, _comm in enumerate(sorted(_comms_i1, key=len, reverse=True)):
            _avg_wr   = np.mean([G1.nodes[n]['win_rate'] for n in _comm])
            _avg_vc   = np.mean([G1.nodes[n]['visit_count'] for n in _comm])
            _avg_ent  = np.mean([ent_i1[n] for n in _comm])
            print(f"  Region {_ci+1}: {len(_comm):2d} nodes | "
                  f"Avg win-rate={_avg_wr:.2f} | "
                  f"Avg visits={_avg_vc:.0f} | "
                  f"Avg entropy={_avg_ent:.2f} bits")
            print(f"    Nodes: {sorted(_comm)[:12]}"
                  f"{'...' if len(_comm)>12 else ''}")
    except Exception as _exc:
        print(f"Community detection: {_exc}")

    # ── E. PageRank — "hub" game states ───────────────────────────────────────
    _pr1 = nx.pagerank(G1, weight='weight')
    _pr_grid = np.array([_pr1.get(n, 0) for n in range(_N)]).reshape(SOM_X2, SOM_Y2)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    im_pr = axes[0].imshow(_pr_grid, cmap='plasma', interpolation='nearest')
    axes[0].set_title('PageRank Centrality — Innings 1\n'
                      '(High = frequently-traversed hub game state)',
                      fontweight='bold')
    plt.colorbar(im_pr, ax=axes[0], label='PageRank')
    for _i in range(SOM_X2):
        for _j in range(SOM_Y2):
            axes[0].text(_j, _i, f'{_pr_grid[_i,_j]:.3f}', ha='center',
                         va='center', fontsize=5,
                         color='white' if _pr_grid[_i,_j] > _pr_grid.max()*0.55 else 'black')

    # Winning vs losing entry nodes: where do winning innings FIRST enter?
    _first_nodes_win  = (inn1_v2[inn1_v2['batting_team_won']==1]
                         .sort_values('over').groupby('match_id')['som_node'].first())
    _first_nodes_lose = (inn1_v2[inn1_v2['batting_team_won']==0]
                         .sort_values('over').groupby('match_id')['som_node'].first())

    _fn_win  = np.bincount(_first_nodes_win.values.astype(int),  minlength=_N)
    _fn_lose = np.bincount(_first_nodes_lose.values.astype(int), minlength=_N)
    _fn_diff = (_fn_win / max(_fn_win.sum(),1)) - (_fn_lose / max(_fn_lose.sum(),1))

    im_fd = axes[1].imshow(_fn_diff.reshape(SOM_X2, SOM_Y2),
                           cmap='RdYlGn', interpolation='nearest')
    axes[1].set_title('Starting Node Advantage (Inn 1 Over 0)\n'
                      'Green=winning innings start here more often',
                      fontweight='bold')
    plt.colorbar(im_fd, ax=axes[1], label='Win − Loss density')
    plt.suptitle('PHASE 9: PageRank Centrality & Starting Node Analysis',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


# === CELL 35: TRAJECTORY ANALYTICS & TACTICAL EVENT DETECTION ================
#
# Every innings is a TRAJECTORY through the SOM latent space.
# We quantify that trajectory with physics-inspired metrics:
#
#   velocity        : how fast the match moves between game states per over
#                     (low = locked into one state; high = volatile match)
#   curvature       : rate of velocity change (sharp changes = turning points)
#   path length     : total distance traversed (complexity of the innings)
#   path entropy    : diversity of visited states (0=always same node)
#   max_stability   : longest consecutive run in the same state
#   avg_state_qual  : mean win-rate of visited nodes (does team "stay green"?)

from scipy.stats import mannwhitneyu as _mwu

def compute_trajectory_metrics(inn_df, n_som_y):
    """Return a DataFrame with one row per innings."""
    _node_wr = inn_df.groupby('som_node')['batting_team_won'].mean().to_dict()
    records  = []
    for _mid, _grp in inn_df.groupby('match_id'):
        _traj = _grp.sort_values('over')
        _nd   = _traj['som_node'].values.astype(int)
        if len(_nd) < 3:
            continue
        _r = _nd // n_som_y; _c = _nd % n_som_y
        _dr = np.diff(_r.astype(float)); _dc = np.diff(_c.astype(float))
        _vel = np.sqrt(_dr**2 + _dc**2)
        _curv = float(np.abs(np.diff(_vel)).mean()) if len(_vel) > 1 else 0.0
        _cnt = np.bincount(_nd, minlength=n_som_y**2+1)
        _p   = _cnt[_cnt > 0] / _cnt.sum()
        _ent = float(-((_p) * np.log2(_p + 1e-12)).sum())
        _max_stab = max(
            sum(1 for _ in _g)
            for _, _g in itertools.groupby(_nd))
        records.append({
            'match_id':         _mid,
            'team':             _traj['batting_team'].iloc[0],
            'won':              int(_traj['batting_team_won'].iloc[0]),
            'avg_velocity':     float(_vel.mean()),
            'max_velocity':     float(_vel.max()),
            'velocity_std':     float(_vel.std()),
            'curvature':        _curv,
            'path_length':      float(_vel.sum()),
            'path_entropy':     _ent,
            'max_stability':    _max_stab,
            'node_diversity':   int(len(np.unique(_nd))),
            'avg_state_quality':float(np.mean([_node_wr.get(n, 0.5) for n in _nd])),
        })
    return pd.DataFrame(records)

traj_i1 = compute_trajectory_metrics(inn1_v2, SOM_Y2)
traj_i2 = compute_trajectory_metrics(inn2_v2, SOM_Y2)

print("Innings 1 trajectory metrics — Won vs Lost:")
print(traj_i1.groupby('won')[
    ['avg_velocity','path_length','path_entropy','avg_state_quality','max_stability']
].mean().round(3).to_string())

print("\nInnings 2 trajectory metrics — Won vs Lost:")
print(traj_i2.groupby('won')[
    ['avg_velocity','path_length','path_entropy','avg_state_quality','max_stability']
].mean().round(3).to_string())

# ── Distribution plots with significance tests ────────────────────────────────
_mets = ['avg_velocity','path_length','path_entropy','avg_state_quality',
         'max_stability','node_diversity']
_titl = ['Avg Velocity\n(state change rate per over)',
         'Path Length\n(total SOM distance)',
         'Path Entropy\n(state diversity)',
         'Avg State Quality\n(mean node win-rate)',
         'Max Stability\n(longest run in one state)',
         'Node Diversity\n(unique nodes visited)']

fig, axes = plt.subplots(2, 3, figsize=(20, 11))
for ax, met, tit in zip(axes.flatten(), _mets, _titl):
    _w = traj_i1[traj_i1['won']==1][met].dropna()
    _l = traj_i1[traj_i1['won']==0][met].dropna()
    ax.hist(_w, bins=30, alpha=0.6, color='#2ecc71', label='Won',  density=True)
    ax.hist(_l, bins=30, alpha=0.6, color='#e74c3c', label='Lost', density=True)
    ax.axvline(_w.mean(), color='darkgreen', lw=2, ls='--')
    ax.axvline(_l.mean(), color='darkred',   lw=2, ls='--')
    _, _p = _mwu(_w, _l, alternative='two-sided')
    _sig = '***' if _p < 0.001 else ('**' if _p < 0.01 else ('*' if _p < 0.05 else 'ns'))
    ax.text(0.97, 0.95, f'p={_p:.3f} {_sig}', transform=ax.transAxes,
            ha='right', va='top', fontsize=9,
            color='darkred' if _p < 0.05 else 'gray', fontweight='bold')
    ax.set_title(tit, fontweight='bold', fontsize=10)
    ax.legend(fontsize=8)

plt.suptitle('PHASE 9: Innings Trajectory Metrics — Won vs Lost\n'
             '(Mann-Whitney U test; *** p<0.001, ** p<0.01, * p<0.05, ns=not significant)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()

# ── Tactical event detection ──────────────────────────────────────────────────
_node_wr_v2 = inn1_v2.groupby('som_node')['batting_team_won'].mean().to_dict()

def detect_events(inn_df, node_wr_dict, threshold=0.12):
    """Detect momentum-altering over transitions (collapse or surge)."""
    _rows = []
    for _mid, _grp in inn_df.groupby('match_id'):
        _traj = _grp.sort_values('over')
        _nd   = _traj['som_node'].values.astype(int)
        _ov   = _traj['over'].values
        _won  = int(_traj['batting_team_won'].iloc[0])
        _team = _traj['batting_team'].iloc[0]
        for _t in range(1, len(_nd)):
            _delta = node_wr_dict.get(_nd[_t], 0.5) - node_wr_dict.get(_nd[_t-1], 0.5)
            if abs(_delta) >= threshold:
                _rows.append({
                    'match_id': _mid, 'over': int(_ov[_t]),
                    'event':    'SURGE'    if _delta > 0 else 'COLLAPSE',
                    'from_node': _nd[_t-1], 'to_node': _nd[_t],
                    'wr_delta': round(_delta, 3),
                    'won': _won, 'team': _team,
                })
    return pd.DataFrame(_rows)

events_i1 = detect_events(inn1_v2, _node_wr_v2)
events_i2 = detect_events(inn2_v2,
                           inn2_v2.groupby('som_node')['batting_team_won'].mean().to_dict())

print(f"\nTactical events detected:")
print(f"  Inn 1 — Collapses: {(events_i1['event']=='COLLAPSE').sum():,}  "
      f"Surges: {(events_i1['event']=='SURGE').sum():,}")
print(f"  Inn 2 — Collapses: {(events_i2['event']=='COLLAPSE').sum():,}  "
      f"Surges: {(events_i2['event']=='SURGE').sum():,}")

# Event-frequency-by-over chart
fig, axes = plt.subplots(2, 2, figsize=(20, 11))
for row_i, (ev_df, inn_label) in enumerate([(events_i1, 'Innings 1 (Bat First)'),
                                             (events_i2, 'Innings 2 (Chase)')]):
    for col_i, (ev_type, color, label) in enumerate([
        ('COLLAPSE', ('#e74c3c', '#c0392b'), 'Collapse'),
        ('SURGE',    ('#2ecc71', '#1a8a50'), 'Surge'),
    ]):
        ax = axes[row_i][col_i]
        _ev = ev_df[ev_df['event'] == ev_type]
        _w_ov = _ev[_ev['won']==1]['over'].value_counts().reindex(range(20), fill_value=0)
        _l_ov = _ev[_ev['won']==0]['over'].value_counts().reindex(range(20), fill_value=0)
        _x = np.arange(20)
        ax.bar(_x - 0.2, _w_ov, 0.4, color=color[0], alpha=0.75, label='In won innings')
        ax.bar(_x + 0.2, _l_ov, 0.4, color=color[1], alpha=0.55, label='In lost innings')
        ax.set_xticks(_x); ax.set_xticklabels(range(20), fontsize=8)
        ax.set_xlabel('Over'); ax.set_ylabel('Event count')
        ax.set_title(f'{inn_label} — {label} Events by Over', fontweight='bold', fontsize=11)
        ax.legend(fontsize=9)
        for (s, e, c) in [(0,5,'#3498db'),(6,14,'#27ae60'),(15,19,'#e74c3c')]:
            ax.axvspan(s-0.5, e+0.5, alpha=0.04, color=c)
        ax.text(2.5,  ax.get_ylim()[1]*0.92, 'PP',    color='#3498db', fontsize=8, fontweight='bold')
        ax.text(10,   ax.get_ylim()[1]*0.92, 'Middle',color='#27ae60', fontsize=8, fontweight='bold')
        ax.text(17.5, ax.get_ylim()[1]*0.92, 'Death', color='#e74c3c', fontsize=8, fontweight='bold')

plt.suptitle('PHASE 9: Tactical Event Detection via SOM State Transitions\n'
             '(Events = overs where win-probability node jumps ≥ 12 pp)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# ── Most "decisive" overs (avg |wr_delta| per over) ──────────────────────────
_dec_i1 = (events_i1.groupby('over')['wr_delta']
           .agg(lambda x: x.abs().mean()).reset_index(name='avg_impact'))
_dec_i2 = (events_i2.groupby('over')['wr_delta']
           .agg(lambda x: x.abs().mean()).reset_index(name='avg_impact'))

fig, axes = plt.subplots(1, 2, figsize=(18, 5))
for ax, _d, title in [(axes[0], _dec_i1, 'Innings 1 — Over Impact'),
                       (axes[1], _dec_i2, 'Innings 2 — Over Impact')]:
    _bars = ax.bar(_d['over'], _d['avg_impact'],
                   color=[plt.cm.RdYlGn(0.8 - v/0.3*0.8) for v in _d['avg_impact']])
    ax.set_xlabel('Over'); ax.set_ylabel('Avg |win-probability shift|')
    ax.set_title(f'{title}\n(Which overs produce the biggest state transitions?)',
                 fontweight='bold', fontsize=11)
    for (s, e, c) in [(0,5,'#3498db'),(6,14,'#27ae60'),(15,19,'#e74c3c')]:
        ax.axvspan(s-0.5, e+0.5, alpha=0.06, color=c)

plt.suptitle('PHASE 9: Over-Level Match Impact (average WP shift via SOM transitions)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()


# === CELL 36: SAVE ARTIFACTS FOR STREAMLIT DASHBOARD =========================
import pickle as _pkl
import os as _os

# ── Pre-compute win probabilities for every over (for match replay) ───────────
_inn1_rp = inn1.copy()
_inn1_rp['node_label'] = _inn1_rp['som_node'].map(node_labels)
_inn1_rp['win_prob'] = win_model_inn1.predict_proba(
    _inn1_rp[INN1_WIN_FEATURES].fillna(0))[:, 1]

_inn2_rp = inn2.copy()
_inn2_rp['node_label'] = _inn2_rp['som_node'].map(node_labels)
_inn2_rp['win_prob'] = win_model_inn2.predict_proba(
    _inn2_rp[INN2_WIN_FEATURES].fillna(0))[:, 1]

_rp_cols = ['match_id', 'innings', 'over', 'batting_team', 'bowling_team',
            'year', 'venue', 'cum_runs', 'cum_wickets', 'batting_team_won',
            'som_node', 'node_label', 'win_prob']
_replay_df = pd.concat([
    _inn1_rp[[c for c in _rp_cols if c in _inn1_rp.columns]],
    _inn2_rp[[c for c in _rp_cols if c in _inn2_rp.columns]],
]).reset_index(drop=True)

# ── Entropy for v1 SOM (build quick Markov from inn1) ─────────────────────────
_N1 = SOM_X * SOM_Y
_T1v1 = np.zeros((_N1, _N1))
for _mid1, _grp1 in inn1.groupby('match_id'):
    _nd1 = _grp1.sort_values('over')['som_node'].values.astype(int)
    for _t1 in range(1, len(_nd1)):
        if _nd1[_t1 - 1] < _N1 and _nd1[_t1] < _N1:
            _T1v1[_nd1[_t1 - 1], _nd1[_t1]] += 1
_row_sum1 = _T1v1.sum(axis=1, keepdims=True)
_row_sum1[_row_sum1 == 0] = 1
_T1v1 /= _row_sum1
_T1v1_safe = np.where(_T1v1 > 0, _T1v1, 1e-12)
_ent_v1 = -(_T1v1_safe * np.log2(_T1v1_safe)).sum(axis=1)
_ent_v1[_T1v1.sum(axis=1) < 1e-10] = 0

# ── Bundle all artifacts ──────────────────────────────────────────────────────
_artifacts = {
    # v1 SOM (6×6, 9 features — used by Phase 6 win/score models)
    'som_state':        som_state,
    'scaler_state':     scaler_state,
    'node_labels':      node_labels,
    'node_wr_map':      node_wr_map,
    'STATE_FEATURES':   STATE_FEATURES,
    'SOM_X': SOM_X, 'SOM_Y': SOM_Y,
    'ent_v1':           _ent_v1,
    'inn1_nodes':       inn1[['match_id', 'over', 'som_node', 'batting_team_won']].copy(),

    # v2 SOM (10×10, 28 features — Phases 7–9)
    'som_state_v2':          som_state_v2,
    'scaler_state_v2':       scaler_state_v2,
    'node_labels_v2':        node_labels_v2,
    'node_wr_v2':            _node_wr_v2,
    'ENHANCED_STATE_FEATURES': ENHANCED_STATE_FEATURES,
    'SOM_X2': SOM_X2, 'SOM_Y2': SOM_Y2,
    'ent_i1':           ent_i1,
    'ent_i2':           ent_i2,
    'inn1_v2_compact':  inn1_v2[['match_id', 'innings', 'over', 'batting_team',
                                  'bowling_team', 'year', 'venue', 'cum_runs',
                                  'cum_wickets', 'batting_team_won',
                                  'som_node', 'node_label']].copy(),

    # Prediction models (Phase 6)
    'win_model_inn1':   win_model_inn1,
    'win_model_inn2':   win_model_inn2,
    'score_model':      score_model,
    'INN1_WIN_FEATURES': INN1_WIN_FEATURES,
    'INN2_WIN_FEATURES': INN2_WIN_FEATURES,
    'SCORE_FEAT':       SCORE_FEAT,

    # Player batter data (Phases 3–4)
    'windows_df':        windows_df[['batter', 'window_idx', 'window_start_year',
                                     'window_end_year'] + PROFILE_FEATURES +
                                    ['archetype']].copy(),
    'clutch_df':         clutch_df.copy(),
    'archetype_labels':  archetype_labels,
    'PROFILE_FEATURES':  PROFILE_FEATURES,
    'PSOM_X': PSOM_X, 'PSOM_Y': PSOM_Y,

    # Bowler data (Phase 7)
    'bowler_windows_df': bowler_windows_df[['bowler', 'window_idx', 'window_start_year',
                                             'window_end_year', 'bowl_type'] +
                                            BOWLER_FEATURES +
                                            ['bowler_archetype']].copy(),
    'bowler_arch_labels': bowler_arch_labels,
    'BOWLER_FEATURES':   BOWLER_FEATURES,
    'BSOM_X': BSOM_X, 'BSOM_Y': BSOM_Y,

    # Match replay (pre-computed win probabilities)
    'replay_df':         _replay_df,

    # Graphs & Markov (Phase 9)
    'T1_prob':           T1_prob,
    'T2_prob':           T2_prob,

    # Trajectory & event data
    'traj_i1':           traj_i1,
    'traj_i2':           traj_i2,
    'events_i1':         events_i1,
    'events_i2':         events_i2,

    # Venue stats
    'venue_stats':       venue_stats.copy(),

    # Metadata
    'n_matches': int(over_raw['match_id'].nunique()),
    'n_balls':   len(df),
}

_os.makedirs('artifacts', exist_ok=True)
_out = 'artifacts/som_artifacts.pkl'
with open(_out, 'wb') as _f:
    _pkl.dump(_artifacts, _f, protocol=4)
print(f"\nArtifacts saved → {_out}  ({_os.path.getsize(_out)/1e6:.1f} MB)")
print(f"  Keys ({len(_artifacts)}): {list(_artifacts.keys())}")
