"""
IPL SOM Analytics Dashboard
Interactive frontend for the Multi-Dimensional Unsupervised Learning Framework
Run with:  streamlit run streamlit_app.py
"""
import os
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

# Transformer (optional — graceful degradation if not trained yet)
try:
    import torch
    from transformer_model import (
        NextBallTransformer, BOS_TOKEN, NUM_FEATURES,
        state_to_features, OUTCOME_LABELS, OUTCOME_COLORS,
    )
    _TORCH_OK = True
except Exception:
    _TORCH_OK = False

st.set_page_config(
    page_title="IPL SOM Analytics",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)

ARTIFACT_PATH = os.path.join(os.path.dirname(__file__), "artifacts", "som_artifacts.pkl")

ARCHETYPE_COLORS = {
    "Elite Aggressor":            "#f39c12",
    "Anchor / Accumulator":       "#3498db",
    "Power Finisher":             "#9b59b6",
    "Aggressive Striker":         "#e74c3c",
    "Role Player":                "#2ecc71",
    "Reliable Run-Scorer":        "#1abc9c",
    "Defensive / Under Pressure": "#95a5a6",
    "Struggling / Out of Form":   "#7f8c8d",
}

BOWLER_ARCHETYPE_COLORS = {
    "Elite All-Rounder":    "#f39c12",
    "Strike Bowler":        "#e74c3c",
    "Economy Specialist":   "#3498db",
    "Death Specialist":     "#9b59b6",
    "Powerplay Specialist": "#1abc9c",
    "Containment Bowler":   "#2ecc71",
    "Expensive / Pressured":"#e67e22",
    "Workhorse / Utility":  "#95a5a6",
}


@st.cache_resource
def load_artifacts():
    with open(ARTIFACT_PATH, "rb") as f:
        return pickle.load(f)


# ─── Shared helpers ───────────────────────────────────────────────────────────

def _win_gauge(value, title, height=300):
    color = "#2ecc71" if value > 0.5 else "#e74c3c"
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value * 100,
        title={"text": title, "font": {"size": 15}},
        delta={"reference": 50, "valueformat": ".1f"},
        gauge={
            "axis": {"range": [0, 100], "ticksuffix": "%"},
            "bar":  {"color": color, "thickness": 0.35},
            "steps": [
                {"range": [0, 35],  "color": "#fadbd8"},
                {"range": [35, 65], "color": "#fef9e7"},
                {"range": [65, 100],"color": "#d5f5e3"},
            ],
            "threshold": {"line": {"color": "black", "width": 3}, "value": 50},
        },
        number={"suffix": "%", "valueformat": ".1f", "font": {"size": 34}},
    ))
    fig.update_layout(height=height, margin=dict(t=60, b=10, l=20, r=20))
    return fig


def _som_heatmap(data_grid, text_grid=None, title="",
                 colorscale="RdYlGn", zmin=0, zmax=1, height=480):
    fig = go.Figure(go.Heatmap(
        z=data_grid,
        colorscale=colorscale,
        zmin=zmin, zmax=zmax,
        text=text_grid,
        texttemplate="%{text}",
        textfont={"size": 7},
        hovertemplate="Node row %{y}  col %{x}<br>Value: %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title={"text": title, "font": {"size": 14}},
        height=height,
        xaxis={"title": "SOM column", "side": "top"},
        yaxis={"title": "SOM row", "autorange": "reversed"},
        margin=dict(t=80, b=20, l=45, r=20),
    )
    return fig


# ─── Page 1: Live Match Prediction ───────────────────────────────────────────

def page_live(arts):
    st.title("Live Match Prediction")
    st.caption("Enter the current game state to get win probability and score forecast.")

    left, right = st.columns([1, 1.6], gap="large")

    with left:
        st.subheader("Game State")
        innings = st.radio("Innings", [1, 2], horizontal=True)
        over = st.slider("Overs completed", 0, 19, 6)
        cr, cw = st.columns(2)
        cum_runs    = cr.number_input("Runs scored",    0, 400, 52)
        cum_wickets = cw.number_input("Wickets fallen", 0, 9,   0)

        target = None
        if innings == 2:
            target = st.number_input("Target (runs)", 50, 400, 162)

        with st.expander("This-over details (optional)", expanded=True):
            ea, eb = st.columns(2)
            runs_this_over    = ea.number_input("Runs this over",    0, 36, 8)
            wickets_this_over = ea.number_input("Wickets this over", 0, 4,  0)
            boundaries        = eb.number_input("Boundaries",        0, 6,  1)
            dots              = eb.number_input("Dot balls",         0, 6,  2)
            sixes             = eb.number_input("Sixes",             0, 6,  0)

    # ── Derived quantities ───────────────────────────────────────────────────
    balls_done      = over * 6
    balls_remaining = 120 - balls_done
    cum_rr          = cum_runs / max(over, 0.5)
    over_rr         = float(runs_this_over)
    momentum        = over_rr - cum_rr
    wkts_remaining  = 10 - cum_wickets

    # Resolve v1 SOM node
    sv    = np.array([[cum_rr, over_rr, momentum, wickets_this_over,
                       wkts_remaining, dots, boundaries, cum_wickets, over]])
    sv_sc = arts["scaler_state"].transform(sv)
    wx, wy = arts["som_state"].winner(sv_sc[0])
    snode       = wx * arts["SOM_Y"] + wy
    node_label  = arts["node_labels"].get(snode, "Unknown")
    node_wr     = arts["node_wr_map"].get(snode, 0.5)

    with right:
        st.subheader("Prediction")

        if innings == 1:
            feat  = np.array([[over, cum_rr, cum_wickets, momentum, wickets_this_over,
                               boundaries, dots, wkts_remaining, snode]])
            sfeat = np.array([[over, cum_rr, cum_wickets, momentum, wkts_remaining,
                               boundaries, dots, sixes, snode]])
            wp = arts["win_model_inn1"].predict_proba(feat)[0, 1]
            sp = arts["score_model"].predict(sfeat)[0]

            st.plotly_chart(_win_gauge(wp, "Batting-First Win Probability"),
                            use_container_width=True)

            m1, m2, m3 = st.columns(3)
            m1.metric("Score Forecast", f"{sp:.0f} runs",
                      f"Range {sp*0.92:.0f}–{sp*1.08:.0f}")
            m2.metric("SOM State", f"#{snode}", node_label[:28])
            m3.metric("Node Win Rate", f"{node_wr:.1%}",
                      f"{node_wr - 0.5:+.1%} vs baseline")

            st.info(
                f"Simple pace projection: **{cum_rr * 20:.0f} runs**  ·  "
                f"SOM-adjusted forecast: **{sp:.0f} runs**  ·  "
                f"Balls remaining: **{balls_remaining}**"
            )

        else:
            if target is None:
                st.warning("Enter a target score for innings 2.")
                return
            runs_req    = max(int(target) - int(cum_runs), 0)
            req_rr      = runs_req / max(balls_remaining / 6, 0.1)
            press_ratio = req_rr / max(cum_rr, 0.1)

            feat = np.array([[over, cum_rr, cum_wickets, momentum, wkts_remaining,
                              req_rr, press_ratio, runs_req, snode]])
            wp_chase = arts["win_model_inn2"].predict_proba(feat)[0, 1]

            fig2 = make_subplots(
                rows=1, cols=2,
                specs=[[{"type": "indicator"}, {"type": "indicator"}]],
            )
            for col_i, (val, ttl, col) in enumerate([
                (wp_chase,     "Chaser Win %",   "#3498db"),
                (1 - wp_chase, "Defender Win %", "#e74c3c"),
            ], start=1):
                fig2.add_trace(go.Indicator(
                    mode="gauge+number",
                    value=val * 100,
                    title={"text": ttl, "font": {"size": 15}},
                    gauge={"axis": {"range": [0, 100]},
                           "bar": {"color": col},
                           "threshold": {"line": {"color": "black", "width": 2}, "value": 50}},
                    number={"suffix": "%", "valueformat": ".1f"},
                ), row=1, col=col_i)
            fig2.update_layout(height=300, margin=dict(t=40, b=10))
            st.plotly_chart(fig2, use_container_width=True)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Need", f"{runs_req} off {balls_remaining}", f"RRR {req_rr:.2f}")
            m2.metric("Pressure", f"{press_ratio:.2f}×", None)
            m3.metric("SOM State", f"#{snode}", node_label[:24])
            m4.metric("Node WR", f"{node_wr:.1%}", None)


# ─── Page 2: Player Explorer ──────────────────────────────────────────────────

def page_players(arts):
    st.title("Player Explorer")

    tab_bat, tab_bowl = st.tabs(["Batters", "Bowlers"])

    # ── Batter tab ─────────────────────────────────────────────────────────
    with tab_bat:
        windows = arts["windows_df"]
        all_batters = sorted(windows["batter"].unique())

        col_a, col_b = st.columns([1, 1])
        default_p1 = all_batters.index("V Kohli") if "V Kohli" in all_batters else 0
        player1 = col_a.selectbox("Select player", all_batters, index=default_p1)
        enable_cmp = col_b.checkbox("Compare with another player")
        default_p2 = all_batters.index("MS Dhoni") if "MS Dhoni" in all_batters else 1
        player2 = col_b.selectbox("Second player", all_batters, index=default_p2,
                                   disabled=not enable_cmp)

        show_players = [player1] + ([player2] if enable_cmp else [])

        # Career average line chart
        fig_avg = go.Figure()
        for pname in show_players:
            pdata = windows[windows["batter"] == pname].sort_values("window_idx")
            if pdata.empty:
                continue
            colors_p = [ARCHETYPE_COLORS.get(a, "#999") for a in pdata["archetype"]]
            hover_txt = [
                f"{r['batter']}<br>{r['archetype']}<br>"
                f"Avg {r['avg']:.1f} · SR {r['sr']:.0f}<br>"
                f"{r['window_start_year']}–{r['window_end_year']}"
                for _, r in pdata.iterrows()
            ]
            fig_avg.add_trace(go.Scatter(
                x=list(range(len(pdata))),
                y=pdata["avg"],
                mode="lines+markers",
                name=pname,
                text=hover_txt,
                hovertemplate="%{text}<extra></extra>",
                marker=dict(color=colors_p, size=9, line=dict(color="white", width=0.8)),
                line=dict(width=2),
            ))
        fig_avg.update_layout(
            title="Career Batting Average — Rolling 15-Match Windows (coloured by archetype)",
            xaxis_title="Window (chronological)",
            yaxis_title="Batting Average",
            height=340,
            hovermode="closest",
            legend=dict(orientation="h", y=1.02),
        )
        st.plotly_chart(fig_avg, use_container_width=True)

        # Strike rate chart
        fig_sr = go.Figure()
        for pname in show_players:
            pdata = windows[windows["batter"] == pname].sort_values("window_idx")
            if pdata.empty:
                continue
            fig_sr.add_trace(go.Scatter(
                x=list(range(len(pdata))),
                y=pdata["sr"],
                mode="lines+markers",
                name=pname,
                marker=dict(size=6),
                line=dict(width=1.5, dash="dot"),
            ))
        fig_sr.update_layout(
            title="Strike Rate over Career",
            xaxis_title="Window", yaxis_title="Strike Rate",
            height=250, hovermode="closest",
            legend=dict(orientation="h", y=1.02),
        )
        st.plotly_chart(fig_sr, use_container_width=True)

        # Archetype distribution
        pdata1 = windows[windows["batter"] == player1].sort_values("window_idx")
        if not pdata1.empty:
            arch_counts = (pdata1["archetype"]
                           .value_counts()
                           .reset_index()
                           .rename(columns={"archetype": "Archetype", "count": "Windows"}))
            fig_arch = px.bar(
                arch_counts, x="Windows", y="Archetype", orientation="h",
                color="Archetype", color_discrete_map=ARCHETYPE_COLORS,
                title=f"{player1} — Career Archetype Distribution",
                height=300,
            )
            fig_arch.update_layout(showlegend=False)
            st.plotly_chart(fig_arch, use_container_width=True)

        # Clutch stats
        st.subheader("Clutch Analysis")
        clutch = arts["clutch_df"]
        p1_clutch = clutch[clutch["batter"] == player1]
        if not p1_clutch.empty:
            row = p1_clutch.iloc[0]
            verdict = ("Clutch Player 🔥" if row["clutch_ratio"] > 1.1
                       else ("Crumbles Under Pressure ❄️" if row["clutch_ratio"] < 0.9
                             else "Consistent ✅"))
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Normal SR",    f"{row['normal_sr']:.1f}")
            mc2.metric("Pressure SR",  f"{row['hp_sr']:.1f}", f"{row['sr_diff']:+.1f}")
            mc3.metric("Clutch Ratio", f"{row['clutch_ratio']:.3f}", verdict)
            mc4.metric("HP Balls",     f"{int(row['hp_balls'])}")
        else:
            st.info(f"{player1} — insufficient high-pressure balls (< 30) in the dataset.")

        with st.expander("Top 20 Clutch Players"):
            top20 = clutch.nlargest(20, "clutch_ratio")[
                ["batter", "normal_sr", "hp_sr", "clutch_ratio", "sr_diff", "hp_balls"]
            ]
            fig_c = px.bar(
                top20, x="clutch_ratio", y="batter", orientation="h",
                color="clutch_ratio", color_continuous_scale="RdYlGn",
                title="Clutch Ratio — Pressure SR / Normal SR",
                text=top20["clutch_ratio"].apply(lambda v: f"{v:.2f}"),
                height=520,
            )
            fig_c.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig_c, use_container_width=True)

        with st.expander("Bottom 15 — Pressure Crumblers"):
            bot15 = clutch.nsmallest(15, "clutch_ratio")[
                ["batter", "normal_sr", "hp_sr", "clutch_ratio", "sr_diff", "hp_balls"]
            ]
            fig_cb = px.bar(
                bot15, x="clutch_ratio", y="batter", orientation="h",
                color="clutch_ratio", color_continuous_scale="RdYlGn",
                title="Pressure Crumblers",
                text=bot15["clutch_ratio"].apply(lambda v: f"{v:.2f}"),
                height=420,
            )
            fig_cb.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig_cb, use_container_width=True)

    # ── Bowler tab ─────────────────────────────────────────────────────────
    with tab_bowl:
        bwindows = arts["bowler_windows_df"]
        all_bowlers = sorted(bwindows["bowler"].unique())
        default_bow = all_bowlers.index("JJ Bumrah") if "JJ Bumrah" in all_bowlers else 0
        bowl_sel = st.selectbox("Select bowler", all_bowlers, index=default_bow)

        bdata = bwindows[bwindows["bowler"] == bowl_sel].sort_values("window_idx")
        if bdata.empty:
            st.warning("No data for this bowler.")
        else:
            arch_cols_b = [BOWLER_ARCHETYPE_COLORS.get(a, "#999")
                           for a in bdata["bowler_archetype"]]
            hover_b = [
                f"{r['bowler']}<br>{r['bowler_archetype']}<br>"
                f"Eco {r['economy']:.2f} · WPO {r['wicket_rate']:.2f} · Dot {r['dot_pct']:.0f}%<br>"
                f"{r['window_start_year']}–{r['window_end_year']}"
                for _, r in bdata.iterrows()
            ]

            fig_eco = go.Figure()
            fig_eco.add_trace(go.Scatter(
                x=list(range(len(bdata))),
                y=bdata["economy"],
                mode="lines+markers",
                name="Economy",
                text=hover_b,
                hovertemplate="%{text}<extra></extra>",
                marker=dict(color=arch_cols_b, size=9, line=dict(color="white", width=0.8)),
                line=dict(width=2, color="#3498db"),
            ))
            fig_eco.add_trace(go.Scatter(
                x=list(range(len(bdata))),
                y=bdata["wicket_rate"],
                mode="lines+markers",
                name="Wickets/over",
                yaxis="y2",
                marker=dict(color="#e74c3c", size=6),
                line=dict(width=1.5, dash="dot", color="#e74c3c"),
            ))
            fig_eco.update_layout(
                title=f"{bowl_sel} — Economy & Wicket Rate over Career (coloured by archetype)",
                xaxis_title="Window (chronological)",
                yaxis_title="Economy rate",
                yaxis2=dict(title="Wickets / over", overlaying="y", side="right",
                            range=[0, 4], showgrid=False),
                height=340,
                hovermode="closest",
                legend=dict(orientation="h", y=1.02),
            )
            st.plotly_chart(fig_eco, use_container_width=True)

            last = bdata.iloc[-1]
            bc1, bc2, bc3, bc4, bc5 = st.columns(5)
            bc1.metric("Current Archetype", last["bowler_archetype"])
            bc2.metric("Economy",           f"{last['economy']:.2f}")
            bc3.metric("Wickets/over",      f"{last['wicket_rate']:.2f}")
            bc4.metric("Dot Ball %",        f"{last['dot_pct']:.1f}%")
            bc5.metric("Death Economy",     f"{last['death_economy']:.2f}")

            with st.expander("Phase-split economy (career averages)"):
                phase_df = pd.DataFrame({
                    "Phase":   ["Powerplay", "Death"],
                    "Economy": [last["pp_economy"], last["death_economy"]],
                    "Overall": [last["economy"], last["economy"]],
                })
                fig_ph = px.bar(phase_df, x="Phase", y=["Economy", "Overall"],
                                barmode="group", title="Economy by Phase",
                                color_discrete_sequence=["#3498db", "#bdc3c7"], height=280)
                st.plotly_chart(fig_ph, use_container_width=True)


# ─── Page 3: Match Replay ─────────────────────────────────────────────────────

def page_replay(arts):
    st.title("Match Replay")
    st.caption("Step through any historical IPL match — win probability and SOM trajectory.")

    replay = arts["replay_df"]
    years  = sorted(replay["year"].unique(), reverse=True)

    fc1, fc2, fc3 = st.columns(3)
    year_sel = fc1.selectbox("Year", years, index=0)
    yr_data  = replay[replay["year"] == year_sel]
    teams    = sorted(yr_data["batting_team"].unique())
    team_sel = fc2.selectbox("Batting team", ["All"] + teams, index=0)
    if team_sel != "All":
        yr_data = yr_data[yr_data["batting_team"] == team_sel]

    inn1_matches = yr_data[yr_data["innings"] == 1].drop_duplicates("match_id").copy()
    inn1_matches["label"] = (inn1_matches["batting_team"] + " vs " +
                              inn1_matches["bowling_team"])
    options = {r["label"]: r["match_id"] for _, r in inn1_matches.iterrows()}

    if not options:
        st.warning("No matches found — adjust filters.")
        return

    match_label = fc3.selectbox("Match", list(options.keys()), index=0)
    mid = options[match_label]

    m1 = replay[(replay["match_id"] == mid) & (replay["innings"] == 1)].sort_values("over")
    m2 = replay[(replay["match_id"] == mid) & (replay["innings"] == 2)].sort_values("over")

    if m1.empty:
        st.warning("Match data unavailable.")
        return

    won_by = ("Batting-first team won" if m1["batting_team_won"].iloc[-1] == 1
              else "Chasing team won")
    st.subheader(f"{match_label}  ·  {won_by}")

    # Win probability timeline (both innings on same chart, x=over 0-39)
    fig_wp = go.Figure()
    bat_team  = m1["batting_team"].iloc[0]
    bowl_team = m1["bowling_team"].iloc[0]

    fig_wp.add_trace(go.Scatter(
        x=m1["over"], y=m1["win_prob"] * 100,
        mode="lines+markers", name=f"{bat_team} (Inn 1)",
        line=dict(color="#2ecc71", width=2.5), marker=dict(size=7),
        text=[f"Over {r['over']}: {r['cum_runs']}/{r['cum_wickets']}  WP {r['win_prob']:.1%}"
              for _, r in m1.iterrows()],
        hovertemplate="%{text}<extra></extra>",
    ))

    if not m2.empty:
        chase_team = m2["batting_team"].iloc[0]
        fig_wp.add_trace(go.Scatter(
            x=m2["over"] + 20, y=m2["win_prob"] * 100,
            mode="lines+markers", name=f"{chase_team} (Inn 2)",
            line=dict(color="#3498db", width=2.5), marker=dict(size=7),
            text=[f"Over {r['over']} (Inn2): {r['cum_runs']}/{r['cum_wickets']}  WP {r['win_prob']:.1%}"
                  for _, r in m2.iterrows()],
            hovertemplate="%{text}<extra></extra>",
        ))

    fig_wp.add_hline(y=50, line_dash="dash", line_color="gray", opacity=0.5,
                     annotation_text="50%")
    fig_wp.add_vline(x=19.5, line_dash="dot", line_color="orange", opacity=0.7,
                     annotation_text="Inns break", annotation_position="top right")

    # Mark turning points (largest swing overs)
    for inn_df, x_offset, color in [(m1, 0, "#2ecc71"), (m2, 20, "#3498db")]:
        if inn_df.empty:
            continue
        _wp = inn_df["win_prob"].values
        _swings = np.abs(np.diff(_wp))
        if len(_swings) > 0:
            top_idx = np.argsort(_swings)[-3:]
            for idx in top_idx:
                ov = inn_df["over"].iloc[idx + 1]
                fig_wp.add_annotation(
                    x=ov + x_offset, y=_wp[idx + 1] * 100,
                    text=f"Ov {ov}", showarrow=True,
                    arrowhead=2, arrowcolor=color,
                    font=dict(size=9, color=color),
                    ax=0, ay=-25,
                )

    fig_wp.update_layout(
        title="Win Probability Timeline (Overs 0–19 = Inn 1, 20–39 = Inn 2)",
        xaxis_title="Over",
        yaxis=dict(title="Win probability (%)", range=[0, 100]),
        height=380,
        legend=dict(orientation="h", y=1.02),
        hovermode="x unified",
    )
    st.plotly_chart(fig_wp, use_container_width=True)

    # SOM trajectory scatter on 6x6 grid (Inn 1)
    if "node_label" in m1.columns and "som_node" in m1.columns:
        SX, SY = arts["SOM_X"], arts["SOM_Y"]
        m1 = m1.copy()
        m1["grid_row"] = m1["som_node"] // SY
        m1["grid_col"] = m1["som_node"] %  SY

        # Background: node win rates as heatmap
        wr_arr = np.full((SX, SY), 0.5)
        for nd, wr in arts["node_wr_map"].items():
            r, c = nd // SY, nd % SY
            if r < SX and c < SY:
                wr_arr[r, c] = wr

        fig_traj = go.Figure()
        fig_traj.add_trace(go.Heatmap(
            z=wr_arr, colorscale="RdYlGn", zmin=0, zmax=1,
            opacity=0.55, showscale=True,
            colorbar=dict(title="Node WR", len=0.6, thickness=12),
            hoverinfo="skip",
        ))
        cmap_ov = m1["over"].values
        fig_traj.add_trace(go.Scatter(
            x=m1["grid_col"], y=m1["grid_row"],
            mode="lines+markers+text",
            text=[str(o) for o in m1["over"]],
            textposition="top center",
            textfont=dict(size=9),
            marker=dict(
                size=12,
                color=cmap_ov,
                colorscale="Blues",
                showscale=False,
                line=dict(color="black", width=0.8),
            ),
            line=dict(color="black", width=1.5, dash="dot"),
            hovertemplate="Over %{text}: Node row %{y} col %{x}<extra></extra>",
        ))
        fig_traj.update_layout(
            title=f"{bat_team} — SOM Game State Trajectory (Inn 1)  "
                  f"[darker blue = later overs]",
            xaxis=dict(title="SOM column", dtick=1, range=[-0.5, SY - 0.5]),
            yaxis=dict(title="SOM row", dtick=1, range=[SX - 0.5, -0.5]),
            height=400,
            margin=dict(t=60, b=40),
        )
        st.plotly_chart(fig_traj, use_container_width=True)

    # Over-by-over table
    tbl_cols = [c for c in ["over", "cum_runs", "cum_wickets", "som_node",
                             "node_label", "win_prob"] if c in m1.columns]
    st.dataframe(
        m1[tbl_cols]
        .rename(columns={"over": "Over", "cum_runs": "Runs", "cum_wickets": "Wkts",
                         "som_node": "SOM", "node_label": "State", "win_prob": "Win Prob"})
        .style.format({"Win Prob": "{:.1%}"}),
        use_container_width=True,
        hide_index=True,
    )


# ─── Page 4: SOM Map Explorer ─────────────────────────────────────────────────

def _render_som_panel(sx, sy, visit_source, node_labels_map,
                      node_wr_dict, ent_arr, key_sfx):
    metric = st.radio(
        "Colour by",
        ["Win Rate", "Visit Count", "Transition Entropy"],
        horizontal=True, key=f"metric_{key_sfx}",
    )
    N = sx * sy
    counts = np.zeros(N)
    for nid, cnt in visit_source["som_node"].value_counts().items():
        if int(nid) < N:
            counts[int(nid)] = cnt

    wr_arr = np.array([node_wr_dict.get(n, 0.5) for n in range(N)])

    if metric == "Win Rate":
        data    = wr_arr.reshape(sx, sy)
        cscale  = "RdYlGn"; zmin, zmax = 0, 1
        lbl_fmt = ".2f"
        title   = f"Node Win Rate (bat-first team, {key_sfx})"
    elif metric == "Visit Count":
        data   = counts.reshape(sx, sy)
        cscale = "Blues"; zmin, zmax = 0, counts.max() or 1
        lbl_fmt = ".0f"
        title  = f"Node Visit Count ({key_sfx})"
    else:
        if ent_arr is None:
            st.info("Entropy not available for this SOM version.")
            return
        data   = ent_arr.reshape(sx, sy)
        cscale = "hot"; zmin, zmax = 0, float(ent_arr.max())
        lbl_fmt = ".1f"
        title  = f"Transition Entropy bits ({key_sfx})"

    text_grid = []
    for r in range(sx):
        row_t = []
        for c in range(sy):
            nid = r * sy + c
            lbl = node_labels_map.get(nid, "")[:10]
            val = data[r, c]
            row_t.append(f"#{nid}<br>{lbl}<br>{val:{lbl_fmt}}")
        text_grid.append(row_t)

    ht = max(380, sx * 48)
    fig = _som_heatmap(data, text_grid, title=title, colorscale=cscale,
                       zmin=zmin, zmax=zmax, height=ht)
    st.plotly_chart(fig, use_container_width=True)

    # Top nodes table
    top_n   = st.slider("Show top N nodes by win rate", 5, 30, 10, key=f"topn_{key_sfx}")
    top_ids = sorted(range(N), key=lambda n: wr_arr[n], reverse=True)[:top_n]
    rows = [{
        "Node":        nid,
        "Label":       node_labels_map.get(nid, "?"),
        "Win Rate":    f"{wr_arr[nid]:.1%}",
        "Visit Count": int(counts[nid]),
    } for nid in top_ids]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def page_som(arts):
    st.title("SOM Map Explorer")
    st.caption("Interactive topological map of IPL game states — "
               "hover over any node to see its profile.")

    tab_v1, tab_v2 = st.tabs(["v1 — 6×6  (9 features, used by live predictor)",
                               "v2 — 10×10  (28 features, Phases 7–9)"])

    with tab_v1:
        _render_som_panel(
            sx=arts["SOM_X"], sy=arts["SOM_Y"],
            visit_source=arts["inn1_nodes"],
            node_labels_map=arts["node_labels"],
            node_wr_dict=arts["node_wr_map"],
            ent_arr=arts.get("ent_v1"),
            key_sfx="v1",
        )

    with tab_v2:
        _render_som_panel(
            sx=arts["SOM_X2"], sy=arts["SOM_Y2"],
            visit_source=arts["inn1_v2_compact"],
            node_labels_map=arts["node_labels_v2"],
            node_wr_dict=arts["node_wr_v2"],
            ent_arr=arts.get("ent_i1"),
            key_sfx="v2",
        )


# ─── Page 5: Venue Analysis ───────────────────────────────────────────────────

def page_venues(arts):
    st.title("Venue Analysis")
    st.caption("Bat-first vs chasing win rates across IPL venues (2008–2025).")

    venue_stats = arts["venue_stats"].copy()
    min_m = st.slider("Minimum matches at venue", 5, 50, 15)
    filtered = venue_stats[venue_stats["matches"] >= min_m].sort_values("bat_first_wr")

    colors = ["#e74c3c" if v < 0.5 else "#2ecc71" for v in filtered["bat_first_wr"]]
    fig = go.Figure(go.Bar(
        x=filtered["bat_first_wr"] * 100,
        y=filtered["venue"].apply(lambda v: v[:38]),
        orientation="h",
        marker_color=colors,
        text=[f"{v*100:.0f}%  ({int(m)}m)"
              for v, m in zip(filtered["bat_first_wr"], filtered["matches"])],
        textposition="outside",
        hovertemplate="%{y}<br>Bat-first WR: %{x:.1f}%<extra></extra>",
    ))
    fig.add_vline(x=50, line_dash="dash", line_color="black", opacity=0.4,
                  annotation_text="50%", annotation_position="top left")
    fig.update_layout(
        title="Bat-First Win Rate by Venue  "
              "(green = favours batting first, red = favours chasing)",
        xaxis=dict(title="Bat-first win rate (%)", range=[15, 85]),
        yaxis_title="",
        height=max(400, len(filtered) * 30),
        margin=dict(l=10, r=130, t=60, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Venues shown", len(filtered))
    c2.metric("Avg bat-first WR", f"{filtered['bat_first_wr'].mean():.1%}")
    best_chase = filtered.loc[filtered["bat_first_wr"].idxmin()]
    c3.metric("Best chasing venue", best_chase["venue"][:28],
              f"{best_chase['bat_first_wr']:.0%} bat-first WR")

    # Trajectory & event summary
    st.markdown("---")
    st.subheader("Tactical Event Summary")
    ev1 = arts.get("events_i1")
    ev2 = arts.get("events_i2")
    if ev1 is not None and ev2 is not None:
        tc1, tc2, tc3, tc4 = st.columns(4)
        tc1.metric("Inn 1 Collapses", f"{(ev1['event']=='COLLAPSE').sum():,}")
        tc2.metric("Inn 1 Surges",    f"{(ev1['event']=='SURGE').sum():,}")
        tc3.metric("Inn 2 Collapses", f"{(ev2['event']=='COLLAPSE').sum():,}")
        tc4.metric("Inn 2 Surges",    f"{(ev2['event']=='SURGE').sum():,}")

        # Collapse/surge by over
        for ev_df, inn_label in [(ev1, "Innings 1"), (ev2, "Innings 2")]:
            with st.expander(f"{inn_label} — Event Frequency by Over"):
                _col = ev_df[ev_df["event"] == "COLLAPSE"]["over"].value_counts().reindex(range(20), fill_value=0)
                _sur = ev_df[ev_df["event"] == "SURGE"   ]["over"].value_counts().reindex(range(20), fill_value=0)
                fig_ev = go.Figure()
                fig_ev.add_trace(go.Bar(x=list(range(20)), y=_col, name="Collapses",
                                        marker_color="#e74c3c", opacity=0.75))
                fig_ev.add_trace(go.Bar(x=list(range(20)), y=_sur, name="Surges",
                                        marker_color="#2ecc71", opacity=0.75))
                fig_ev.update_layout(
                    title=f"{inn_label} — Collapses & Surges by Over",
                    xaxis_title="Over", yaxis_title="Event count",
                    barmode="group", height=280,
                    legend=dict(orientation="h", y=1.02),
                )
                st.plotly_chart(fig_ev, use_container_width=True)

    # Markov transition heatmap
    with st.expander("Markov Transition Matrix (Innings 1, 10×10 SOM)"):
        T = arts["T1_prob"]
        fig_t = px.imshow(
            T, color_continuous_scale="YlOrRd", aspect="auto",
            title="Innings-1 Transition Probabilities (row = from node, col = to node)",
            labels={"x": "Destination node", "y": "Source node"},
        )
        fig_t.update_layout(height=520)
        st.plotly_chart(fig_t, use_container_width=True)


# ─── Transformer loader ───────────────────────────────────────────────────────

@st.cache_resource
def load_transformer():
    if not _TORCH_OK:
        return None, None
    path = os.path.join(os.path.dirname(__file__), "artifacts", "transformer_model.pt")
    if not os.path.exists(path):
        return None, None
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg  = ckpt["model_config"]
    model = NextBallTransformer(**cfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def _predict_state(model, over, cum_runs, cum_wickets,
                   innings=1, runs_this_over=0.0, ball_in_over=0):
    feats = state_to_features(over, cum_runs, cum_wickets,
                              innings, ball_in_over, runs_this_over)
    past  = torch.full((1, 1), BOS_TOKEN, dtype=torch.long)
    with torch.no_grad():
        probs = model.predict_next(feats, past)
    return probs.numpy()


# ─── Page 6: Transformer — Next-Ball Predictor ────────────────────────────────

def page_transformer(arts):
    st.title("Next-Ball Outcome Predictor")
    st.caption(
        "GPT-style causal transformer trained on 278K IPL deliveries.  "
        "Given match context, predicts the probability distribution over the next ball's outcome."
    )

    model, ckpt = load_transformer()

    if model is None:
        st.warning(
            "Transformer model not found.  "
            "Run `python3 train_transformer.py` from the project directory first."
        )
        return

    tab_pred, tab_sim, tab_meta = st.tabs(
        ["Next-Ball Distribution", "Ball-by-Ball Simulator", "Model Diagnostics"]
    )

    # ── Tab 1: Distribution for current state ──────────────────────────────
    with tab_pred:
        st.subheader("Predict next ball from current match state")
        lc, rc = st.columns([1, 1.6], gap="large")

        with lc:
            innings      = st.radio("Innings", [1, 2], horizontal=True, key="t_inn")
            over         = st.slider("Overs completed", 0, 19, 8,  key="t_ov")
            cum_runs     = st.number_input("Runs so far",     0, 400, 68, key="t_cr")
            cum_wickets  = st.number_input("Wickets fallen",  0, 9,   2,  key="t_cw")
            runs_this_over = st.number_input("Runs this over", 0, 36,  5, key="t_rto")
            ball_in_over = st.slider("Ball in over (0-5)", 0, 5, 2,       key="t_bov")

        probs  = _predict_state(model, over, cum_runs, cum_wickets,
                                innings, runs_this_over, ball_in_over)
        freqs  = np.array(ckpt.get("class_freqs", [0.35,0.37,0.06,0.003,0.115,0.052,0.05,0.0]))

        with rc:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=OUTCOME_LABELS, y=(probs * 100).tolist(),
                marker_color=OUTCOME_COLORS,
                name="Model prediction",
                text=[f"{p:.1f}%" for p in probs * 100],
                textposition="outside",
            ))
            fig.add_trace(go.Scatter(
                x=OUTCOME_LABELS, y=(freqs * 100).tolist(),
                mode="markers+lines",
                name="Historical base rate",
                line=dict(color="gray", dash="dot", width=1.5),
                marker=dict(color="gray", size=7),
            ))
            fig.update_layout(
                title="Next-Ball Outcome Probabilities vs Historical Base Rate",
                yaxis=dict(title="Probability (%)", range=[0, max(probs.max()*130, 55)]),
                xaxis_title="Outcome",
                height=360,
                legend=dict(orientation="h", y=1.02),
                bargap=0.3,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Highlight biggest deviations from base rate
            deltas = probs - freqs
            top_up  = OUTCOME_LABELS[int(deltas.argmax())]
            top_dn  = OUTCOME_LABELS[int(deltas.argmin())]
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Most likely", OUTCOME_LABELS[int(probs.argmax())],
                         f"{probs.max():.1%}")
            col_b.metric("Boosted vs baseline", top_up,
                         f"{deltas.max():+.1%}")
            col_c.metric("Suppressed vs baseline", top_dn,
                         f"{deltas.min():+.1%}")

        # Context-sensitivity demo: sweep over number
        with st.expander("How do probabilities change over the innings?"):
            overs_range = list(range(0, 20))
            rows = []
            for ov in overs_range:
                # Use a typical score at that over (6 rr baseline)
                cr  = ov * 6
                cw  = min(int(ov * 0.15), 9)
                rto = 6.0
                p   = _predict_state(model, ov, cr, cw, innings=1, runs_this_over=rto)
                rows.append(p)
            arr = np.array(rows)   # (20, 8)

            fig_ov = go.Figure()
            for cls_i, (lbl, col) in enumerate(zip(OUTCOME_LABELS, OUTCOME_COLORS)):
                fig_ov.add_trace(go.Scatter(
                    x=overs_range, y=(arr[:, cls_i] * 100).tolist(),
                    mode="lines", name=lbl,
                    line=dict(color=col, width=2),
                ))
            fig_ov.update_layout(
                title="Predicted Next-Ball Probabilities Across the Innings "
                      "(typical score trajectory, Inn 1)",
                xaxis_title="Over",
                yaxis_title="Probability (%)",
                height=360,
                legend=dict(orientation="h", y=-0.2),
                hovermode="x unified",
            )
            st.plotly_chart(fig_ov, use_container_width=True)

    # ── Tab 2: Ball-by-Ball Simulator ─────────────────────────────────────
    with tab_sim:
        st.subheader("Step through an innings ball by ball")
        st.caption(
            "The model uses the FULL history of realized outcomes (not just current state summary) — "
            "probabilities update based on exactly what has happened so far."
        )

        # Session-state: hold simulation
        def _reset_sim():
            st.session_state.sim_feats    = []
            st.session_state.sim_outcomes = []
            st.session_state.sim_over     = 0
            st.session_state.sim_bov      = 0
            st.session_state.sim_runs     = 0
            st.session_state.sim_wkts     = 0
            st.session_state.sim_rto      = 0
            st.session_state.sim_balls    = 0
            st.session_state.sim_inn      = 1

        if "sim_over" not in st.session_state:
            _reset_sim()

        s_inn = st.radio("Innings", [1, 2], horizontal=True,
                         index=st.session_state.sim_inn - 1, key="sim_inn_sel")
        if s_inn != st.session_state.sim_inn:
            st.session_state.sim_inn = s_inn
            _reset_sim()

        status_cols = st.columns(5)
        status_cols[0].metric("Over",    f"{st.session_state.sim_over}.{st.session_state.sim_bov}")
        status_cols[1].metric("Score",   f"{st.session_state.sim_runs}/{st.session_state.sim_wkts}")
        status_cols[2].metric("RR",      f"{st.session_state.sim_runs / max(st.session_state.sim_balls/6, 0.5):.2f}")
        status_cols[3].metric("Balls",   f"{st.session_state.sim_balls}")
        status_cols[4].metric("Deliveries", f"{len(st.session_state.sim_outcomes)}")

        # Get current prediction
        h_feats    = st.session_state.sim_feats
        h_outcomes = st.session_state.sim_outcomes

        if h_feats:
            feat_seq = torch.tensor(np.array(h_feats), dtype=torch.float32).unsqueeze(0)  # (1,T,F)
            out_seq  = torch.tensor(h_outcomes, dtype=torch.long).unsqueeze(0)             # (1,T)
            with torch.no_grad():
                curr_probs = model.predict_next(feat_seq, out_seq).numpy()
        else:
            curr_probs = _predict_state(
                model, 0, 0, 0, st.session_state.sim_inn
            ).numpy() if not isinstance(_predict_state(
                model, 0, 0, 0, st.session_state.sim_inn), np.ndarray
            ) else _predict_state(model, 0, 0, 0, st.session_state.sim_inn)

        # Probability bar
        fig_sim = go.Figure(go.Bar(
            x=OUTCOME_LABELS, y=(curr_probs * 100).tolist(),
            marker_color=OUTCOME_COLORS,
            text=[f"{p:.1f}%" for p in curr_probs * 100],
            textposition="outside",
        ))
        fig_sim.update_layout(
            title="Predicted probabilities for NEXT ball",
            yaxis=dict(title="Probability (%)", range=[0, 60]),
            height=300, bargap=0.3, margin=dict(t=50, b=30),
        )
        st.plotly_chart(fig_sim, use_container_width=True)

        # Outcome buttons
        OUTCOME_RUNS   = [0, 1, 2, 3, 4, 6, 0, 0]   # runs for each class
        IS_VALID_BALL  = [1, 1, 1, 1, 1, 1, 1, 0]   # 0 = extra (no ball count)
        IS_WICKET      = [0, 0, 0, 0, 0, 0, 1, 0]

        st.markdown("**Realize the next delivery:**")
        btn_cols = st.columns(len(OUTCOME_LABELS))
        for i, (lbl, col_hex) in enumerate(zip(OUTCOME_LABELS, OUTCOME_COLORS)):
            if btn_cols[i].button(lbl, key=f"btn_{i}",
                                  disabled=(st.session_state.sim_wkts >= 10 or
                                            st.session_state.sim_over >= 20)):
                ov  = st.session_state.sim_over
                bov = st.session_state.sim_bov
                cr  = st.session_state.sim_runs
                cw  = st.session_state.sim_wkts
                rto = st.session_state.sim_rto
                bb  = st.session_state.sim_balls

                # Record features BEFORE this ball
                feat_vec = state_to_features(ov, cr, cw,
                    st.session_state.sim_inn, bov, rto
                ).squeeze(0).squeeze(0).numpy()
                st.session_state.sim_feats.append(feat_vec)
                st.session_state.sim_outcomes.append(i)

                # Advance state
                runs = OUTCOME_RUNS[i]
                st.session_state.sim_runs  += runs
                st.session_state.sim_rto   += runs
                if IS_WICKET[i]:
                    st.session_state.sim_wkts += 1
                if IS_VALID_BALL[i]:
                    st.session_state.sim_balls += 1
                    st.session_state.sim_bov  += 1
                    if st.session_state.sim_bov >= 6:
                        st.session_state.sim_over += 1
                        st.session_state.sim_bov   = 0
                        st.session_state.sim_rto   = 0
                else:
                    st.session_state.sim_runs += 1  # extra penalty run
                st.rerun()

        st.button("Reset innings", on_click=_reset_sim)

        if len(h_outcomes) > 1:
            with st.expander("Delivery history"):
                hist_df = pd.DataFrame({
                    "Ball": range(1, len(h_outcomes) + 1),
                    "Outcome": [OUTCOME_LABELS[o] for o in h_outcomes],
                })
                st.dataframe(hist_df, use_container_width=True, hide_index=True)

    # ── Tab 3: Model Diagnostics ───────────────────────────────────────────
    with tab_meta:
        history = ckpt.get("train_history", [])
        baseline_ce = ckpt.get("baseline_ce", 1.477)
        best_val_ce = ckpt.get("best_val_ce", 1.482)

        if history:
            hist_df = pd.DataFrame(history)
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(
                x=hist_df["epoch"], y=hist_df["train_ce"],
                mode="lines+markers", name="Train CE",
                line=dict(color="#3498db", width=2),
            ))
            fig_loss.add_trace(go.Scatter(
                x=hist_df["epoch"], y=hist_df["val_ce"],
                mode="lines+markers", name="Val CE",
                line=dict(color="#e74c3c", width=2),
            ))
            fig_loss.add_hline(
                y=baseline_ce, line_dash="dash", line_color="gray",
                annotation_text=f"Frequency baseline ({baseline_ce:.3f})",
                annotation_position="right",
            )
            fig_loss.update_layout(
                title="Training History — Cross-Entropy Loss",
                xaxis_title="Epoch", yaxis_title="Cross-Entropy",
                height=320,
                legend=dict(orientation="h", y=1.02),
            )
            st.plotly_chart(fig_loss, use_container_width=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Best Val CE",    f"{best_val_ce:.4f}")
        c2.metric("Baseline CE",    f"{baseline_ce:.4f}")
        c3.metric("Val Perplexity", f"{float(np.exp(best_val_ce)):.2f}")
        c4.metric("Parameters",     "151 K")

        st.markdown("""
**Architecture**
- GPT-style causal transformer with pre-LayerNorm
- `d_model=64`, `nhead=4`, `num_layers=3`, `dropout=0.1`
- Input: outcome class embedding (32-dim) + 10 numerical features (32-dim) → 64-dim
- Causal self-attention prevents look-ahead leakage
- Output: 8-class softmax over next-ball outcomes

**Training data**
- 278,205 deliveries across 1,169 IPL matches (2008–2025)
- 2,365 innings sequences (avg 118 balls each)
- Train/val split: 85/15 by match ordering

**Why perplexity is close to baseline**
Cricket ball outcomes are inherently stochastic — even knowing the full innings
history, a six on any given delivery is genuinely unpredictable. The model's
context-sensitivity shows up as *relative* probability shifts (e.g., higher
boundary probability in death overs), not lower absolute entropy.
        """)

        # Class frequency comparison
        freqs = np.array(ckpt.get("class_freqs",
                                   [0.35, 0.37, 0.06, 0.003, 0.115, 0.052, 0.05, 0.0]))
        fig_freq = go.Figure(go.Bar(
            x=OUTCOME_LABELS, y=(freqs * 100).tolist(),
            marker_color=OUTCOME_COLORS,
            text=[f"{f:.1f}%" for f in freqs * 100],
            textposition="outside",
        ))
        fig_freq.update_layout(
            title="Training Set Outcome Distribution",
            yaxis_title="Frequency (%)", height=280, bargap=0.3,
        )
        st.plotly_chart(fig_freq, use_container_width=True)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    arts = load_artifacts()

    st.sidebar.title("🏏 IPL SOM Analytics")
    st.sidebar.caption("Multi-Dimensional Unsupervised Learning Framework")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Navigate",
        [
            "Live Match Prediction",
            "Next-Ball Predictor",
            "Player Explorer",
            "Match Replay",
            "SOM Map Explorer",
            "Venue Analysis",
        ],
    )

    st.sidebar.markdown("---")
    n_m = arts.get("n_matches", "?")
    n_b = arts.get("n_balls",   "?")
    st.sidebar.caption(f"Dataset: {n_m:,} matches · {n_b:,} balls")
    st.sidebar.caption("IPL 2008–2025")

    if page == "Live Match Prediction":
        page_live(arts)
    elif page == "Next-Ball Predictor":
        page_transformer(arts)
    elif page == "Player Explorer":
        page_players(arts)
    elif page == "Match Replay":
        page_replay(arts)
    elif page == "SOM Map Explorer":
        page_som(arts)
    elif page == "Venue Analysis":
        page_venues(arts)


if __name__ == "__main__":
    main()
