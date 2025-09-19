import warnings
warnings.filterwarnings("ignore")

import time
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

import streamlit as st
from rapidfuzz import process, fuzz
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import plotly.express as px

# ----------------------------
# Defaults
# ----------------------------
DEFAULT_SEASON = "2023-24"               # Format: YYYY-YY
DEFAULT_PER_MODE = "Per100Possessions"   # "PerGame", "Per36", "Per100Possessions"
MIN_MINUTES = 500

DEFAULT_WEIGHTS = {
    # efficiency
    "TS_PCT": 1.3,
    "EFG_PCT": 0.8,
    # shooting profile
    "THREE_PAR": 1.0,
    "FTR": 0.7,
    # usage-ish volume
    "FGA": 0.7,
    "FG3A": 0.8,
    "FTA": 0.6,
    # box score impact
    "PTS": 1.2,
    "AST": 1.0,
    "REB": 0.9,
    "STL": 0.8,
    "BLK": 0.8,
    "TOV": 0.8,
}

FEATURE_ORDER = list(DEFAULT_WEIGHTS.keys())

# ----------------------------
# Data fetch & caching
# ----------------------------
@st.cache_data(show_spinner=True, ttl=60 * 60)
def fetch_season(season: str, per_mode: str) -> pd.DataFrame:
    """Fetch one season of league player stats and engineer features."""
    try:
        from nba_api.stats.endpoints import leaguedashplayerstats
        
        # NBA stats can occasionally rate-limit; small retry loop helps.
        for attempt in range(3):
            try:
                st.write(f"🔄 Fetching data (attempt {attempt + 1}/3)...")
                data = leaguedashplayerstats.LeagueDashPlayerStats(
                    season=season,
                    per_mode_detailed=per_mode,
                    measure_type_detailed_defense="Base",
                    season_type_all_star="Regular Season",
                ).get_data_frames()[0]
                break
            except Exception as e:
                st.write(f"⚠️ Attempt {attempt + 1} failed: {str(e)}")
                if attempt == 2:
                    raise
                time.sleep(2)  # Wait longer between attempts

        st.write("✅ Data fetched successfully!")
        
        # Aggregate across teams when traded, sum counting stats for stability
        g = data.groupby(["PLAYER_ID", "PLAYER_NAME"], as_index=False)
        data = g.aggregate({
            "GP": "sum", "MIN": "sum", "TEAM_ABBREVIATION": "last",
            "FGM": "sum", "FGA": "sum", "FG3M": "sum", "FG3A": "sum",
            "FTM": "sum", "FTA": "sum",
            "OREB": "sum", "DREB": "sum", "REB": "sum",
            "AST": "sum", "STL": "sum", "BLK": "sum", "TOV": "sum",
            "PF": "sum", "PTS": "sum",
            "FG_PCT": "mean", "FG3_PCT": "mean", "FT_PCT": "mean",
        })

        # Filter tiny samples (minutes total in the per-mode space)
        data = data[data["MIN"] >= MIN_MINUTES].copy()

        # Derived profiles
        data["TS_PCT"]     = data["PTS"] / (2 * (data["FGA"] + 0.44 * data["FTA"]).replace(0, np.nan))
        data["EFG_PCT"]    = (data["FGM"] + 0.5 * data["FG3M"]) / data["FGA"].replace(0, np.nan)
        data["THREE_PAR"]  = data["FG3A"] / data["FGA"].replace(0, np.nan)
        data["FTR"]        = data["FTA"] / data["FGA"].replace(0, np.nan)

        for c in ["TS_PCT", "EFG_PCT", "THREE_PAR", "FTR"]:
            data[c] = data[c].fillna(0)

        feats = FEATURE_ORDER
        keep_cols = ["PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "GP", "MIN"] + feats
        df = data[keep_cols].copy()
        df.insert(1, "SEASON", season)
        
        st.write(f"✅ Processed {len(df)} players")
        return df
        
    except Exception as e:
        st.error(f"❌ Error fetching data: {str(e)}")
        st.write("This might be due to:")
        st.write("- NBA API being temporarily unavailable")
        st.write("- Network connectivity issues")
        st.write("- Rate limiting")
        st.write("Please try again in a few minutes.")
        return pd.DataFrame()  # Return empty dataframe

def build_embedding(df: pd.DataFrame, feature_weights: Dict[str, float]) -> Tuple[List[str], np.ndarray]:
    if df.empty:
        return [], np.array([])
        
    feats = list(feature_weights.keys())
    X = df[feats].astype(float).values

    # Standardize within season (df is a single season here, but this supports multi-season later)
    X_scaled = StandardScaler().fit_transform(X)

    # Apply weights
    w = np.array([feature_weights[f] for f in feats], dtype=float)
    Xw = X_scaled * w
    return feats, Xw

def fuzzy_find(df: pd.DataFrame, query: str, limit: int = 8):
    if df.empty:
        return []
    choices = df["PLAYER_NAME"].tolist()
    return process.extract(query, choices, scorer=fuzz.WRatio, limit=limit)

def get_comps(df: pd.DataFrame, Xw: np.ndarray, player_name: str, k: int = 10):
    if df.empty or len(Xw) == 0:
        return None, [], []
        
    matches = fuzzy_find(df, player_name, limit=5)
    if not matches:
        return None, [], []

    best_name, score, _ = matches[0]
    base_idxs = df.index[df["PLAYER_NAME"] == best_name].tolist()
    if not base_idxs:
        return None, [], matches
    base_idx = base_idxs[0]

    sims = cosine_similarity(Xw[base_idx:base_idx + 1], Xw)[0]
    order = np.argsort(-sims)

    rows = []
    rank = 1
    for j in order:
        if j == base_idx:
            continue
        rows.append({
            "Rank": rank,
            "Player": df.loc[j, "PLAYER_NAME"],
            "Team": df.loc[j, "TEAM_ABBREVIATION"],
            "Season": df.loc[j, "SEASON"],
            "Similarity": round(float(sims[j]), 4)
        })
        rank += 1
        if rank > k:
            break

    return best_name, rows, matches

# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="NBA Player Comps (AI-powered)", layout="wide")
st.title("🏀 AI-Powered NBA Player Comparisons")

# Add a status indicator
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

with st.sidebar:
    st.header("Controls")
    season = st.text_input("Season (YYYY-YY)", value=DEFAULT_SEASON, help='Example: "2023-24"')
    per_mode = st.selectbox("Per Mode", ["PerGame", "Per36", "Per100Possessions"], index=2)

    st.markdown("### Feature Weights")
    weights = {}
    for f in FEATURE_ORDER:
        default = DEFAULT_WEIGHTS[f]
        # Use sliders in sensible ranges: efficiency [0–2], volume/impact [0–2]
        max_val = 2.0
        step = 0.05
        weights[f] = st.slider(f, 0.0, max_val, float(default), step=step)

    st.markdown("### Filters")
    topk = st.slider("Number of comps", 5, 25, 10, 1)
    min_minutes = st.number_input("Minimum minutes (season total)", min_value=0, value=MIN_MINUTES, step=50)

st.caption("Tip: weights change the notion of 'style'. Bump **TS_PCT & THREE_PAR** for perimeter comps; bump **REB/BLK** for bigs.")

# Fetch data
if st.button("🔄 Load Data", type="primary") or st.session_state.data_loaded:
    with st.spinner(f"Fetching {season} league stats ({per_mode})..."):
        df = fetch_season(season, per_mode).copy()
        
        if not df.empty:
            # Apply UI min minutes filter
            df = df[df["MIN"] >= int(min_minutes)].reset_index(drop=True)
            st.session_state.data_loaded = True
            st.success(f"Loaded {len(df)} players for {season}.")
            
            # Player selection
            all_names = sorted(df["PLAYER_NAME"].unique().tolist())
            col_left, col_right = st.columns([1, 1])
            with col_left:
                anchor_name = st.selectbox("Choose a player", all_names, index=all_names.index("Stephen Curry") if "Stephen Curry" in all_names else 0)
            with col_right:
                query_text = st.text_input("...or type a name (fuzzy search)", value="")

            if query_text.strip():
                # Show fuzzy results to help the user pick
                candidates = fuzzy_find(df, query_text.strip(), limit=6)
                st.write("**Closest matches:** " + ", ".join([c[0] for c in candidates]))
                # Best match becomes anchor_name
                if candidates:
                    anchor_name = candidates[0][0]

            # Build embedding & comps
            feats, Xw = build_embedding(df, weights)
            anchor, rows, matches = get_comps(df, Xw, anchor_name, k=topk)

            if anchor is None:
                st.error("No matching player found. Try a different name.")
            else:
                st.subheader(f"Top {topk} comps for **{anchor}** — {season}, {per_mode}")
                comp_df = pd.DataFrame(rows)
                st.dataframe(comp_df, use_container_width=True)

                # ----------------------------
                # 2D Visualization (PCA)
                # ----------------------------
                st.markdown("### Visualize player space (PCA)")
                with st.expander("Show 2D map"):
                    pca = PCA(n_components=2, random_state=42)
                    coords = pca.fit_transform(Xw)
                    df_plot = df.copy()
                    df_plot["x"] = coords[:, 0]
                    df_plot["y"] = coords[:, 1]

                    # Identify neighbors indices used in rows
                    neighbor_names = set(comp_df["Player"].tolist())
                    df_plot["Group"] = np.where(df_plot["PLAYER_NAME"] == anchor, "Anchor",
                                         np.where(df_plot["PLAYER_NAME"].isin(neighbor_names), "Neighbor", "Other"))

                    fig = px.scatter(
                        df_plot,
                        x="x", y="y",
                        color="Group",
                        hover_name="PLAYER_NAME",
                        hover_data={"TEAM_ABBREVIATION": True, "x": False, "y": False, "Group": False},
                        opacity=0.9
                    )
                    # Emphasize anchor & neighbors
                    fig.update_traces(marker=dict(size=9))
                    for grp, size in [("Anchor", 16), ("Neighbor", 12), ("Other", 7)]:
                        fig.update_traces(selector=lambda t: t.legendgroup == grp, marker=dict(size=size))
                    st.plotly_chart(fig, use_container_width=True)

                # ----------------------------
                # Feature importance explain
                # ----------------------------
                with st.expander("What do these features mean?"):
                    st.markdown("""
                    - **TS_PCT** – True shooting% (overall scoring efficiency).
                    - **EFG_PCT** – Effective FG% (weights 3PT makes as 1.5 shots).
                    - **THREE_PAR** – 3PA / FGA (how three-point heavy a player is).
                    - **FTR** – FTA / FGA (how often they get to the line).
                    - **FGA/FG3A/FTA** – Shot volume profile under your chosen *Per Mode*.
                    - **PTS/AST/REB/STL/BLK/TOV** – Box score impact under your chosen *Per Mode*.
                    """)

                st.caption("Built with normalized, weighted cosine similarity. Season-only for apples-to-apples. Try different weights to change 'style' emphasis.")
        else:
            st.error("Failed to load data. Please try again.")

else:
    st.info("👆 Click 'Load Data' to start analyzing NBA players!")
    st.write("This app will:")
    st.write("- Fetch current NBA player statistics")
    st.write("- Find similar players using AI")
    st.write("- Show interactive visualizations")
    st.write("- Allow you to customize analysis parameters")
