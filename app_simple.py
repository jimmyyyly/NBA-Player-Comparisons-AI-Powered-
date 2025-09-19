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
DEFAULT_SEASON = "2023-24"
DEFAULT_PER_MODE = "Per100Possessions"
MIN_MINUTES = 500

DEFAULT_WEIGHTS = {
    "TS_PCT": 1.3,
    "EFG_PCT": 0.8,
    "THREE_PAR": 1.0,
    "FTR": 0.7,
    "FGA": 0.7,
    "FG3A": 0.8,
    "FTA": 0.6,
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
        
        st.write("🔄 Fetching NBA data...")
        
        # Try with timeout
        data = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            per_mode_detailed=per_mode,
            measure_type_detailed_defense="Base",
            season_type_all_star="Regular Season",
        ).get_data_frames()[0]
        
        st.write(f"✅ Raw data loaded: {len(data)} players")
        
        # Process data
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

        # Filter by minutes
        data = data[data["MIN"] >= MIN_MINUTES].copy()
        st.write(f"✅ Filtered to {len(data)} players with {MIN_MINUTES}+ minutes")

        # Calculate advanced stats
        data["TS_PCT"] = data["PTS"] / (2 * (data["FGA"] + 0.44 * data["FTA"]).replace(0, np.nan))
        data["EFG_PCT"] = (data["FGM"] + 0.5 * data["FG3M"]) / data["FGA"].replace(0, np.nan)
        data["THREE_PAR"] = data["FG3A"] / data["FGA"].replace(0, np.nan)
        data["FTR"] = data["FTA"] / data["FGA"].replace(0, np.nan)

        # Fill NaN values
        for c in ["TS_PCT", "EFG_PCT", "THREE_PAR", "FTR"]:
            data[c] = data[c].fillna(0)

        # Select final columns
        feats = FEATURE_ORDER
        keep_cols = ["PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "GP", "MIN"] + feats
        df = data[keep_cols].copy()
        df.insert(1, "SEASON", season)
        
        st.write(f"✅ Final dataset: {len(df)} players ready")
        return df
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.write("This might be due to:")
        st.write("- NBA API being temporarily unavailable")
        st.write("- Network connectivity issues")
        st.write("- Rate limiting")
        return pd.DataFrame()

def build_embedding(df: pd.DataFrame, feature_weights: Dict[str, float]) -> Tuple[List[str], np.ndarray]:
    if df.empty:
        return [], np.array([])
        
    feats = list(feature_weights.keys())
    X = df[feats].astype(float).values
    X_scaled = StandardScaler().fit_transform(X)
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

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'df' not in st.session_state:
    st.session_state.df = pd.DataFrame()

with st.sidebar:
    st.header("Controls")
    season = st.text_input("Season (YYYY-YY)", value=DEFAULT_SEASON, help='Example: "2023-24"')
    per_mode = st.selectbox("Per Mode", ["PerGame", "Per36", "Per100Possessions"], index=2)

    st.markdown("### Feature Weights")
    weights = {}
    for f in FEATURE_ORDER:
        default = DEFAULT_WEIGHTS[f]
        max_val = 2.0
        step = 0.05
        weights[f] = st.slider(f, 0.0, max_val, float(default), step=step)

    st.markdown("### Filters")
    topk = st.slider("Number of comps", 5, 25, 10, 1)
    min_minutes = st.number_input("Minimum minutes (season total)", min_value=0, value=MIN_MINUTES, step=50)

st.caption("Tip: weights change the notion of 'style'. Bump **TS_PCT & THREE_PAR** for perimeter comps; bump **REB/BLK** for bigs.")

# Data loading section
if not st.session_state.data_loaded:
    st.info("👆 Click 'Load Data' to start analyzing NBA players!")
    
    if st.button("🔄 Load Data", type="primary"):
        with st.spinner("Fetching NBA data... This may take a moment."):
            df = fetch_season(season, per_mode)
            
            if not df.empty:
                # Apply UI min minutes filter
                df = df[df["MIN"] >= int(min_minutes)].reset_index(drop=True)
                st.session_state.df = df
                st.session_state.data_loaded = True
                st.success(f"✅ Successfully loaded {len(df)} players for {season}!")
                st.rerun()
            else:
                st.error("❌ Failed to load data. Please try again.")
else:
    df = st.session_state.df
    
    if not df.empty:
        st.success(f"✅ Data loaded: {len(df)} players for {season}")
        
        # Player selection
        all_names = sorted(df["PLAYER_NAME"].unique().tolist())
        col_left, col_right = st.columns([1, 1])
        with col_left:
            anchor_name = st.selectbox("Choose a player", all_names, index=all_names.index("Stephen Curry") if "Stephen Curry" in all_names else 0)
        with col_right:
            query_text = st.text_input("...or type a name (fuzzy search)", value="")

        if query_text.strip():
            candidates = fuzzy_find(df, query_text.strip(), limit=6)
            st.write("**Closest matches:** " + ", ".join([c[0] for c in candidates]))
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

            # 2D Visualization
            st.markdown("### Visualize player space (PCA)")
            with st.expander("Show 2D map"):
                try:
                    pca = PCA(n_components=2, random_state=42)
                    coords = pca.fit_transform(Xw)
                    df_plot = df.copy()
                    df_plot["x"] = coords[:, 0]
                    df_plot["y"] = coords[:, 1]

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
                    fig.update_traces(marker=dict(size=9))
                    for grp, size in [("Anchor", 16), ("Neighbor", 12), ("Other", 7)]:
                        fig.update_traces(selector=lambda t: t.legendgroup == grp, marker=dict(size=size))
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating visualization: {str(e)}")

            # Feature explanation
            with st.expander("What do these features mean?"):
                st.markdown("""
                - **TS_PCT** – True shooting% (overall scoring efficiency).
                - **EFG_PCT** – Effective FG% (weights 3PT makes as 1.5 shots).
                - **THREE_PAR** – 3PA / FGA (how three-point heavy a player is).
                - **FTR** – FTA / FGA (how often they get to the line).
                - **FGA/FG3A/FTA** – Shot volume profile under your chosen *Per Mode*.
                - **PTS/AST/REB/STL/BLK/TOV** – Box score impact under your chosen *Per Mode*.
                """)

        # Reset button
        if st.button("🔄 Load New Data"):
            st.session_state.data_loaded = False
            st.session_state.df = pd.DataFrame()
            st.rerun()

st.caption("Built with normalized, weighted cosine similarity. Season-only for apples-to-apples. Try different weights to change 'style' emphasis.")
