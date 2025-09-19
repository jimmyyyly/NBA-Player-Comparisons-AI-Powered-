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

# Sample data as fallback
SAMPLE_DATA = {
    "PLAYER_NAME": [
        "Stephen Curry", "LeBron James", "Kevin Durant", "Giannis Antetokounmpo", 
        "Luka Doncic", "Jayson Tatum", "Joel Embiid", "Nikola Jokic",
        "Damian Lillard", "Jimmy Butler", "Kawhi Leonard", "Paul George",
        "Anthony Davis", "Russell Westbrook", "Chris Paul", "Kyle Lowry",
        "Devin Booker", "Bradley Beal", "Donovan Mitchell", "Trae Young"
    ],
    "TEAM_ABBREVIATION": [
        "GSW", "LAL", "PHX", "MIL", "DAL", "BOS", "PHI", "DEN",
        "MIL", "MIA", "LAC", "LAC", "LAL", "LAC", "GSW", "MIA",
        "PHX", "PHX", "CLE", "ATL"
    ],
    "PTS": [29.4, 25.0, 28.2, 31.1, 32.4, 30.1, 33.1, 24.5, 32.2, 22.9, 23.6, 23.8, 26.1, 15.8, 13.9, 11.2, 27.8, 23.2, 28.3, 26.2],
    "AST": [6.3, 6.9, 5.0, 5.7, 8.0, 4.6, 3.2, 9.8, 7.3, 5.3, 3.9, 5.1, 2.6, 7.6, 9.0, 5.1, 4.5, 5.4, 4.3, 10.2],
    "REB": [6.1, 7.4, 6.7, 11.8, 8.2, 8.8, 10.2, 11.8, 4.8, 5.9, 6.2, 6.1, 12.5, 5.7, 4.3, 3.1, 4.5, 3.9, 4.3, 3.0],
    "STL": [1.0, 1.3, 0.9, 1.2, 1.4, 1.1, 1.0, 1.3, 0.9, 1.8, 1.4, 1.5, 1.2, 1.1, 1.5, 1.0, 0.9, 0.9, 1.4, 1.1],
    "BLK": [0.4, 0.5, 1.5, 0.8, 0.5, 0.7, 1.7, 0.8, 0.3, 0.3, 0.6, 0.4, 2.0, 0.2, 0.1, 0.2, 0.3, 0.2, 0.4, 0.1],
    "TOV": [3.2, 3.5, 3.4, 3.4, 3.7, 2.9, 3.4, 3.6, 2.8, 2.1, 2.0, 2.8, 2.3, 3.4, 2.0, 1.8, 3.1, 2.8, 2.8, 4.1],
    "FGA": [20.4, 18.2, 19.2, 20.3, 22.6, 21.0, 20.0, 16.8, 20.8, 16.4, 17.8, 18.1, 18.8, 12.8, 10.8, 8.9, 20.1, 18.4, 21.2, 19.3],
    "FG3A": [11.4, 6.9, 5.7, 2.1, 8.2, 8.4, 3.1, 3.0, 10.1, 2.8, 4.1, 6.8, 1.2, 4.1, 4.8, 4.2, 6.8, 5.9, 8.7, 7.2],
    "FTA": [5.1, 6.0, 7.1, 11.4, 7.8, 6.8, 11.7, 4.2, 6.8, 5.8, 4.2, 4.1, 7.8, 2.8, 1.2, 1.1, 4.8, 3.8, 4.2, 6.8],
    "MIN": [34.7, 35.5, 35.6, 32.1, 36.2, 36.9, 34.6, 33.7, 36.3, 33.4, 34.1, 34.6, 34.0, 30.2, 32.0, 31.2, 35.7, 36.0, 35.4, 34.0]
}

def create_sample_data():
    """Create sample data with calculated advanced stats."""
    df = pd.DataFrame(SAMPLE_DATA)
    
    # Calculate advanced stats
    df["TS_PCT"] = df["PTS"] / (2 * (df["FGA"] + 0.44 * df["FTA"]))
    df["EFG_PCT"] = (df["FGA"] * 0.5 + df["FG3A"] * 0.5) / df["FGA"]  # Simplified
    df["THREE_PAR"] = df["FG3A"] / df["FGA"]
    df["FTR"] = df["FTA"] / df["FGA"]
    
    # Add required columns
    df["PLAYER_ID"] = range(1, len(df) + 1)
    df["GP"] = 70  # Sample games played
    df["SEASON"] = DEFAULT_SEASON
    
    return df

@st.cache_data(show_spinner=True, ttl=60 * 60)
def fetch_season(season: str, per_mode: str) -> pd.DataFrame:
    """Fetch one season of league player stats and engineer features."""
    try:
        from nba_api.stats.endpoints import leaguedashplayerstats
        
        st.write("🔄 Fetching NBA data...")
        
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

        data = data[data["MIN"] >= MIN_MINUTES].copy()
        st.write(f"✅ Filtered to {len(data)} players with {MIN_MINUTES}+ minutes")

        # Calculate advanced stats
        data["TS_PCT"] = data["PTS"] / (2 * (data["FGA"] + 0.44 * data["FTA"]).replace(0, np.nan))
        data["EFG_PCT"] = (data["FGM"] + 0.5 * data["FG3M"]) / data["FGA"].replace(0, np.nan)
        data["THREE_PAR"] = data["FG3A"] / data["FGA"].replace(0, np.nan)
        data["FTR"] = data["FTA"] / data["FGA"].replace(0, np.nan)

        for c in ["TS_PCT", "EFG_PCT", "THREE_PAR", "FTR"]:
            data[c] = data[c].fillna(0)

        feats = FEATURE_ORDER
        keep_cols = ["PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "GP", "MIN"] + feats
        df = data[keep_cols].copy()
        df.insert(1, "SEASON", season)
        
        st.write(f"✅ Final dataset: {len(df)} players ready")
        return df
        
    except Exception as e:
        st.warning(f"⚠️ NBA API failed: {str(e)}")
        st.write("🔄 Using sample data instead...")
        
        # Use sample data as fallback
        df = create_sample_data()
        st.write(f"✅ Sample data loaded: {len(df)} players")
        return df

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
    
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("🔄 Load Real NBA Data", type="primary"):
            with st.spinner("Fetching NBA data... This may take a moment."):
                df = fetch_season(season, per_mode)
                
                if not df.empty:
                    df = df[df["MIN"] >= int(min_minutes)].reset_index(drop=True)
                    st.session_state.df = df
                    st.session_state.data_loaded = True
                    st.success(f"✅ Successfully loaded {len(df)} players for {season}!")
                    st.rerun()
                else:
                    st.error("❌ Failed to load data. Please try again.")
    
    with col2:
        if st.button("🎮 Use Sample Data"):
            st.write("🎮 Loading sample data...")
            try:
                df = create_sample_data()
                df = df[df["MIN"] >= int(min_minutes)].reset_index(drop=True)
                st.session_state.df = df
                st.session_state.data_loaded = True
                st.success(f"✅ Sample data loaded: {len(df)} players!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Sample data failed: {str(e)}")
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
