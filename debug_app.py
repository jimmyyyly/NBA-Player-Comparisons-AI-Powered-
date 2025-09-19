import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Debug NBA App", layout="wide")
st.title("🐛 Debug NBA App")

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'df' not in st.session_state:
    st.session_state.df = pd.DataFrame()

st.write("**Session State:**")
st.write(f"- data_loaded: {st.session_state.data_loaded}")
st.write(f"- df empty: {st.session_state.df.empty}")

# Sample data
SAMPLE_DATA = {
    "PLAYER_NAME": [
        "Stephen Curry", "LeBron James", "Kevin Durant", "Giannis Antetokounmpo", 
        "Luka Doncic", "Jayson Tatum", "Joel Embiid", "Nikola Jokic"
    ],
    "TEAM_ABBREVIATION": ["GSW", "LAL", "PHX", "MIL", "DAL", "BOS", "PHI", "DEN"],
    "PTS": [29.4, 25.0, 28.2, 31.1, 32.4, 30.1, 33.1, 24.5],
    "AST": [6.3, 6.9, 5.0, 5.7, 8.0, 4.6, 3.2, 9.8],
    "REB": [6.1, 7.4, 6.7, 11.8, 8.2, 8.8, 10.2, 11.8],
    "STL": [1.0, 1.3, 0.9, 1.2, 1.4, 1.1, 1.0, 1.3],
    "BLK": [0.4, 0.5, 1.5, 0.8, 0.5, 0.7, 1.7, 0.8],
    "TOV": [3.2, 3.5, 3.4, 3.4, 3.7, 2.9, 3.4, 3.6],
    "FGA": [20.4, 18.2, 19.2, 20.3, 22.6, 21.0, 20.0, 16.8],
    "FG3A": [11.4, 6.9, 5.7, 2.1, 8.2, 8.4, 3.1, 3.0],
    "FTA": [5.1, 6.0, 7.1, 11.4, 7.8, 6.8, 11.7, 4.2],
    "MIN": [34.7, 35.5, 35.6, 32.1, 36.2, 36.9, 34.6, 33.7]
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
    df["SEASON"] = "2023-24"
    
    return df

# Buttons
col1, col2 = st.columns([1, 1])

with col1:
    if st.button("🔄 Load Real NBA Data", type="primary"):
        st.write("🔄 Trying to load real data...")
        try:
            from nba_api.stats.endpoints import leaguedashplayerstats
            data = leaguedashplayerstats.LeagueDashPlayerStats(
                season="2023-24",
                per_mode_detailed="PerGame",
                measure_type_detailed_defense="Base",
                season_type_all_star="Regular Season",
            ).get_data_frames()[0]
            st.write(f"✅ Real data loaded: {len(data)} players")
        except Exception as e:
            st.error(f"❌ Real data failed: {str(e)}")

with col2:
    if st.button("🎮 Use Sample Data"):
        st.write("🎮 Loading sample data...")
        try:
            df = create_sample_data()
            st.session_state.df = df
            st.session_state.data_loaded = True
            st.write(f"✅ Sample data loaded: {len(df)} players")
            st.write("**Sample data preview:**")
            st.dataframe(df[['PLAYER_NAME', 'TEAM_ABBREVIATION', 'PTS', 'AST', 'REB']].head())
            st.rerun()
        except Exception as e:
            st.error(f"❌ Sample data failed: {str(e)}")

# Show current data
if st.session_state.data_loaded and not st.session_state.df.empty:
    st.success("✅ Data is loaded!")
    st.write("**Current data:**")
    st.dataframe(st.session_state.df[['PLAYER_NAME', 'TEAM_ABBREVIATION', 'PTS', 'AST', 'REB']])
    
    if st.button("🔄 Reset Data"):
        st.session_state.data_loaded = False
        st.session_state.df = pd.DataFrame()
        st.rerun()
else:
    st.info("👆 Click a button to load data")

# Debug info
with st.expander("Debug Info"):
    st.write("**Session State:**")
    st.write(st.session_state)
    st.write("**DataFrame info:**")
    if not st.session_state.df.empty:
        st.write(f"Shape: {st.session_state.df.shape}")
        st.write("Columns:", list(st.session_state.df.columns))
    else:
        st.write("DataFrame is empty")
