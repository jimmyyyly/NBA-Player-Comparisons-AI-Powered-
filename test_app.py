import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="NBA Player Comps (AI-powered)", layout="wide")
st.title("🏀 AI-Powered NBA Player Comparisons")

st.write("App is loading...")

# Test basic functionality
try:
    st.write("✅ Streamlit is working")
    
    # Test data loading
    st.write("🔄 Testing NBA API...")
    
    from nba_api.stats.endpoints import leaguedashplayerstats
    
    # Try to fetch a small amount of data
    data = leaguedashplayerstats.LeagueDashPlayerStats(
        season="2023-24",
        per_mode_detailed="PerGame",
        measure_type_detailed_defense="Base",
        season_type_all_star="Regular Season",
    ).get_data_frames()[0]
    
    st.write(f"✅ NBA API working - loaded {len(data)} players")
    st.write("First few players:")
    st.dataframe(data[['PLAYER_NAME', 'TEAM_ABBREVIATION', 'PTS']].head())
    
except Exception as e:
    st.error(f"❌ Error: {str(e)}")
    st.write("This might be why your deployed app isn't working.")

st.write("If you see this message, the basic app structure is working!")
