# NBA Player Comparisons (AI-powered)

A Streamlit web application that uses machine learning to find similar NBA players based on their statistical profiles.

## Features

- **Player Similarity Analysis**: Find players with similar playing styles using cosine similarity
- **Customizable Weights**: Adjust feature importance to focus on different aspects of play
- **Interactive Visualizations**: 2D PCA plots showing player relationships
- **Fuzzy Search**: Easy player lookup with intelligent name matching
- **Multiple Seasons**: Compare players across different seasons
- **Per-Mode Analysis**: Analyze per-game, per-36, or per-100-possessions stats

## Installation

1. Make sure you have Python 3.8+ installed
2. Install the required dependencies:
   ```bash
   py -m pip install -r requirements.txt
   ```

## Usage

1. Run the application:
   ```bash
   py -m streamlit run app.py
   ```

2. Open your browser and go to `http://localhost:8501`

3. Use the sidebar to:
   - Select a season (format: YYYY-YY)
   - Choose per-mode analysis
   - Adjust feature weights
   - Set minimum minutes filter
   - Choose number of comparisons

4. Select a player using the dropdown or type a name for fuzzy search

## Features Explained

### Statistical Features
- **TS_PCT**: True shooting percentage (overall scoring efficiency)
- **EFG_PCT**: Effective field goal percentage (weights 3PT makes as 1.5 shots)
- **THREE_PAR**: 3-point attempt rate (3PA / FGA)
- **FTR**: Free throw rate (FTA / FGA)
- **FGA/FG3A/FTA**: Shot volume profile
- **PTS/AST/REB/STL/BLK/TOV**: Box score impact

### How It Works
1. Fetches NBA player statistics from the official NBA API
2. Calculates advanced metrics and normalizes data
3. Applies user-defined weights to features
4. Uses cosine similarity to find the most similar players
5. Visualizes results in an interactive 2D plot using PCA

## Troubleshooting

If you encounter any issues:
1. Make sure all dependencies are installed: `py -m pip install -r requirements.txt`
2. Check your internet connection (app fetches data from NBA API)
3. Try a different season if data seems incomplete
4. Adjust the minimum minutes filter if you're not seeing enough players

## Requirements

- Python 3.8+
- All dependencies listed in `requirements.txt`
- Internet connection for NBA API access
