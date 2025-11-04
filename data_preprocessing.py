# Parse, sort, and clean data
from typing import Optional, Tuple
import pandas as pd, re
import numpy as np

def preprocess_data(location="nba_fantasy_points_2024_25_dk.csv"):
    # Load data
    df = pd.read_csv(location)
    
    # Parse dates and sort -> need to clean to put into model
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], format="%b %d, %Y")
    df = df.sort_values(["Player_ID","GAME_DATE"]).reset_index(drop=True)

    # Parse opponent team and home/away status 
    def parse_matchup(s):
        parts = re.split(r"\s+@|\s+vs\.?", s)
        return (parts[0].strip(), parts[1].strip()) if len(parts)==2 else (None, None)

    df[["TEAM_ABBR","OPP_ABBR"]] = df["MATCHUP"].apply(parse_matchup).apply(pd.Series)
    df["home"] = df["MATCHUP"].str.contains(r"\bvs\b|\bvs\.", case=False).astype(int)

    # # Remove any games with 0 minutes
    # df = df[df["MIN"] > 0].reset_index(drop=True)

    # Pick sequence features and targets
    seq_cols = [
        "MIN","FGM","FGA","FG3M","FG3A","FTM","FTA",
        "OREB", "DREB","AST","STL","BLK","TOV","PTS","home"
    ]
    target_col = "fantasy_points_dk"
    df = df.dropna(subset=[target_col])

    # Make integer ID for embeddings
    player2idx = {pid:i for i,pid in enumerate(df["Player_ID"].unique())}
    teams = sorted(set(df["TEAM_ABBR"].dropna().unique()) | set(df["OPP_ABBR"].dropna().unique()))
    team2idx = {t:i for i,t in enumerate(teams)}

    df["player_idx"] = df["Player_ID"].map(player2idx)
    df["team_idx"]   = df["TEAM_ABBR"].map(team2idx).fillna(0).astype(int)
    df["opp_idx"]    = df["OPP_ABBR"].map(team2idx).fillna(0).astype(int)

    # Clean up dataframe
    df = df[seq_cols + [target_col, "Player_ID", "team_idx", "opp_idx", "GAME_DATE"]]

    X, Y = create_sequences(df, seq_length=5, feature_cols=seq_cols, target_col=target_col)
    
    return df, X, Y


def create_sequences(df: pd.DataFrame, seq_length: int, 
                     feature_cols: list, 
                     target_col: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Creates sliding-window sequences from player gameweek data.
    Each sequence (shape [seq_length, num_features]) is paired with the target value from the next gameweek.
    """
    sequences = []
    targets = []
    if df is None:
        return None, None
    if 'GAME_DATE' not in df.columns:
        print("Column 'GAME_DATE' not found in player game data. Cannot create sequences.")
        return None, None
    df = df.sort_values(['Player_ID', 'GAME_DATE'])
    for player in df['Player_ID'].unique():
        player_data = df[df['Player_ID'] == player].reset_index(drop=True)
        if len(player_data) <= seq_length:
            continue
        data_array = player_data[feature_cols].values
        target_array = player_data[target_col].values
        for i in range(len(player_data) - seq_length):
            sequences.append(data_array[i:i+seq_length])
            targets.append(target_array[i+seq_length])
    return np.array(sequences), np.array(targets)

preprocess_data()