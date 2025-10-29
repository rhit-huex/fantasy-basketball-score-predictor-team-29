# Parse, sort, and clean data
import pandas as pd, re
import numpy as np

def preprocess_data(location="nba_fantasy_points_2024_25_dk.csv"):
    df = pd.read_csv(location)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], format="%d-%b-%y")
    df = df.sort_values(["Player_ID","GAME_DATE"]).reset_index(drop=True)

    # Parse opponent team and home/away status
    def parse_matchup(s):
        parts = re.split(r"\s+@|\s+vs\.?", s)
        return (parts[0].strip(), parts[1].strip()) if len(parts)==2 else (None, None)

    df[["TEAM_ABBR","OPP_ABBR"]] = df["MATCHUP"].apply(parse_matchup).apply(pd.Series)
    df["home"] = df["MATCHUP"].str.contains(r"\bvs\b|\bvs\.", case=False).astype(int)

    # Remove any games with 0 minutes
    df = df[df["MIN"] > 0].reset_index(drop=True)

    # Pick sequence features and targets
    seq_cols = [
        "MIN","FGM","FGA","FG3M","FG3A","FTM","FTA",
        "REB","AST","STL","BLK","TOV","PTS","home"
    ]
    target_col = "fantasy_points_dk"
    df = df.dropna(subset=[target_col]).copy()

    # Make integer ID for embeddings
    player2idx = {pid:i for i,pid in enumerate(df["Player_ID"].unique())}
    teams = sorted(set(df["TEAM_ABBR"].dropna().unique()) | set(df["OPP_ABBR"].dropna().unique()))
    team2idx = {t:i for i,t in enumerate(teams)}

    df["player_idx"] = df["Player_ID"].map(player2idx)
    df["team_idx"]   = df["TEAM_ABBR"].map(team2idx).fillna(0).astype(int)
    df["opp_idx"]    = df["OPP_ABBR"].map(team2idx).fillna(0).astype(int)

