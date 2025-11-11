import gradio as gr
import pandas as pd
from nba_api.stats.static import players
from model import train_model, build_model, predict_next_game
from data_preprocessing import preprocess_data
import numpy as np

PLAYER_ID_COL = "Player_ID"
GAME_DATE_COL = "GAME_DATE"
FEATURE_COLS = ['MIN','FGM','FGA','FG3M','FG3A','FTM','FTA','OREB','DREB','AST','STL','BLK','TOV','PTS','home']
SEQ_LENGTH = 5

df, X, Y = preprocess_data()
model, history = train_model(X, Y, epochs=20)

player_names = [p['full_name'] for p in players.get_players() if p.get('is_active')]

def predict_from_date(player_name, date_str):
    player_list = players.find_players_by_full_name(player_name)
    if not player_list:
        return "Player not found. Check the name."
    player_id = player_list[0]['id']
    try:
        picked = pd.to_datetime(date_str, format="%m/%d/%Y", errors='coerce')
        if pd.isna(picked):
            picked = pd.to_datetime(date_str, errors='coerce')
        if pd.isna(picked):
            return "Couldn't parse date. Use MM/DD/YYYY."
        picked_date = picked.date()
    except Exception:
        return "Couldn't parse date. Use MM/DD/YYYY."
    if PLAYER_ID_COL not in df.columns or GAME_DATE_COL not in df.columns:
        return "Dataset missing required columns."
    df_dates = pd.to_datetime(df[GAME_DATE_COL], errors='coerce').dt.date
    player_rows = df[(df[PLAYER_ID_COL] == player_id) & (df_dates <= picked_date)].sort_values(GAME_DATE_COL)
    if len(player_rows) < SEQ_LENGTH:
        return f"Not enough games before {picked_date.strftime('%m/%d/%Y')} (need {SEQ_LENGTH})."
    seq_df = player_rows.iloc[-SEQ_LENGTH:][FEATURE_COLS].fillna(0).values
    seq = np.expand_dims(seq_df, axis=0)
    pred = model.predict(seq, verbose=0)
    return f"Predicted fantasy points (based on last {SEQ_LENGTH} games up to {picked_date.strftime('%m/%d/%Y')}): {float(pred[0,0]):.2f}"

iface = gr.Interface(
    fn=predict_from_date,
    inputs=[
        gr.Dropdown(choices=player_names, label="Select Player"),
        gr.Textbox(label="Enter date (MM/DD/YYYY)", placeholder="MM/DD/YYYY")
    ],
    outputs="text",
    title="NBA Fantasy Points Predictor (Date-based)",
    description="Enter a date in MM/DD/YYYY. The model will use the player's last 5 games up to that date to produce a prediction."
)

if __name__ == "__main__":
    iface.launch(share=True)
