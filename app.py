import gradio as gr
from nba_api.stats.static import players
import numpy as np
import pandas as pd
from datetime import datetime

from model import train_model, build_model, predict_player_game
from data_preprocessing import preprocess_data

# Load and train (unchanged)
data, x, y = preprocess_data()
model, history = train_model(x, y, epochs=20)

# Make names the ones you suggested (so your exact call works)
SEQ_LENGTH = 5
df = data  # alias so predict_player_game can accept 'df' as in your suggested call

def gradio_predict(player_name, date_str):
    # Validate date format
    try:
        game_date = datetime.strptime(date_str, "%m/%d/%Y").strftime("%Y-%m-%d")
    except ValueError:
        return "Invalid date format. Please use MM/DD/YYYY format (e.g., 03/15/2024)."
    
    # Find player
    player_dict = players.find_players_by_full_name(player_name)
    if not player_dict:
        return "Player not found. Please check the name and try again."
    player_id = player_dict[0]['id']
    
    # Get player data and show the last 5 games being used
    player_data = data[data['Player_ID'] == player_id].sort_values('GAME_DATE').reset_index(drop=True)
    historical_data = player_data[player_data['GAME_DATE'] < game_date]
    
    if len(historical_data) < SEQ_LENGTH:
        return f"Not enough historical data before {date_str}. Player has only {len(historical_data)} games before this date."
    
    last_5_games = historical_data.iloc[-SEQ_LENGTH:]
    games_info = "\n".join([f"  {row['GAME_DATE']}: {row['PTS']} pts" for _, row in last_5_games.iterrows()])
    
    # Predict for specific date using the suggested call signature
    try:
        # Using your suggested variable names: df, picked_date, SEQ_LENGTH
        picked_date = game_date
        pred = predict_player_game(model, df, player_id, picked_date, seq_length=SEQ_LENGTH)

        return f"Predicted Fantasy Points for {player_name} on {date_str}: {pred:.2f}\n\nBased on last {SEQ_LENGTH} games:\n{games_info}"
    except ValueError as e:
        return f"Error: {str(e)}"
    except Exception as e:
        return f"An error occurred: {str(e)}"

player_names = [p['full_name'] for p in players.get_players() if p['is_active']]

iface = gr.Interface(
    fn=gradio_predict,
    inputs=[
        gr.Dropdown(choices=player_names, label="Select Player"),
        gr.Textbox(label="Game Date (MM/DD/YYYY)", placeholder="03/15/2024")
    ],
    outputs="text",
    title="NBA Fantasy Points Predictor",
    description="Select an NBA player and enter a date to see the predicted fantasy points based on their last 5 games before that date."
)

iface.launch(share=True)
