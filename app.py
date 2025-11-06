import gradio as gr
from nba_api.stats.static import players
import numpy as np
import pandas as pd

from model import train_model, build_model, predict_next_game
from data_preprocessing import preprocess_data



data,x,y = preprocess_data()
model, history = train_model(x,y, epochs=20)
predictions = predict_next_game(model,data, seq_length=5)

def gradio_predict(player_name):
    player_dict = players.find_players_by_full_name(player_name)
    if not player_dict:
        return "Player not found. Please check the name and try again."
    player_id = player_dict[0]['id']
    
    if player_id in predictions:
        return f"Predicted Fantasy Points for next game: {predictions[player_id]:.2f}"
    else:
        return "No prediction available for this player."
    
player_names = [p['full_name'] for p in players.get_players() if p['is_active']]

iface = gr.Interface(
    fn=gradio_predict,
    inputs=gr.Dropdown(choices=player_names, label="Select Player"),
    outputs="text",
    title="NBA Fantasy Points Predictor",
    description="Select an NBA player to see the predicted fantasy points for their next game."
)
iface.launch()


