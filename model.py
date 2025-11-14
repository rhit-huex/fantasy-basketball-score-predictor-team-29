import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping

from data_preprocessing import preprocess_data
from nba_api.stats.static import players

def build_model(input_shape):
    model = Sequential()
    # LSTM layer with dropout for regularization
    model.add(Bidirectional(LSTM(32, activation='relu', input_shape=input_shape)))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2)) # Prevent overfitting
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1))  # Predicting a single continuous value (fantasy points)
    model.compile(optimizer='RMSprop', loss='mean_absolute_error', metrics=['mae', 'mse'])
    return model

def train_model(X, y, epochs=50, batch_size=64, validation_split=0.2):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=123)
    

    input_shape = (X_train.shape[1], X_train.shape[2])
    n_steps, n_feats = X_train.shape[1], X_train.shape[2]
    feature_scaler = StandardScaler()
    X_train_scaled = feature_scaler.fit_transform(X_train.reshape(-1, n_feats)).reshape(-1, n_steps, n_feats)
    X_test_scaled  = feature_scaler.transform(X_test.reshape(-1, n_feats)).reshape(-1, n_steps, n_feats)
    
    model = build_model(input_shape)
    model.summary()
    
    # Add early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    history = model.fit(
        X_train_scaled, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Evaluate the model on test data
    loss, mae, mse = model.evaluate(X_test_scaled, y_test, verbose=0)
    print("Test Loss: {:.4f}, Test MAE: {:.4f}, Test MSE: {:.4f}, Test RMSE: {:.4f}".format(loss, mae, mse, np.sqrt(mse)))
    
    return model, history, feature_scaler

def predict_next_game(model, player_gw_df, seq_length=5, feature_cols=[
    'MIN','FGM','FGA','FG3M','FG3A','FTM','FTA','OREB','DREB','AST','STL','BLK','TOV','PTS', 'home'], feature_scaler: StandardScaler | None = None):
    """
    For each player in the aggregated gameweek data, extract the most recent sequence of length 'seq_length'
    and use the trained model to predict the fantasy points for the next gameweek.
    Returns a dictionary mapping player_id to predicted fantasy points.
    
    Args:
        player_gw_df: pandas DataFrame with columns including 'Player_ID', 'GAME_DATE', and feature_cols
    """
    predictions = {}
    # Ensure data is sorted by player_id and date
    sorted_df = player_gw_df.sort_values(['Player_ID', 'GAME_DATE'])
    for player in sorted_df['Player_ID'].unique():
        player_data = sorted_df[sorted_df['Player_ID'] == player].reset_index(drop=True)
        if len(player_data) >= seq_length:
            seq = player_data.iloc[-seq_length:][feature_cols].values  # (seq_length, num_features)
            if feature_scaler is not None:
                n_feats = seq.shape[1]
                seq = feature_scaler.transform(seq.reshape(-1, n_feats)).reshape(1, seq_length, n_feats)
            else:
                seq = np.expand_dims(seq, axis=0)  # shape (1, seq_length, num_features)
            pred = model.predict(seq, verbose=0)
            predictions[player] = pred[0, 0]
    return predictions

def predict_player_game(model, player_gw_df, player_id, game_date, seq_length=5, feature_cols=[
    'MIN','FGM','FGA','FG3M','FG3A','FTM','FTA','OREB','DREB','AST','STL','BLK','TOV','PTS', 'home'], feature_scaler: StandardScaler | None = None):
    """
    Predict the fantasy points for the next game of a specific player.
    
    Args:
        model: Trained Keras model
        player_gw_df: pandas DataFrame with player gameweek data including salary
        player_id: ID of the player to predict
        game_date: Date for which to make prediction (YYYY-MM-DD)
        seq_length: Number of previous games to use for prediction
        feature_cols: List of feature column names
    """
    player_data = player_gw_df[player_gw_df['Player_ID'] == player_id].sort_values('GAME_DATE').reset_index(drop=True)
    
    if len(player_data) < seq_length:
        raise ValueError(f"Not enough data to predict for player ID {player_id}")
    
    # Get games before the target date
    historical_data = player_data[player_data['GAME_DATE'] < game_date][feature_cols]
    
    if len(historical_data) < seq_length:
        raise ValueError(f"Not enough historical data before {game_date} for player ID {player_id}. "
                        f"Need {seq_length} games, but only have {len(historical_data)}")
    
    # Get the last seq_length games before game_date
    seq = historical_data.iloc[-seq_length:][feature_cols].values
    if feature_scaler is not None:
        n_feats = seq.shape[1]
        seq = feature_scaler.transform(seq.reshape(-1, n_feats)).reshape(1, seq_length, n_feats)
    else:
        seq = np.expand_dims(seq, axis=0)  # shape (1, seq_length, num_features)
    pred = model.predict(seq, verbose=0)
    return pred[0, 0]

def get_best_lineup(model, player_gw_df, my_players, game_date, seq_length=5, feature_cols=[
    'MIN','FGM','FGA','FG3M','FG3A','FTM','FTA','OREB','DREB','AST','STL','BLK','TOV','PTS', 'home'], 
    feature_scaler: StandardScaler | None = None, lineup_size=5):
    """
    Predict fantasy points for all players and select the best lineup.
    
    Args:
        model: Trained Keras model
        player_gw_df: DataFrame with player gameweek data
        my_players: List of player names to choose from
        game_date: Date for predictions (YYYY-MM-DD)
        seq_length: Number of previous games to use
        feature_cols: List of feature column names
        feature_scaler: StandardScaler fitted on training data
        lineup_size: Number of players to select for lineup
    
    Returns:
        lineup: List of tuples (player_name, predicted_points)
        total_points: Sum of predicted points for the lineup
    """
    team_predictions = {}
    
    for player_name in my_players:
        try:
            player_data = player_gw_df[player_gw_df['full_name'] == player_name].sort_values('GAME_DATE').reset_index(drop=True)
            
            if player_data.empty:
                print(f"Warning: No data found for player {player_name}")
                continue
                
            player_id = player_data['Player_ID'].iloc[0]
            
            if len(player_data) < seq_length:
                print(f"Warning: Not enough data for {player_name} (need {seq_length} games, have {len(player_data)})")
                continue
            
            # Pass feature_scaler to predict_player_game
            predicted_points = predict_player_game(
                model, player_gw_df[feature_cols + ["Player_ID", "GAME_DATE"]], player_id=player_id, game_date=game_date, 
                seq_length=seq_length, feature_cols=feature_cols
            )
            team_predictions[player_name] = predicted_points
            
        except Exception as e:
            print(f"Error predicting for {player_name}: {str(e)}")
            continue
    
    # Sort by predicted points (descending) and select top lineup_size players
    sorted_predictions = sorted(team_predictions.items(), key=lambda x: x[1], reverse=True)
    lineup = sorted_predictions[:lineup_size]
    total_points = sum(points for _, points in lineup)
    
    return lineup, total_points


if __name__ == "__main__":
    # For testing purposes, if this file is run standalone, generate dummy data
    df, X, Y = preprocess_data(seq_length=5)  # Assuming this function exists in data_preprocessing.py    
    model, history, feature_scaler = train_model(X, Y, epochs=15, batch_size=16)

    model.save("fantasy_basketball_model.keras")
        