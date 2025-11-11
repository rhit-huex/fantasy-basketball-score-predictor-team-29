import numpy as np
import pandas as pd
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
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2)) # Prevent overfitting
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1))  # Predicting a single continuous value (fantasy points)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae', 'mse'])
    return model

def train_model(X, y, epochs=50, batch_size=32, validation_split=0.2):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
    
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_model(input_shape)
    model.summary()
    
    # Add early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Evaluate the model on test data
    loss, mae, mse = model.evaluate(X_test, y_test, verbose=0)
    print("Test Loss: {:.4f}, Test MAE: {:.4f}, Test MSE: {:.4f}, Test RMSE: {:.4f}".format(loss, mae, mse, np.sqrt(mse)))
    
    # Perform K-Fold cross validation
    # kf = KFold(n_splits=5, shuffle=True, random_state=42)
    # cv_losses = []
    # cv_maes = []
    # for train_index, test_index in kf.split(X):
    #     X_cv_train, X_cv_test = X[train_index], X[test_index]
    #     y_cv_train, y_cv_test = y[train_index], y[test_index]
    #     cv_model = build_model(input_shape)
    #     cv_model.fit(X_cv_train, y_cv_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split, verbose=0)
    #     loss_cv, mae_cv = cv_model.evaluate(X_cv_test, y_cv_test, verbose=0)
    #     cv_losses.append(loss_cv)
    #     cv_maes.append(mae_cv)
    
    # print("Cross Validation Loss: {:.4f} ± {:.4f}".format(np.mean(cv_losses), np.std(cv_losses)))
    # print("Cross Validation MAE: {:.4f} ± {:.4f}".format(np.mean(cv_maes), np.std(cv_maes)))
    
    return model, history

def predict_next_game(model, player_gw_df, seq_length=5, feature_cols=[
    'MIN','FGM','FGA','FG3M','FG3A','FTM','FTA','OREB','DREB','AST','STL','BLK','TOV','PTS', 'home']):
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
            seq = player_data.iloc[-seq_length:][feature_cols].values
            seq = np.expand_dims(seq, axis=0)  # shape (1, seq_length, num_features)
            pred = model.predict(seq, verbose=0)
            predictions[player] = pred[0, 0]
    return predictions

def predict_player_game(model, player_gw_df, player_id, game_date, seq_length=5, feature_cols=[
    'MIN','FGM','FGA','FG3M','FG3A','FTM','FTA','OREB','DREB','AST','STL','BLK','TOV','PTS', 'home']):
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
    historical_data = player_data[player_data['GAME_DATE'] < game_date]
    
    if len(historical_data) < seq_length:
        raise ValueError(f"Not enough historical data before {game_date} for player ID {player_id}. "
                        f"Need {seq_length} games, but only have {len(historical_data)}")
    
    # Get the last seq_length games before game_date
    seq = historical_data.iloc[-seq_length:][feature_cols].values
    seq = np.expand_dims(seq, axis=0)  # shape (1, seq_length, num_features)
    pred = model.predict(seq, verbose=0)
    return pred[0, 0]

def get_best_lineup(model, player_gw_df, my_players, game_date, budget=50000, seq_length=5, feature_cols=[
    'MIN','FGM','FGA','FG3M','FG3A','FTM','FTA','OREB','DREB','AST','STL','BLK','TOV','PTS', 'home'], 
    salary_col='SALARY'):
    """
    Predict fantasy points for all players and select the best lineup within the given budget.
    
    Args:
        model: Trained Keras model
        player_gw_df: pandas DataFrame with player gameweek data including salary
        budget: Total budget for the lineup
    """

    team_predictions = {}
    for player_id in my_players:
        player_data = player_gw_df[player_gw_df['Player_ID'] == player_id].sort_values('GAME_DATE').reset_index(drop=True)
        if len(player_data) < seq_length:
            raise ValueError(f"Not enough data to predict for player ID {player_id}")
        team_predictions[player_id] = predict_player_game(model, player_gw_df, player_id=player_id, game_date=game_date, seq_length=seq_length)
    
    player_salaries = player_gw_df[['Player_ID', salary_col]].drop_duplicates().set_index('Player_ID')
    lineup = []
    total_cost = 0
    total_points = 0
    
    for player_id, predicted_points in sorted(predictions.items(), key=lambda x: x[1], reverse=True):
        player_salary = player_salaries.loc[player_id][salary_col]
        if total_cost + player_salary <= budget:
            lineup.append((player_id, predicted_points, player_salary))
            total_cost += player_salary
            total_points += predicted_points
    
    return lineup, total_cost, total_points


if __name__ == "__main__":
    # For testing purposes, if this file is run standalone, generate dummy data
    df, X, Y = preprocess_data(seq_length=5)  # Assuming this function exists in data_preprocessing.py    
    model, history = train_model(X, Y, epochs=15, batch_size=16)
    predictions = predict_next_game(model, df, seq_length=5)

    model.save("fantasy_basketball_model.keras")
    
    # # Make infinite loop to predict next game for a player that user inputs the name of
    # while True:
    #     player_name = input("Enter player full name (or 'exit' to quit): ")
    #     if player_name.lower() == 'exit':
    #         break
    #     # Find player by their full anme
    #     player_dict = players.find_players_by_full_name(player_name)
    #     if not player_dict:
    #         print("Player not found. Please try again\n")
    #         continue
    #     # Convert full name to player ID
    #     player_id = player_dict[0]['id']
    #     if player_id in predictions:
    #         print(f"Predicted number of fantasy points next game for {player_name}: {predictions[player_id]:.2f}\n")
    #     else:
    #         print(f"No prediction available for {player_name}\n")
        