import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from data_preprocessing import preprocess_data

def build_model(input_shape):
    model = Sequential()
    # LSTM layer with dropout for regularization
    model.add(LSTM(64, activation='tanh', input_shape=input_shape))
    model.add(Dropout(0.2)) # Prevent overfitting
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1))  # Predicting a single continuous value (fantasy points)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

def train_model(X, y, epochs=50, batch_size=32, validation_split=0.2):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
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
    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    print("Test Loss: {:.4f}, Test MAE: {:.4f}".format(loss, mae))
    
    # Optionally, perform K-Fold cross validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_losses = []
    cv_maes = []
    for train_index, test_index in kf.split(X):
        X_cv_train, X_cv_test = X[train_index], X[test_index]
        y_cv_train, y_cv_test = y[train_index], y[test_index]
        cv_model = build_model(input_shape)
        cv_model.fit(X_cv_train, y_cv_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split, verbose=0)
        loss_cv, mae_cv = cv_model.evaluate(X_cv_test, y_cv_test, verbose=0)
        cv_losses.append(loss_cv)
        cv_maes.append(mae_cv)
    print("Cross Validation Loss: {:.4f} ± {:.4f}".format(np.mean(cv_losses), np.std(cv_losses)))
    print("Cross Validation MAE: {:.4f} ± {:.4f}".format(np.mean(cv_maes), np.std(cv_maes)))
    
    return model, history

def predict_next_game(model, player_gw_df, seq_length=5, feature_cols=[
    'MIN','FGM','FGA','FG3M','FG3A','FTM','FTA','OREB','DREB','AST','STL','BLK','TOV','PTS']):
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

if __name__ == "__main__":
    # For testing purposes, if this file is run standalone, generate dummy data
    df, _, _, _, _ = preprocess_data()  # Assuming this function exists in data_preprocessing.py

    df.shape

    X_dummy = np.random.rand(100, 5, 14)  # 100 samples, sequence length of 5 games, 14 features
    y_dummy = np.random.rand(100)
    
    # Create dummy DataFrame for prediction testing
    feature_cols = ['MIN','FGM','FGA','FG3M','FG3A','FTM','FTA','OREB','DREB','AST','STL','BLK','TOV','PTS']
    X = df[feature_cols + ['Player_ID', 'GAME_DATE']]
    Y = df['fantasy_points_dk'].values
    
    model, history = train_model(X, Y, epochs=10)
    
    dummy_data = {
        'Player_ID': [1]*10 + [2]*10,
        'GAME_DATE': pd.date_range('2024-01-01', periods=20),
    }
    for col in feature_cols:
        dummy_data[col] = np.random.rand(20)
    
    dummy_df = pd.DataFrame(dummy_data)
    preds = predict_next_game(model, dummy_df)
    print(preds)