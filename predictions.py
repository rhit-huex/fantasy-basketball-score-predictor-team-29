import tensorflow as tf

from data_preprocessing import preprocess_data
from nba_api.stats.static import players
from model import predict_next_game, predict_player_game, get_best_lineup

if __name__ == "__main__":
    loaded_model = tf.keras.models.load_model('fantasy_basketball_model.keras')
    df, X, Y = preprocess_data()

    while True:
        input_action = input("Enter 'predict' to make a prediction or 'exit' to quit: ").strip().lower()
        if input_action == 'exit':
            break
        elif input_action == 'predict':
            player_name = input("Enter player full name: ")
            # Here you would add code to load player data and make predictions using loaded_model
            player_dict = players.find_players_by_full_name(player_name)
            if not player_dict:
                print("Player not found. Please try again\n")
                continue
            # Convert full name to player ID
            player_id = player_dict[0]['id']
            predictions = predict_next_game(loaded_model, df, seq_length=5)
            if player_id in predictions:
                print(f"Predicted number of fantasy points next game for {player_name}: {predictions[player_id]:.2f}\n")
            else:
                print(f"No prediction available for {player_name}\n")
        elif input_action == 'lineup':
            budget = int(input("Enter your budget for the lineup: "))
            my_players = input("Enter player IDs separated by commas: ").split(',')
            my_players = [int(pid.strip()) for pid in my_players]
            game_date = input("Enter game date (YYYY-MM-DD): ")
            lineup, total_cost, total_points = get_best_lineup(loaded_model, df, my_players, game_date, budget=budget)
            print("Best Lineup:")
            for player_id, predicted_points, player_salary in lineup:
                print(f"Player ID: {player_id}, Predicted Points: {predicted_points:.2f}, Salary: {player_salary}")
            print(f"Total Cost: {total_cost}, Total Predicted Points: {total_points:.2f}\n")
        elif input_action == 'exit':
            break
        elif input_action == "player at date" or input_action == "pd":
            player_name = input("Enter player full name: ")
            game_date = input("Enter game date (YYYY-MM-DD): ")
            player_dict = players.find_players_by_full_name(player_name)
            if not player_dict:
                print("Player not found. Please try again\n")
                continue
            player_id = player_dict[0]['id']
            prediction = predict_player_game(loaded_model, df, player_id, game_date, seq_length=5)
            if prediction is not None:
                print(f"Predicted fantasy points for {player_name} on {game_date}: {prediction:.2f}\n")
            else:
                print(f"No prediction available for {player_name} on {game_date}\n")

            