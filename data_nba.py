from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players


# player_dict = players.find_players_by_full_name("LeBron James")
# print(player_dict)

# log = playergamelog.PlayerGameLog(season='2024-25')
# df = log.get_data_frames()[0]
# print(df.head())

all_players = players.get_active_players()

for p in all_players[:5]:  # Example: first 5 players
    logs = playergamelog.PlayerGameLog(player_id=p['id'], season='2024-25')
    df = logs.get_data_frames()[0]
    print(p['full_name'], df[['GAME_DATE', 'PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV']].head())
    print(df.keys())