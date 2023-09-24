% Queries: Example queries to retrieve information
?-appears_in_game('Luigi', X).
?-protagonist('Bowser').
?-appears_in_game_on_platform('Mario', 'Nintendo Switch').
?-release_year(X, Y), Y > 2020.
?-appears_in_3d_game('Mario', X).
?-(appears_in_game('Hoot', X); appears_in_game('Dorrie', X)), (game_on_platform(X, 'Nintendo 64')), game_graphics(X, '3D').
