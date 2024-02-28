# Reinforcement learning for Tic Tac Toe and Connect Four

This project contains a set up of the games Tic Tac Toe and Connect Four. It also contains various players that are based on Tree search methods or Reinforcement learning methods. You can train the players that require training and you can evaluate their performance by letting them play against each other. You can also run a demo of a game between two desired players.

## Structure

There are two folders, one for each game. Both contain the following files and directories:

- ''game.py'' <-- contains the structure of the game
- players.py <-- contains the list of players
- training.py <-- starts a training with parameres that one can specify
- evaluation.py <-- evaluates the performance of specified players by letting them play against each other
- demo.py <-- allows you to render a game between to players
- neuralnetwork.py <-- contains the neural network for the so-called deep player
- game_tree_info.py <-- gives you information about the game try
- tools.py <-- contains functions that are used in training.py and evaluation.py
- training_data/ <-- contains the repositories that are created once training.py is executed
- evaluation_data/ <-- contains the repositories that are created once evaluation.py is executed
