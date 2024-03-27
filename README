# Reinforcement learning for Tic Tac Toe and Connect Four

This project contains a set up of the games Tic Tac Toe and Connect Four. It also contains various players that are based on Reinforcement learning or Tree search methods methods. You can train the players that require training and you can evaluate their performance by letting them play against each other. You can also run a demo of a game between two desired players.

The RL players are the following:

- **The Q Player**: only for Tic Tac Toe: uses the Q-Learning method
- **The TD Player**: only for Tic Tac Toe: uses the TD($\lambda$) method
- **The Deep Player**: for both games: uses a Monte Carlo method to train a Neural network

The other players are the following:

- **The Prun Player**: for both games: uses the Alpha-beta pruning method. If you use it for Connect Four, you have to specif a depth
- **The Random Player**: for both games: plays randomly
- **The Chain Player**: only for Connect Four: you can pick 'offensive' or 'defensive'. The 'offensive' one always extends its longest chain on the grid and the 'defensive' one alway blocks the longest chain of the opponent.
- **The Human Player**: for both games: allows you to play

In `pseudocodes.pdf` you find pseudo-codes of the methods that were used for the RL players.

## Structure

There are two folders `./tictactoe/` and `./connectfour/`. Both contain the following files and directories:

- `game.py` <-- contains the structure of the game
- `players.py` <-- contains the players
- `training.py` <-- starts a training with parameres that you can specify
- `evaluation.py` <-- evaluates the performance of specified players by letting them play against each other
- `demo.py` <-- allows you to render a game between to players
- `neuralnetwork.py` <-- contains the neural network for the Deep Player
- `game_tree_info.py` <-- gives you information about the game tree
- `tools.py` <-- contains functions that are used in training.py and evaluation.py
- `training_data/` <-- contains the repositories that are created once training.py is executed
- `evaluation_data/` <-- contains the repositories that are created once evaluation.py is executed

## Installation

Download this repo to your local device and check if you have installed the necessary packages. To do so, run:

```
git clone https://github.com/Benjaminreinhard/reinforcement_learning_for_tictactoe_and_connectfour
cd reinforcement_learning_for_tictactoe_and_connectfour
pip3 install -r requirements.txt
```

## Getting started

Choose one of the games. To choose Tic Tac Toe, run:

```
cd tictactoe
```

To choose Connect Four, run:

```
cd connectfour
```

### Training
To train an agent, run:

```
python training.py
```

If you have chosen Tic Tac Toe, then it will train the Q, TD and Deep Player. If you have chosen Connect Four, then it will only train the Deep Player. Each time the code is run, a new directory `training_data/training_#` is created with `#` being a new index unique to the training session. Each training procedure of the players has many parameters and they can be adjusted in the file itself.

### Evaluation
To evaluate the performance, run:

```
python evaluation.py
```

This will evaluate the performance of the players by letting them play against each other. In the file itself you can determine which players should be evaluated and how many games per match up should be played. Each time this code is run, a new directory `evaluation_data/evaluation_#_##` is created with `#` being the index of the trained players that are contained in `training_data/training_#` and `##` being a new index unique to to the evaluation session.

### Demo
To view a demonstation of a game, run:

```
python demo.py
```

When you run it, you will be asked to pick the first and second player. If you want to play yourself, you can choose 'Human'. Clear instructions will be given to you, once you run it.

### Game tree info
To get information on the game tree, run:

```
python game_tree_info.py
```

It will tell you how many nodes it contains, how many terminal states there are and how many board configurations there are. Note that for Connect Four you can only do this up to a specified depth due to the large size of the tree. The parameters can be adjusted in the file itself.
