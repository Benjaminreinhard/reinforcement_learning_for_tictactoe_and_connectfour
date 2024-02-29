'''Evaluation

This file can be run to start an evaluation.
Every player plays against every other player for a specified number of times.
You can specify the number with the variable 'games_per_pair'.

	You can specify which trained players you want to evaluate
	with the variable 'index'. It denotes the number of the directory where the
	trained players are saved, e.g. choosing 'index = 1' will evaluate
	the performance of the players in directory './training_data/training_1/'

Every time this file is run, it creates an new directory './evaluation_data/evaluation_#_##'
where '#' stands for the index you chose and '##' is the index unique to every evaluation session.
'''


import pickle
import os

from game import TicTacToe
from players import RandomPlayer, PrunPlayer
from tools import evaluation, visualize_scores


# Initialize game
tictactoe = TicTacToe()


# Initialize players that do not need training
randomplayer = RandomPlayer()
prunplayer = PrunPlayer()


# Choose an index from which forlder the trained players should be selected from
index = 1


# Load trained players
dir_path = './training_data/training_' + str(index)
with open(dir_path + '/qplayer.pickle', 'rb') as file:
	qplayer = pickle.load(file)
with open(dir_path + '/tdplayer.pickle', 'rb') as file:
	tdplayer = pickle.load(file)
with open(dir_path + '/deepplayer.pickle', 'rb') as file:
	deepplayer = pickle.load(file)


# Declare epsilon for each trained player
qplayer.epsilon = 0
tdplayer.epsilon = 0
deepplayer.epsilon = 0


# Evaluation parameters
players = [randomplayer, prunplayer, qplayer, tdplayer, deepplayer]
games_per_pair = 100
first_action_random = True


# Evaluation
scores = evaluation(tictactoe, players, games_per_pair, first_action_random)


# Create directory to save scores
dir_path = './evaluation_data/'
if not os.path.isdir(dir_path):
	os.mkdir(dir_path)
dir_path +=  '/evaluation_' + str(index) + '_'
index2 = 1
while True:
	path = dir_path + str(index2)
	if os.path.isdir(path):
		index2 += 1
	else:
		break
dir_path = path
os.mkdir(dir_path)


# Save scores
with open(dir_path + '/scores.pickle', 'wb') as file:
	pickle.dump(scores, file)


# Visualize scores
visualize_scores(dir_path + '/scores.pickle')