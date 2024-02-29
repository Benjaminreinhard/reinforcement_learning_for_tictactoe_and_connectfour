'''Demonstration of a game

Run this file to view a game, you will receive constructions in the console.

	The variable 'index' allows you to specify the directory from which
	the trained players should be taken from. E.g. 'index = 1' will get trained
	players from the directory './training_data/training_1/'
'''

import pickle

from game import TicTacToe
from players import HumanPlayer, RandomPlayer, PrunPlayer
from tools import play


# Initialize game
tictactoe = TicTacToe()


# Initialize players that do not need training
humanplayer = HumanPlayer()
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

qplayer.epsilon = 0
tdplayer.epsilon = 0
deepplayer.epsilon = 0

players = [humanplayer, randomplayer, prunplayer, qplayer, tdplayer, deepplayer]


# Start Demo
print('Demo of a game of Tic Tac Toe')
print()

while True:
	in_ = input('To select player x, enter a name of the list [Human, Random, Prun, Q, TD, Deep] (Enter Human, if you want to be player x): ')
	while True:
		if in_ not in [player.name for player in players]:
			in_ = input('Not a name in the list, please try again: ')
		else:
			break
	
	for player in players:
		if player.name == in_:
			player_x = player
	
	print()
	
	in_ = input('Now select player o, by entering a name of the list [Human, Random, Prun, Q, TD, Deep] (Enter Human, if you want to be player o): ')
	while True:
		if in_ not in [player.name for player in players]:
			in_ = input('Not a name in the list, please try again: ')
		else:
			break

	for player in players:
		if player.name == in_:
			player_o = player

	print()

	in_ = input('Should the first action be random? (y/n)')
	while True:
		if in_ not in ['y', 'n']:
			in_ = input('Not a valid input, please try again: ')
		else:
			break

	first_action_random = True if in_ == 'y' else False

	print()

	print('Game starts:')

	print()

	play(tictactoe, player_x, player_o, first_action_random=first_action_random, render=True)

	in_ = input('Do you want to restart the game? (y/n)')
	while True:
		if in_ not in ['y', 'n']:
			in_ = input('Not a valid input, please try again: ')
		else:
			break

	if in_ == 'y':
		print()
	else:
		break