'''Training

This file can be run to start a training for a QPlayer, TDPlayer and DeepPlayer.
Below you can specify the parameters for each player.

In the end it will create a new directory './training_data/training_#' where '#' is
a unique index to a training session. It will contain each player saved as a pickle file,
a pickle file of a dictionary with all the information of the training and a
visualization of the loss of the DeepPlayer.
'''


import os
import pickle
import time

from game import TicTacToe
from players import QPlayer, TDPlayer, DeepPlayer
from tools import visualize_loss


# Initialize game
tictactoe = TicTacToe()


# Initialize Q Player
alpha_q = 0.3
epsilon_q = 0.2
gamma_q = 0.9
qplayer = QPlayer(alpha_q, epsilon_q, gamma_q)


# Initialize TD Player
alpha_td = 0.3
epsilon_td = 0.2
gamma_td = 0.9
lanbda_td = 0.8
tdplayer = TDPlayer(alpha_td, epsilon_td, gamma_td, lanbda_td)


# Initialize Deep Player
alpha_deep = 0.15
epsilon_deep = 0.2
gamma_deep = 0.9
num_central_layers = 1
central_layer_dim = 58
deepplayer = DeepPlayer(alpha_deep, epsilon_deep, gamma_deep, num_central_layers, central_layer_dim)


# Training parameters
render = True

number_of_games_q = 200000
number_of_games_td = 100000
number_of_games_deep = 70000

decrease_parameters = {
	'decrease_alpha' : True,
	'alpha_decrease_factor' : 0.8,
	'decrease_epsilon' : False,
	'epsilon_decrease_factor' : 0.9,
	'decrease_after' : number_of_games_deep/10
}


# Training
time1 = time.time()
qplayer.train(tictactoe, number_of_games_q, render)
time2 = time.time()
tdplayer.train(tictactoe, number_of_games_td, render)
time3 = time.time()
loss = deepplayer.train(tictactoe, number_of_games_deep, decrease_parameters, render)
time4 = time.time()


# Training info
training_time_q = time2 - time1
training_time_td = time3 - time2
training_time_deep = time4 - time3
length_qtable = len(qplayer.Q1) + len(qplayer.Q2)
length_vtable = len(tdplayer.V)


#Print training info
print('Duration of Q Player training:', round(training_time_q, 0), 'seconds')
print('Duration of TD Player training:', round(training_time_td, 0), 'seconds')
print('Duration of Deep Player training:', round(training_time_deep, 0), 'seconds')
print()
print('Summed length of tables Q1, Q2 of Q Player: ', length_qtable)
print('Length of table V of TD Player: ', length_vtable)
print()


# Make new directory
dir_path = './training_data/'
if not os.path.isdir(dir_path):
	os.mkdir(dir_path)
dir_path +=  '/training_'
index = 1
while True:
	path = dir_path + str(index)
	if os.path.isdir(path):
		index += 1
	else:
		break
dir_path += str(index)
os.mkdir(dir_path)


# Save players as pickle files
with open(dir_path + '/qplayer.pickle', 'wb') as file:
	pickle.dump(qplayer, file)
with open(dir_path + '/tdplayer.pickle', 'wb') as file:
	pickle.dump(tdplayer, file)
with open(dir_path + '/deepplayer.pickle', 'wb') as file:
	pickle.dump(deepplayer, file)


# Save loss in a pickle file
with open(dir_path + '/loss_of_deepplayer.pickle', 'wb') as file:
	pickle.dump(loss, file)


# Visualize loss
visualize_loss(dir_path + '/loss_of_deepplayer.pickle', number_of_games_deep, num_central_layers, central_layer_dim)


# Training info
training_info = {
	'alpha_q' : alpha_q,
	'epsilon_q' : epsilon_q,
	'gamma_q' : gamma_q,
	'number_of_games_q' : number_of_games_q,
	'training_time_q' : training_time_q,
	'length_qtable' : length_qtable,

	'alpha_td' : alpha_td,
	'epsilon_td' : epsilon_td,
	'gamma_td' : gamma_td,
	'lanbda_td' : lanbda_td,
	'number_of_games_td' : number_of_games_td,
	'training_time_td' : training_time_td,
	'length_vtable' : length_vtable,

	'alpha_deep' : alpha_deep,
	'epsilon_deep' : epsilon_deep,
	'gamma_deep' : gamma_deep,
	'num_central_layers' : num_central_layers,
	'central_layer_dim' : central_layer_dim,
	'number_of_games_deep' : number_of_games_deep,
	'decrease_parameters' : decrease_parameters,
	'training_time_deep' : training_time_deep,
}


# Save training info as pickle file
with open(dir_path + '/training_info.pickle', 'wb') as file:
	pickle.dump(training_info, file)