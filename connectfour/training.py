'''Training

This file can be run to start a training for a DeepPlayer.
Below you can specify the parameters of the player.

In the end it will create a new directory './training_data/training_#' where '#' is
a unique index to a training session. It will contain the player saved as a pickle file,
a pickle file of a dictionary with all the information of the training and a
visualization of the loss of the DeepPlayer.
'''


import os
import pickle
import time

from game import ConnectFour
from players import DeepPlayer
from tools import visualize_loss


# Initialize game
connectfour = ConnectFour()


# Initialize Deep Player
alpha = 0.15
epsilon = 0.2
gamma = 0.9
num_central_layers = 1
central_layer_dim = 40
deepplayer = DeepPlayer(alpha, epsilon, gamma, num_central_layers, central_layer_dim)


# Training parameters
render = True

number_of_games = 70

decrease_parameters = {
	'decrease_alpha' : True,
	'alpha_decrease_factor' : 0.8,
	'decrease_epsilon' : True,
	'epsilon_decrease_factor' : 0.9,
	'decrease_after' : number_of_games/10
}


# Training the approximate player
time1 = time.time()
loss = deepplayer.train(connectfour, number_of_games, decrease_parameters, render)
time2 = time.time()


# Training info
training_time = time2 - time1


#Print training info
print('Duration of Deep Player training:', round(training_time, 0), 'seconds')
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


# Save player as pickle files
with open(dir_path + '/deepplayer.pickle', 'wb') as file:
	pickle.dump(deepplayer, file)


# Save loss in a pickle file
with open(dir_path + '/loss_of_deepplayer.pickle', 'wb') as file:
	pickle.dump(loss, file)


# Visualize loss
visualize_loss(dir_path + '/loss_of_deepplayer.pickle', number_of_games, num_central_layers, central_layer_dim)


# Training info
training_info = {
	'alpha' : alpha,
	'epsilon' : epsilon,
	'gamma' : gamma,
	'num_central_layers' : num_central_layers,
	'central_layer_dim' : central_layer_dim,
	'number_of_games' : number_of_games,
	'decrease_parameters' : decrease_parameters,
	'training_time' : training_time,
}


# Save training info as pickle file
with open(dir_path + '/training_info.pickle', 'wb') as file:
	pickle.dump(training_info, file)