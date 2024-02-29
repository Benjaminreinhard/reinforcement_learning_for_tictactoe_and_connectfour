'''Tools for training, evaluation and demonstration

This file contains functions that are used in the files 'training.py', 'evaluation.py' and 'demonstration.py'.
'''


import random
import copy
import pickle
import os
import matplotlib.pyplot as plt

from statistics import mean


def play(game, player_x, player_o, first_action_random, render):
	game.reset()
	if render:
		game.render()
	if first_action_random:
		action = random.choice(game.legal_actions())
		game.execute(action)
		if render:
			game.render()
	while not game.terminated:
		if game.player == 1:
			action = player_x.choice(game)
		else:
			action = player_o.choice(game)
		game.execute(action)
		if render:
			game.render()


def evaluation(game, players, games_per_pair, first_action_random):
	scores = {}
	for player_x in players:
		for player_o in players:
			score = {'x':0, 'o': 0, None: 0}
			for i in range(games_per_pair):
				play(game, player_x, player_o, first_action_random, False)
				score[game.winner] += 1
			scores[(player_x.name, player_o.name)] = score
	return scores


def visualize_scores(file_path):
	with open(file_path, 'rb') as file:
		scores = pickle.load(file)

	player_xs = []
	player_os = []
	for key in scores.keys():
		if key[0] not in player_xs:
			player_xs.append(key[0])
		if key[1] not in player_os:
			player_os.append(key[1])

	m = len(player_xs)
	n = len(player_os)

	matrix = [[0 for j in range(n*3)] for i in range(m)]
	for i in range(m):
		for j in range(n):
			matrix[i][j*3+0] = j%2
			matrix[i][j*3+1] = j%2
			matrix[i][j*3+2] = j%2

	fig, ax = plt.subplots()
	im = ax.imshow(matrix, cmap='Oranges', vmin=0, vmax=7)

	ax.set_title('Scores')
	ax.set_xlabel('Player o')
	ax.set_xticks([j for j in range(n*3)])
	xticklabels = []
	for j in range(n):
		xticklabels += ['x','o\n'+player_os[j],'.']
	ax.set_xticklabels(xticklabels)
	ax.set_ylabel('Player x')
	ax.set_yticks([i for i in range(m)])
	ax.set_yticklabels([player_xs[i] for i in range(m)])

	for i in range(m):
		for j in range(n):
			score = scores[(player_xs[i], player_os[j])]
			text = ax.text(j*3+0, i, score['x'], ha="center", va="center", color="black")
			text = ax.text(j*3+1, i, score['o'], ha="center", va="center", color="black")
			text = ax.text(j*3+2, i, score[None], ha="center", va="center", color="black")
	
	i = file_path.find('.pickle')
	file_path = file_path[:i] + '.png'
	plt.savefig(file_path, bbox_inches='tight')
	plt.close()


def visualize_loss(file_path, number_of_games, num_central_layers, central_layer_dim, average_over=100):
	with open(file_path, 'rb') as file:
		loss = pickle.load(file)

	average_loss = []
	n = len(loss)
	k = average_over
	for i in range(n-k):
		m = mean(loss[i:i+k])
		average_loss.append(m)
	title = 'Games: '+str(number_of_games)
	title += '\nNum. central layers: '+str(num_central_layers)+', central layer dim: '+str(central_layer_dim)
	plt.title(title)
	plt.plot(loss, label='Loss')
	plt.plot(average_loss, label='Av. over '+str(k)+' losses')
	plt.legend(loc="upper right")
	
	i = file_path.find('.pickle')
	file_path = file_path[:i] + '.png'
	plt.savefig(file_path)
	plt.close()