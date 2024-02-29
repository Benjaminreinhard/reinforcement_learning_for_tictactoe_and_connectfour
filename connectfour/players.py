'''Players

This file contains the players. For every player there is a class.
The mosti important function of every class is 'choice(self, game)':

	The parameter 'game' is an object of type 'ConnectFour' found in 'game.py'.
	It returns the players choice of action.
	An action is a value in {0,...,num_of_columns} describing the column of the grid.

The class 'DeepPlayer' contains two additional important functions:

	'train_single_game(self, game)' and 'train(...)'. They are the implementations of the pseudocodes
	that can be found in './reinforcement_learning_for_tictactoe_and_connectfour/pseudocodes.pdf'
'''


import random
import torch
import torch.nn as nn

from math import inf as infinity
from statistics import mean
from copy import copy, deepcopy

from neuralnetwork import FNN


class HumanPlayer:
	'''Used for you to play'''

	def __init__(self):
		self.name = 'Human'

	@staticmethod
	def choice(game):
		legal_actions_as_str = [str(i) for i in game.legal_actions()]
		action_as_str = input('Insert position: ')
		while True:
			if action_as_str not in legal_actions_as_str:
				action_as_str = input('Not a legal action, please choose a number in ' + str(game.legal_actions()) + ': ')
			else:
				break
		return int(action_as_str)


class RandomPlayer:
	def __init__(self):
		self.name = 'Random'

	@staticmethod
	def choice(game):
		return random.choice(game.legal_actions())


class PrunPlayer:
	def __init__(self, depth):
		self.name = 'Prun' + str(depth)
		self.depth = depth
	
	def Phi(self, game, alpha, beta, depth):
		if game.terminated:
			if game.winner == 'x':
				return 1
			elif game.winner == 'o':
				return -1
			else:
				return 0
		elif depth == 0:
			if game.player == 1:
				return 0.5
			else:
				return -0.5
		else:
			if game.player == 1:
				r = -infinity
				for action in game.legal_actions():
					game.execute(action)
					e = self.Phi(game, alpha, beta, depth-1)
					r = max(r,e)
					alpha = max(alpha,e)
					if alpha >= beta:
						game.undo(action)
						break
					game.undo(action)
				return r
			else:
				r = infinity
				for action in game.legal_actions():
					game.execute(action)
					e = self.Phi(game, alpha, beta, depth-1)
					r = min(r,e)
					beta = min(beta,e)
					if alpha >= beta:
						game.undo(action)
						break
					game.undo(action)
				return r
	
	def choice(self, game):
		alpha, beta = -infinity, infinity
		if game.player == 1:
			r = -infinity
			best_action = None
			legal_actions = game.legal_actions()
			random.shuffle(legal_actions)
			for action in legal_actions:
				game.execute(action)
				e = self.Phi(game, alpha, beta, self.depth-1)
				if e > r:
					r = e
					best_action = action
				alpha = max(alpha,e)
				if alpha >= beta:
					game.undo(action)
					break
				game.undo(action)
			return best_action
		else:
			r = infinity
			best_action = None
			legal_actions = game.legal_actions()
			random.shuffle(legal_actions)
			for action in legal_actions:
				game.execute(action)
				e = self.Phi(game, alpha, beta, self.depth-1)
				if e < r:
					r = e
					best_action = action
				beta = min(beta,e)
				if alpha >= beta:
					game.undo(action)
					break
				game.undo(action)
			return best_action


class ChainPlayer:
	def __init__(self, type):
		self.name = 'O.Chain' if type == 'offensive' else 'D.Chain'
		self.sign = 1 if type == 'offensive' else -1

	def choice(self, game):
		action = None
		max_count = 0
		legal_actions = game.legal_actions()
		random.shuffle(legal_actions)
		for column in legal_actions:
			for i in range(game.num_of_rows-1,-1,-1):
				if game.board[i][column] == 0:
					row = i
					break
			lines = [1,2,3,4]
			random.shuffle(lines)
			for line in lines:
				c = game.counter(game.player*self.sign, row, column, line)
				if c > max_count:
					max_count = c
					action = column
		return action


class DeepPlayer:
	def __init__(self, alpha, epsilon, gamma, num_central_layers, central_layer_dim):
		self.name = 'Deep'
		self.alpha = alpha
		self.epsilon = epsilon
		self.gamma = gamma

		# In the file 'neuralnetwork.py' there is a class 'FNN' that describes the network
		self.net = FNN(7*6*3 + 2, num_central_layers, central_layer_dim, 1)
		self.loss_fn = nn.MSELoss()
		self.optimizer = torch.optim.SGD(self.net.parameters(), lr = self.alpha)

	@torch.no_grad()
	def preprocess(self, board, player):
		m = len(board)
		n = len(board[0])
		array = []
		for i in range(m):
			for j in range(n):
				if board[i][j] == 1:
					array += [1,0,0]
				elif board[i][j] == -1:
					array += [0,1,0]
				else:
					array += [0,0,1]

		if player == 1:
			array += [1,0]
		else:
			array += [0,1]
		return torch.tensor(array, dtype=torch.float32)

	@torch.no_grad()
	def choice(self, game):
		if random.uniform(0,1) < self.epsilon:
			return random.choice(game.legal_actions())
		else:
			if game.player == 1:
				max_V = -infinity
				max_action = None
				for action in game.legal_actions():
					game.execute(action)
					x = self.preprocess(game.board, game.player)
					V = self.net(x).item()
					if V > max_V:
						max_V = V
						max_action = copy(action)
					game.undo(action)
				return max_action
			else:
				min_V = infinity
				min_action = None
				for action in game.legal_actions():
					game.execute(action)
					x = self.preprocess(game.board,game.player)
					V = self.net(x).item()
					if V < min_V:
						min_V = V
						min_action = copy(action)
					game.undo(action)
				return min_action

	def update(self, board, player, T, t, reward):
		target = torch.tensor([0.5*(self.gamma**(T-t))*reward + 0.5], dtype=torch.float32)
		
		x = self.preprocess(board, player)

		self.optimizer.zero_grad()
		loss = self.loss_fn(target, self.net(x))
		loss.backward()
		self.optimizer.step()
		
		return loss

	def train_single_game(self, game):
		losslist = []

		game.reset()

		episode = []
		episode.append([copy(game.board), game.player, game.time])
		
		while not game.terminated:
			action = self.choice(game)
			game.execute(action)
			episode.append([copy(game.board), game.player, game.time])

		T = game.time

		if game.winner == 'x':
			reward = 1
		elif game.winner == 'o':
			reward = -1
		else:
			reward = 0

		for board, player, t in episode:
			loss = self.update(board, player, T, t, reward)
			losslist.append(loss.item())
		
		return losslist

	def train(self, game, number_of_games, decrease_parameters, render):
		decrease_alpha = decrease_parameters['decrease_alpha']
		alpha_decrease_factor = decrease_parameters['alpha_decrease_factor']
		decrease_epsilon = decrease_parameters['decrease_epsilon']
		epsilon_decrease_factor = decrease_parameters['epsilon_decrease_factor']
		decrease_after = decrease_parameters['decrease_after']

		losslist = []
		alpha_render = 0

		if render:
			print('Start training of', self.name, ' with the following parameters:')
			print('Number of games:', number_of_games)
			print('Alpha:', self.alpha, ' | decrease alpha:', decrease_alpha, ' | alpha decrease factor', alpha_decrease_factor)
			print('Epsilon:', self.epsilon, ' | decrease epsilon:', decrease_epsilon, ' | epsilon decrease factor', epsilon_decrease_factor)
			print()
			print(0, 'games completed')
		for i in range(1, number_of_games + 1):
			loss = self.train_single_game(game)
			losslist.append(mean(loss))
			if i % decrease_after == 0:
				if decrease_alpha:
					for g in self.optimizer.param_groups:
						g['lr'] *= alpha_decrease_factor
						alpha_render = g['lr']
				if decrease_epsilon:
					self.epsilon *= epsilon_decrease_factor
				if render:
					print(i, 'games completed.', 'New alpha:', round(alpha_render,3), 'New epsilon:', round(self.epsilon,3))
		if render:
			print()

		return losslist