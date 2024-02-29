'''Players

This file contains the players. For every player there is a class.
The mosti important function of every class is 'choice(self, game)':

	The parameter 'game' is an object of type 'TicTacToe' found in 'game.py'.
	It returns the players choice of action.
	An action is a value in {0,...,8} describing the board position.

The classes 'QPlayer', 'TDPlayer', 'DeepPlayer' contain two additional important functions:

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


class MinimaxPlayer:
	def __init__(self):
		self.name = 'Minimax'

	def Phi(self,game):
		if game.terminated:
			if game.winner == 'x':
				return 1
			elif game.winner == 'o':
				return -1
			else:
				return 0
		else:
			if game.player == 1:
				r = -infinity
				for action in game.legal_actions():
					game.execute(action)
					r = max(r, self.Phi(game))
					game.undo(action)
				return r
			else:
				r = infinity
				for action in game.legal_actions():
					game.execute(action)
					r = min(r, self.Phi(game))
					game.undo(action)
				return r
	
	def choice(self, game):
		if game.player == 1:
			r = -infinity
			best_action = None
			legal_actions = game.legal_actions()
			random.shuffle(legal_actions)
			for action in legal_actions:
				game.execute(action)
				r = self.Phi(game)
				if r > r:
					r = r
					best_action = action
				game.undo(action)
			return best_action
		else:
			best_action = None
			r = infinity
			legal_actions = game.legal_actions()
			random.shuffle(legal_actions)
			for action in legal_actions:
				game.execute(action)
				r = self.Phi(game)
				if r < r:
					r = r
					best_action = action
				game.undo(action)
			return best_action


class PrunPlayer:
	def __init__(self):
		self.name = 'Prun'
	
	def Phi(self, game , alpha, beta):
		if game.terminated:
			if game.winner == 'x':
				return 1
			elif game.winner == 'o':
				return -1
			else:
				return 0
		else:
			if game.player == 1:
				r = -infinity
				for action in game.legal_actions():
					game.execute(action)
					e = self.Phi(game, alpha, beta)
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
					e = self.Phi(game, alpha, beta)
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
				e = self.Phi(game, alpha, beta)
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
				e = self.Phi(game, alpha, beta)
				if e < r:
					r = e
					best_action = action
				beta = min(beta,e)
				if alpha >= beta:
					game.undo(action)
					break
				game.undo(action)
			return best_action


class QPlayer:
	def __init__(self, alpha, epsilon, gamma):
		self.name = 'Q'
		self.alpha = alpha
		self.epsilon = epsilon
		self.gamma = gamma
		self.Q1 = {}
		self.Q2 = {}

	def get_Q1(self, state, action):
		if self.Q1.get((state,action)) == None:
			self.Q1[(state,action)] = random.uniform(-1,1)*0.2
		return self.Q1[(state,action)]

	def get_Q2(self, state, action):
		if self.Q2.get((state,action)) == None:
			self.Q2[(state,action)] = random.uniform(-1,1)*0.2
		return self.Q2[(state,action)]

	def max_Q1(self, game):
		max_Q1 = -infinity
		max_action = None
		for action in game.legal_actions():
			state = tuple(game.board)
			Q1 = self.get_Q1(state, action)
			if Q1 > max_Q1:
				max_Q1 = Q1
				max_action = action
		return max_Q1, max_action

	def min_Q2(self, game):
		min_Q2 = infinity
		min_action = None
		for action in game.legal_actions():
			state = tuple(game.board)
			Q2 = self.get_Q2(state, action)
			if Q2 < min_Q2:
				min_Q2 = Q2
				min_action = action
		return min_Q2, min_action

	def choice(self, game):
		if random.uniform(0,1) < self.epsilon:
			return random.choice(game.legal_actions())
		else:
			if game.player == 1:
				return self.max_Q1(game)[1]
			else:
				return self.min_Q2(game)[1]

	def train_single_game(self, game):
		game.reset()

		state, action = tuple(game.board), self.choice(game)
		game.execute(action)
		
		opponent_state, opponent_action = tuple(game.board), self.choice(game)
		game.execute(opponent_action)
		
		next_state = tuple(game.board)

		while not game.terminated:
			if game.player == 1:
				self.Q1[(state,action)] = self.get_Q1(state,action) + self.alpha*(0 + self.gamma*self.max_Q1(game)[0] - self.get_Q1(state,action))
			else:
				self.Q2[(state,action)] = self.get_Q2(state,action) + self.alpha*(0 + self.gamma*self.min_Q2(game)[0] - self.get_Q2(state,action))
			
			next_action = self.choice(game)
			game.execute(next_action)
			
			next_opponent_state = tuple(game.board)
			
			state, action = opponent_state, opponent_action
			opponent_state, opponent_action = next_state, next_action
			next_state = next_opponent_state

		if game.winner == 'x':
			reward = 1
		elif game.winner == 'o':
			reward = -1
		else:
			reward = 0

		if game.player == 1:
			self.Q1[(state,action)] = self.get_Q1(state,action) + self.alpha*(reward - self.get_Q1(state,action))
			self.Q2[(opponent_state,opponent_action)] = self.get_Q2(opponent_state,opponent_action) + self.alpha*(reward - self.get_Q2(opponent_state,opponent_action))
		else:
			self.Q2[(state,action)] = self.get_Q2(state,action) + self.alpha*(reward - self.get_Q2(state,action))
			self.Q1[(opponent_state,opponent_action)] = self.get_Q1(opponent_state,opponent_action) + self.alpha*(reward - self.get_Q1(opponent_state,opponent_action))

	def train(self, game, number_of_games, render):
		if render:
			print('Start training of', self.name, ' with the following parameters:')
			print('Number of games:', number_of_games, ' | alpha:', self.alpha, ' | epsilon:', self.epsilon, ' | gamma:', self.gamma)
			print()
			print(0, 'games completed')
		for i in range(1, number_of_games + 1):
			self.train_single_game(game)
			if i % (number_of_games/10) == 0:
				if render:
					print(i, 'games completed')
		if render:
			print()


class TDPlayer:
	def __init__(self, alpha, epsilon, gamma, lanbda):
		self.name = 'TD'
		self.alpha = alpha
		self.epsilon = epsilon
		self.gamma = gamma
		self.lanbda = lanbda
		self.V = {}
		self.z = {}

	def get_V(self, state):
		if self.V.get(state) == None:
			self.V[state] = random.uniform(-1,1)*0.2
		return self.V[state]

	def choice(self, game):
		if random.uniform(0,1) < self.epsilon:
			return random.choice(game.legal_actions())
		else:
			if game.player == 1:
				max_V = -infinity
				best_action = None
				for action in game.legal_actions():
					game.execute(action)
					state = tuple(game.board)
					V = self.get_V(state)
					if V > max_V:
						max_V = V
						best_action = action
					game.undo(action)
				return best_action
			else:
				min_V = infinity
				best_action = None
				for action in game.legal_actions():
					game.execute(action)
					state = tuple(game.board)
					V = self.get_V(state)
					if V < min_V:
						min_V = V
						best_action = action
					game.undo(action)
				return best_action

	def train_single_game(self, game):
		game.reset()

		state, action = tuple(game.board), self.choice(game)
		game.execute(action)

		next_state = tuple(game.board)

		z = {}

		while not game.terminated:
			if z.get(state) == None:
				z[state] = 1
			else:
				z[state] +=1

			for x, eligibility in z.items():
				self.V[x] = self.get_V(x) + self.alpha*(0 + self.gamma*self.get_V(next_state) - self.get_V(state))*eligibility
				z[x] *= self.gamma*self.lanbda

			action = self.choice(game)
			game.execute(action)
			next_next_state = tuple(game.board)

			state = next_state
			next_state = next_next_state

		if z.get(state) == None:
			z[state] = 1
		else:
			z[state] +=1

		if game.winner == 'x':
			reward = 1
		elif game.winner == 'o':
			reward = -1
		else:
			reward = 0

		for x, eligibility in z.items():
			self.V[x] = self.get_V(x) + self.alpha*(reward - self.get_V(state))*eligibility
			z[x] *= self.gamma*self.lanbda

		self.V[next_state] = reward

	def train(self, game, number_of_games, render):
		if render:
			print('Start training of', self.name, ' with the following parameters:')
			print('Number of games:', number_of_games, ' | alpha:', self.alpha, ' | epsilon:', self.epsilon, ' | gamma:', self.gamma)
			print()
			print(0, 'games completed')
		for i in range(1, number_of_games + 1):
			self.train_single_game(game)
			if i % (number_of_games/10) == 0:
				if render:
					print(i, 'games completed')
		if render:
			print()


class DeepPlayer:
	def __init__(self, alpha, epsilon, gamma, num_central_layers, central_layer_dim):
		self.name = 'Deep'
		self.alpha = alpha
		self.epsilon = epsilon
		self.gamma = gamma

		# In the file 'neuralnetwork.py' there is a class 'FNN' that describes the network
		self.net = FNN(9*3 + 2, num_central_layers, central_layer_dim, 1)
		self.loss_fn = nn.MSELoss()
		self.optimizer = torch.optim.SGD(self.net.parameters(), lr = self.alpha)

	@torch.no_grad()
	def preprocess(self, board, player):
		array = []
		for i in range(9):
			if board[i] == 1:
				array += [1,0,0]
			elif board[i] == -1:
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