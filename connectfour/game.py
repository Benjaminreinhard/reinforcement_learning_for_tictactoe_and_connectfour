'''Connect Four architecture

This file contains the class that describes the game. Comments:

	A grid is saved as a list of shape (num_of_rows, num_of_columns). You can choose the shape yourself.
	Time is being tracked.
	Player 1 is indicated by '1' and player 2 as '-1'.
	Terminated tells you if the game ended.
	Winner is 'None' at first and if terminated and player 1(2) wins, it changes to 'x'('o').

The most important function is 'execute(self, action)':

	An action is a value in the set {0,...,num_of_columns} and it denotes a position on the board.
	When the function is executed, it changes the internal values accordingly.
'''


class ConnectFour:
	def __init__(self, num_of_rows=6, num_of_columns=7):
		self.name = 'ConnectFour' + str(num_of_rows) + str(num_of_columns)
		self.num_of_rows = num_of_rows
		self.num_of_columns = num_of_columns
		self.board = [[0 for j in range(self.num_of_columns)] for i in range(self.num_of_rows)]
		self.player = 1
		self.time = 0
		self.max_time = self.num_of_rows*self.num_of_columns
		self.terminated = False
		self.winner = None

	def reset(self):
		self.board = [[0 for j in range(self.num_of_columns)] for i in range(self.num_of_rows)]
		self.player = 1
		self.time = 0
		self.terminated = False
		self.winner = None

	def render(self):
		for i in range(self.num_of_rows):
			string = ''
			for j in self.board[i]:
				if j==0:
					string += '.'
				elif j==1:
					string += 'x'
				else:
					string += 'o'
			if i == 0:
				print(string, '   ', 'turn:', 'x' if self.player == 1 else 'o')
			elif i == 1:
				print(string, '   ', 'time:', self.time)
			elif i == 2:
				print(string, '   ', 'winner:', self.winner)
			elif i == 3:
				print(string, '   ', 'terminated:', self.terminated)
			else:
				print(string)
		print()

	def legal_actions(self):
		return [j for j in range(self.num_of_columns) if self.board[0][j]==0]

	def counter(self, player, row, column, line):
		'''Counts the number of discs of a player at a coordinate (row, column)

			Parameter 'line' lies in {1,2,3,4}.
			Line 1 goes from top left to bottom right
			Line 2 goes from left to right
			Line 3 goes from bottom left to top right
			Line 4 goes from bottom to center
		'''

		count = 1
		if line == 1:
			top_left = min(row + 1, column + 1, 4)
			for k in range(1, top_left):
				if self.board[row-k][column-k] != player:
					break
				else:
					count += 1
			bottom_right = min(self.num_of_rows - row, self.num_of_columns - column, 4)
			for k in range(1, bottom_right):
				if self.board[row+k][column+k] != player:
					break
				else:
					count += 1
		elif line == 2:
			left = min(column + 1, 4)
			for k in range(1, left):
				if self.board[row][column-k] != player:
					break
				else:
					count += 1
			right = min(self.num_of_columns - column, 4)
			for k in range(1, right):
				if self.board[row][column+k] != player:
					break
				else:
					count += 1
		elif line == 3:
			bottom_left = min(column +1, self.num_of_rows - row, 4)
			for k in range(1, bottom_left):
				if self.board[row+k][column-k] != player:
					break
				else:
					count += 1
			top_right = min(self.num_of_columns - column, row + 1, 4)
			for k in range(1, top_right):
				if self.board[row-k][column+k] != player:
					break
				else:
					count += 1
		elif line == 4:
			bottom = min(self.num_of_rows - row, 4)
			for k in range(1, bottom):
				if self.board[row+k][column] != player:
					break
				else:
					count += 1
		return count

	def is_winner(self, action):
		for i in range(self.num_of_rows):
			if self.board[i][action] != 0:
				row = i
				break

		for line in [1,2,3,4]:
			count = self.counter(self.player,row,action,line)
			if count >= 4:
				return True

	def execute(self, action):
		for i in range(self.num_of_rows-1,-1,-1):
			if self.board[i][action] == 0:
				self.board[i][action] = self.player
				break

		if self.is_winner(action):
			self.winner = 'x' if self.player == 1 else 'o'
			self.terminated = True

		self.time += 1
		if self.time == self.max_time:
			self.terminated = True

		self.player = self.player*(-1)

	def undo(self, action):
		self.player = self.player*(-1)
		self.terminated = False
		self.time -= 1
		self.winner = None

		for i in range(self.num_of_rows):
			if self.board[i][action] != 0:
				self.board[i][action] = 0
				break