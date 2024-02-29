'''Tic Tac Toe architecture'''

class TicTacToe:
	def __init__(self):
		'''
		Initializes a new game with the following values:
		A board is saved as a list with 9 entries.
		Time is being tracked.
		Player 1 is indicated by '1' and player 2 as '-1'
		Terminated tells you if the game ended
		Winner is 'None' at first and if the game terminated and player 1/2 wins, it changes to 'x'/'o'
		'''

		self.name = 'TicTacToe'
		self.board = [0]*9
		self.time = 0
		self.player = 1
		self.terminated = False
		self.winner = None

	def reset(self):
		'''Resets the game to its initial values'''
		
		self.board = [0]*9
		self.time = 0
		self.player = 1
		self.terminated = False
		self.winner = None

	def render(self):
		string = ''
		for i in range(9):
			if self.board[i]==0:
				string += '.'
			elif self.board[i]==1:
				string += 'x'
			else:
				string += 'o'

		print(string[0],string[1],string[2], '   ', 'turn:', 'x' if self.player == 1 else 'o')
		print(string[3],string[4],string[5], '   ', 'time:', self.time)
		print(string[6],string[7],string[8], '   ', 'winner:', self.winner, '|', 'terminated:', self.terminated)
		print()
	
	def legal_actions(self):
		return [i for i in range(9) if self.board[i] == 0]

	def is_winner(self):
		a1 = self.board[0] == self.board[1] == self.board[2] == self.player
		a2 = self.board[3] == self.board[4] == self.board[5] == self.player
		a3 = self.board[6] == self.board[7] == self.board[8] == self.player

		a4 = self.board[0] == self.board[3] == self.board[6] == self.player
		a5 = self.board[1] == self.board[4] == self.board[7] == self.player
		a6 = self.board[2] == self.board[5] == self.board[8] == self.player

		a7 = self.board[0] == self.board[4] == self.board[8] == self.player
		a8 = self.board[6] == self.board[4] == self.board[2] == self.player

		return a1 or a2 or a3 or a4 or a5 or a6 or a7 or a8

	def execute(self, action):
		self.board[action] = self.player

		if self.is_winner():
			self.winner = 'x' if self.player == 1 else 'o'
			self.terminated = True

		self.time += 1
		if self.time == 9:
			self.terminated = True

		self.player = self.player*(-1)

	def undo(self, action):
		self.player = self.player*(-1)
		self.time -=1

		self.terminated = False
		self.winner = None

		self.board[action] = 0