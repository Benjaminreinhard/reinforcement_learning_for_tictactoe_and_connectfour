'''Feedforward neural network

This file contains the Feedforward neural network (FNN) that is used in the
class 'DeepPlayer'. 
'''


import torch
import torch.nn as nn


class FNN(nn.Module):
	def __init__(self, input_dim, num_central_layers, central_layer_dim, output_dim):
		super().__init__()
		
		self.first_layer = nn.Linear(input_dim, central_layer_dim)

		self.central_layers = []
		for i in range(num_central_layers):
			self.central_layers.append(nn.Linear(central_layer_dim, central_layer_dim))

		self.final_layer = nn.Linear(central_layer_dim, output_dim)

		self.relu = nn.ReLU()
		self.sigmoid = nn.Sigmoid()
	
	def forward(self, x):

		out = self.first_layer(x)
		out = self.relu(out)

		for layer in self.central_layers:
			out = layer(out)
			out = self.relu(out)

		out = self.final_layer(out)
		out = self.sigmoid(out)

		return out