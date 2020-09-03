import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random

from Models.Networks.CustomLayer.InitLinearModel import InitLinear
class Network(nn.Module):
	def __init__(self, config):
		super(Network, self).__init__()

		#Conv layers
		"""
		input_shape[0] to get the image dim
		input_shape[1] to get the navigation inputs dim
		"""
		input_shape = config["input_shape"]
		n_actions = config["n_actions"]
		out_channel1 = config["out_channel1"]
		out_channel2 = config["out_channel2"]

		hidden_size1 = config["hidden_size1"]
		hidden_size2 = config["hidden_size2"]

		learning_rate = config["learning_rate"]
		
		self.conv = nn.Sequential(           
			nn.Conv2d(input_shape[0][0], out_channel1, kernel_size=1, stride=2),
			nn.ReLU(),
			nn.Conv2d(out_channel1, out_channel2, kernel_size=1, stride=1),
			nn.ReLU(),
			nn.Conv2d(out_channel2, out_channel2, kernel_size=1, stride=1),
			nn.ReLU(),
			nn.Flatten()
		)
				
		conv_out_size = self._get_conv_out(input_shape[0])
		#Fully connected layers
		self.net_navigation = nn.Sequential(
			InitLinear(input_shape[1], hidden_size1),
			nn.ReLU()
		)
		

		self.net_Q = nn.Sequential(
			InitLinear(hidden_size1+conv_out_size, hidden_size2),
			nn.ReLU(),
			InitLinear(hidden_size2, n_actions)
		)
			
		self.optimizer = optim.Adam(params=self.parameters(), lr=learning_rate)
				
		
		self.loss_function = nn.MSELoss()
				
	#get the output dim of the conv layers
	def _get_conv_out(self, shape):
		o = self.conv(torch.zeros(1, *shape))
		return int(np.prod(o.size()))

	def forward(self, x_image, x_nav):
		#Split the input in two, in one side the image, in the other side the navigation input
		x_image = x_image.unsqueeze(-1)
		conv_out = self.conv(x_image)
		navigation_out = self.net_navigation(x_nav)

		X = torch.cat((conv_out, navigation_out), 1-len(navigation_out.size()))

		return self.net_Q(X)