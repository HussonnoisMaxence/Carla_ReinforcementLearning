
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
import random
#create seed
from utils.tool import read_seed
torch.manual_seed(read_seed())

class InitLinear(nn.Module):
	def __init__(self, in_features, out_features):
		super(InitLinear, self).__init__()
		self.in_features = in_features
		self.out_features = out_features
		self.weight = nn.Parameter(torch.empty(out_features, in_features))
		self.bias = nn.Parameter(torch.empty(out_features))

		self.reset_parameters()



	def reset_parameters(self):
		mu_range = 1.0 / math.sqrt(self.in_features)
		self.weight.data.uniform_(-mu_range, mu_range)
		self.bias.data.uniform_(-mu_range, mu_range)


	def forward(self, inp):
		return F.linear(inp, self.weight, self.bias)

