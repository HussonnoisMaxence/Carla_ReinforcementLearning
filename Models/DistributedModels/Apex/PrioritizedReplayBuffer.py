
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import sys
import random
from utils.tool import read_seed
np.random.seed(read_seed())


class Memory:
	def __init__(self, config):
		limite = config['size_buffer']
		self.transitions = deque(maxlen = limite)
		self.priority = deque(maxlen = limite)


		self.probability = deque(maxlen=limite)
		self.td_error = deque(maxlen=limite)

		self.alpha = config['alpha']
		self.beta = config['beta']

		self.priority_beta_increment = ((1 - self.beta)/ config['learning_step'])
		self.epsilon = config['epsilon_buffer']

		self.index_sample = None


	def add_all(self, experiences, errors):
		for exp, err in zip(experiences, errors):
			self.priority.append((abs(err )+ self.epsilon)**(self.alpha))
			self.transitions.append(exp)

	def update_error(self, error):
		t = (abs(error )+ self.epsilon)
		np.array(self.priority)[self.index_sample] = t**(self.alpha)


	def get_batch(self, size):
		N = int(self.get_size())
		self.probability = np.array(self.priority)[:N]/sum(np.array(self.priority)[:N])


		#np.random.seed(self.choose_seed())
		index_sample = np.random.choice(N, size, p=self.probability, replace=True)

		t = np.array([x for x in self.transitions])
	
		mini_batch = t[index_sample]
		weight = N*np.array(self.probability)[index_sample]**(-self.beta)
		isw =  weight/max(weight)
		self.index_sample = index_sample
		self.beta += self.priority_beta_increment

		return mini_batch, isw

	def get_size(self):
		return len(self.transitions)