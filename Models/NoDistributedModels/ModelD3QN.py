import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import time
import sys

from Models.Networks.DuelingNetwork import Network

from utils.tool import read_config, read_seed

np.random.seed(read_seed())

config = read_config("./Configs/configD3QN.yml")

N_ACTIONS = config["n_actions"]
GAMMA = config["gamma"]
EPSILON_DECAY = config["epsilon_decay"]
EPSILON_MIN = config["epsilon_min"]
BATCH_SIZE = config["batch_size"]
MIN_REPLAY_SIZE = config["min_replay_size"]
TARGET_UPDATE = config["target_update"]
SAMPLING_TIME = config["sampling_time"]

NAME = config["model_name"]

# Deep Q Network agent
class Agent:
	def __init__(self):

		self.replay = deque(maxlen = config["size_buffer"])
		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

		self.Q = Network(config).to(self.device)

		self.target_Q = Network(config).to(self.device)
		self.target_Q.load_state_dict(self.Q.state_dict()) #synchronization of the parameters

		#Initialisation of agent variables	
		self.epsilon = config["epsilon"]	
		self.target_update_counter = 0
		self.loss = None
		self.name = config["model_name"]

	def save_model(self):
		torch.save(self.Q.state_dict(), "./ModelSaved/" + NAME + '.pth')
		print("Model saved")

	def add_transition(self, obs, action, reward, next_obs, done):
		self.replay.append((obs, action, self.reward_clipping(reward), next_obs, done)) 

	def reward_clipping(self, reward):
		if reward > 1:
			reward = 1
		elif reward <-1:
			reward = -1
		return reward
			
	def choose_action(self, obs):
		#Choose an action according epsilon-greedy
		if np.random.uniform() < self.epsilon:
			action = np.random.choice(N_ACTIONS)
		else:
			y = self.Q(torch.tensor(obs[0], device=self.device, dtype=torch.float).unsqueeze(0),
						torch.tensor(obs[1],  device=self.device, dtype=torch.float).unsqueeze(0))
			action = torch.argmax(y).item()
		return action

	def train_nn(self):
		
		if len(self.replay) < MIN_REPLAY_SIZE:
			return
		#Sample transitions from the minibatch 
		idx = np.random.choice(len(self.replay), BATCH_SIZE, replace=True) 
		mini_batch = np.array(self.replay)[idx]

		#Split data transitions into multiples tensors
		current_states_img = torch.tensor([transition[0][0] for transition in mini_batch], device=self.device, dtype=torch.float)
		current_states_nav = torch.tensor([transition[0][1] for transition in mini_batch], device=self.device, dtype=torch.float)
			
		actions = torch.tensor([transition[1] for transition in mini_batch], device=self.device, dtype=torch.long)
		rewards = torch.tensor([transition[2] for transition in mini_batch], device=self.device, dtype=torch.float)

		new_current_states_img = torch.tensor([transition[3][0] for transition in mini_batch], device=self.device, dtype=torch.float)
		new_current_states_nav = torch.tensor([transition[3][1] for transition in mini_batch], device=self.device, dtype=torch.float)
			
		dones = torch.tensor([not(transition[4]) for transition in mini_batch], device=self.device, dtype=torch.bool)


		#Estimate the next Q value with the target network

		actions_eval = torch.argmax(self.Q(new_current_states_img, new_current_states_nav), dim=1)

		next_state_values = self.target_Q(new_current_states_img, new_current_states_nav).gather(dim=1, index=actions_eval.unsqueeze(-1)).squeeze(-1)					
		values = rewards + GAMMA*next_state_values*dones

		target_values = self.Q(current_states_img, current_states_nav).gather(dim=1,index=actions.unsqueeze(-1)).squeeze(-1)



		#Perform a gradient descent step on the error
				#Compute the loss with MSE Loss
		loss_t = self.Q.loss_function(values, target_values)
		self.loss = loss_t

		self.Q.optimizer.zero_grad()
		loss_t.backward()
		for param in self.Q.parameters():
			param.grad.data.clamp(-1,1)
		self.Q.optimizer.step()

		self.update_target()
		self.update_epsilon()
	 
	 
		 
	def update_target(self):
		#update target counter
		self.target_update_counter +=1   
		#Every C update target network 
		if self.target_update_counter > TARGET_UPDATE:
			self.target_Q.load_state_dict(self.Q.state_dict())
			self.target_update_counter = 0

	def update_epsilon(self):
		#update epsilon
		self.epsilon *= EPSILON_DECAY
		self.epsilon = max(self.epsilon, EPSILON_MIN)