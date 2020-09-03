import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import time
import psutil
from Models.Networks.NoisyDuelingNetwork import Network
from torch.utils.tensorboard import SummaryWriter
from utils.tool import read_config
import sys
from utils.tool import read_seed
torch.manual_seed(read_seed())
config = read_config("./Configs/configApeX.yml")

GAMMA = config["gamma"]
BATCH_SIZE = config["batch_size"]
TARGET_UPDATE = config["target_update"]
LEARNING_STEP = config["learning_step"]
MIN_REPLAY_SIZE = config["min_replay_size"]

class Learner:
	def __init__(self, config):
		#Define networks

		self.Q = Network(config)
		self.target_Q = Network(config)
		self.target_Q.load_state_dict(self.Q.state_dict())
		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

		self.gamma = config["gamma"]
		self.batch_size = config["batch_size"]
		self.min_replay_size = config["min_replay_size"]

		self.target_update_counter = 0
		self.target_update = config["target_update"]

		self.learning_step = config["learning_step"]
	
	def compute_td(self, mini_batch):
		current_states_img = torch.tensor([transition[0][0] for transition in mini_batch], device=self.device, dtype=torch.float)
		current_states_nav = torch.tensor([transition[0][1] for transition in mini_batch], device=self.device, dtype=torch.float)

		actions = torch.tensor([transition[1] for transition in mini_batch], device=self.device, dtype=torch.long)
		rewards = torch.tensor([transition[2] for transition in mini_batch], device=self.device, dtype=torch.float)

		new_current_states_img = torch.tensor([transition[3][0] for transition in mini_batch], device=self.device, dtype=torch.float)
		new_current_states_nav = torch.tensor([transition[3][1] for transition in mini_batch], device=self.device, dtype=torch.float)
	  
		dones = torch.tensor([not(transition[4]) for transition in mini_batch], device=self.device, dtype=torch.bool)

		actions_eval = torch.argmax(self.Q(new_current_states_img, new_current_states_nav), dim=1)
		next_state_values = self.target_Q(new_current_states_img, new_current_states_nav).gather(dim=1, index=actions_eval.unsqueeze(-1)).squeeze(-1)
		values = rewards + self.gamma*next_state_values*dones

		target_values = self.Q(current_states_img, current_states_nav).gather(dim=1, index=actions.unsqueeze(-1)).squeeze(-1)

		td_error = target_values - values
		return td_error.detach().cpu().numpy()


	def train_nn(self, replay, server):
		updated_step = 0
		start_mem = psutil.virtual_memory().used
		while replay.get_size() < MIN_REPLAY_SIZE:
			continue
		print("Start learning")
		writer =  SummaryWriter(comment="LossLearner")
		while updated_step < LEARNING_STEP:
				
			
			#Sample random minibatch of transitions from replay
			r = replay.get_size()
			mini_batch, weight = replay.get_batch(BATCH_SIZE) # appel remote object method
			
			weight = torch.tensor(weight, device=self.device, dtype=torch.float)
			#Split data transitions into multiples tensors
			current_states_img = torch.tensor([transition[0][0] for transition in mini_batch], device=self.device, dtype=torch.float)
			current_states_nav = torch.tensor([transition[0][1] for transition in mini_batch], device=self.device, dtype=torch.float)
				
			actions = torch.tensor([transition[1] for transition in mini_batch], device=self.device, dtype=torch.long)
			rewards = torch.tensor([transition[2] for transition in mini_batch], device=self.device, dtype=torch.float)

			new_current_states_img = torch.tensor([transition[3][0] for transition in mini_batch], device=self.device, dtype=torch.float)
			new_current_states_nav = torch.tensor([transition[3][1] for transition in mini_batch], device=self.device, dtype=torch.float)
				
			dones = torch.tensor([not(transition[4]) for transition in mini_batch], device=self.device, dtype=torch.bool)
			


			actions_eval = torch.argmax(self.Q(new_current_states_img, new_current_states_nav), dim=1)

			next_state_values = self.target_Q(new_current_states_img, new_current_states_nav).gather(dim=1, index=actions_eval.unsqueeze(-1)).squeeze(-1)

		
			
			values = rewards + GAMMA*next_state_values*dones

			target_values = self.Q(current_states_img, current_states_nav).gather(dim=1,index=actions.unsqueeze(-1)).squeeze(-1)

			td_error = target_values - values

			replay.update_error(td_error.detach().cpu().numpy()) # appel remote object method

			#fit/backpropagation
			self.Q.optimizer.zero_grad()
			loss_t = self.Q.loss_function(values*weight, target_values*weight) 
			loss_t.backward()
			#for param in self.Q.parameters():
				#param.grad.data.clamp(-1,1)	
			self.Q.optimizer.step()
			#self.replay.update_memory()
			self.update_target()
			server.update_params(self.return_params()) # appel remote object method
			updated_step += 1

			writer.add_scalar("Loss", loss_t, updated_step)
			writer.add_scalar("replay_size:", r, updated_step)
			writer.add_scalar("mem used:", (psutil.virtual_memory().used - start_mem)/1_000_000, updated_step)

		
		print("finish")
		torch.save(self.Q.state_dict(),"./ModelSaved/Learner.pth")
		writer.close()

	def update_target(self):
		self.target_update_counter +=1
		if self.target_update_counter > TARGET_UPDATE:
			self.target_Q.load_state_dict(self.Q.state_dict())
			self.target_update_counter = 0

	def return_params(self):
		params = []
		for q_param in (self.Q.parameters()):
			params.append(q_param.detach().cpu())
		return params
	