import torch
import numpy as np
import time
import random, sys
from Models.Networks.NoisyDuelingNetwork import Network
import Environments.Environment as EnvCarla
from torch.utils.tensorboard import SummaryWriter

from utils.tool import read_config, read_seed

config = read_config("./Configs/configApeX.yml")
N_ACTIONS = config["n_actions"]
EPSILON_DECAY = config["epsilon_decay"]
UPDATE_REPLAY = config['update_replay']
UPDATE_NETWORK_TIME = config['update_network']



np.random.seed(read_seed())

class Actor:
	def __init__ (self, config, num, epsilon_min):
		#Define env
		self.epsilon = config['epsilon']

		self.epsilon_min = epsilon_min

		self.local_replay = []

		self.num = num

		self.Q = Network(config)

		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	def save_model(self):
		torch.save(self.Q.state_dict(),"./ModelSaved/Actor" + str(self.epsilon_min)+".pth")
		print("Model saved")


	def choose_action(self, obs):
		#Choose a action according to greedy epsilon

		if np.random.uniform() < self.epsilon:
			action = np.random.choice(N_ACTIONS)
		else:
			y = self.Q(torch.tensor(obs[0], device=self.device, dtype=torch.float).unsqueeze(0),
						torch.tensor(obs[1],  device=self.device, dtype=torch.float).unsqueeze(0))
			action = torch.argmax(y).item()
		
		return action

	def update_epsilon(self):
		self.epsilon *= EPSILON_DECAY
		self.epsilon = max(self.epsilon, self.epsilon_min)

	def add_transition(self, obs, action, reward, next_obs, done):
		self.local_replay.append([obs, action, self.reward_clipping(reward), next_obs, done]) 

	def reward_clipping(self, reward):
		if reward > 1:
			reward = 1
		elif reward <-1:
			reward = -1
		return reward



def run(replay, learner, server, config, epsilon_min, num):
	env = EnvCarla.CarEnv(num)
	actor = Actor(config, num, epsilon_min=epsilon_min)
	update_network_count = 0

	writer =  SummaryWriter(comment="Actor=" +str(num) +"-eps="+ str(epsilon_min))

	for episode in range(config["episode_batch"]):
		obs = env.reset()
		done = False
		total_reward = 0
	
		update_network_count +=1 
		t = time.time()
		while not(done):
			action = actor.choose_action(obs)
			#perform action
			next_obs, reward, done, _ = env.step(action)           
			#store transition in replay
			actor.add_transition(obs, action, reward, next_obs, done)  
			obs = next_obs
			total_reward += reward  
			actor.update_epsilon()
		timestamp = time.time() - t
		if len(actor.local_replay) > UPDATE_REPLAY:
			#print(str(num)+" Send transitions")
			td_errors = learner.compute_td(actor.local_replay)
			replay.add_all(actor.local_replay, td_errors)# appel remote object method
			actor.local_replay = []

		if update_network_count > UPDATE_NETWORK_TIME:
			params = server.get_params()            
			if params:
				for q_param, param in zip(actor.Q.parameters(),
										   params):
					new_param = torch.Tensor(param)
					q_param.data.copy_(new_param)
				#print(str(num)+" Get Parameter")
			update_network_count = 0
		if not(episode%5000):
			torch.save(actor.Q.state_dict(),"./ModelSaved/Actor" + str(actor.epsilon_min) + str(episode) + ".pth")
		writer.add_scalar("Score", total_reward, episode)
		writer.add_scalar("Time/Episode", timestamp, episode)
		writer.add_scalar("epsilon", actor.epsilon, episode)
	
	actor.save_model()
	env.close_env()
	writer.close()