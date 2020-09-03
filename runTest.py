import random
import numpy as np
import torch
import time

import Environments.EnvironmentTest as EnvCarla

from Models.Networks import SimpleNetwork, DuelingNetwork, NoisyDuelingNetwork 
from utils.tool import read_config
import argparse

def run(agent, env):
		device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	
	#for episode in range(1):
			#initialized sequence
		obs = env.reset()
		done = False
		total_reward = 0
			#play the episode
		while not(done):				 
			y = agent(torch.tensor(obs[0], device=device, dtype=torch.float).unsqueeze(0),
						torch.tensor(obs[1], device=device, dtype=torch.float).unsqueeze(0))
			action = torch.argmax(y).item()				
			next_obs, reward, done, _ = env.step(action)	
			obs = next_obs	
			total_reward += reward

		print("reward:", total_reward)
 

		env.close_env()
	#env.show_recorder()

def create_agent(model, name_model, config):
	if model in ["DQN","D2QN"]:
		agent = SimpleNetwork.Network(config)

	elif model in ["D3QN"]:
		agent = DuelingNetwork.Network(config)
	elif model in ["Noisy","PER","ApeX"]:
		agent = NoisyDuelingNetwork.Network(config)

	agent.load_state_dict(torch.load(name_model))
	agent.eval()

	return agent

if __name__ == "__main__":

	parser = argparse.ArgumentParser(
			description='-')
	parser.add_argument('-model', type=str, default='ApeX', help='DQN/D2QN/D3QN/Noisy/PER/ApeX')
	parser.add_argument('-modelSaved', type=str, default='./ModelSavedKeep/ApexFinalModel25.pth', help='./path.pth')
	parser.add_argument('-p', type=int, default=2000, help='DQN/D2QN/D3QN/Noisy/PER')
	parser.add_argument('-namefile', type=str, default='new', help='DQN/D2QN/D3QN/Noisy/PER')
	args = parser.parse_args()

	config = read_config("./Configs/config" + args.model + ".yml")

	agent = create_agent(args.model, args.modelSaved, config)
	env = EnvCarla.CarEnv(0, args.p, args.namefile) 

	run(agent, env)