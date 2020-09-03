import glob
import os
import sys

import random
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import time


def run(config, agent, env):
	agent_name = config["model_name"]
	print("Test " + agent_name + " start")
	try:
		writer =  SummaryWriter(comment=agent_name)
		print("Training phase")
		for episode in range(config["episode_batch"]):
			#initialized sequence
			obs = env.reset()
			done = False
			total_reward = 0
			#play the episode
			while not(done):				 
				action = agent.choose_action(obs)				
				next_obs, reward, done, _ = env.step(action)	

				#store transition in replay
				agent.add_transition(obs, action, reward, next_obs, done)
				agent.train_nn()

				obs = next_obs
				total_reward += reward
			
			writer.add_scalar("Score", total_reward, episode)
			if agent.loss:
				writer.add_scalar("Loss", agent.loss, episode)
			#print("step:", episode,"reward:", total_reward,"eps:", agent.epsilon)

		writer.close()

	finally:
		print("Test " + agent_name + " end")
		env.close_env()
		agent.save_model()#save models parameters
