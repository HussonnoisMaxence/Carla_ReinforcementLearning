
import multiprocessing
from multiprocessing import Process, Manager
from multiprocessing.managers import BaseManager
import numpy as np

from Models.DistributedModels.Apex.Learner import Learner
from Models.DistributedModels.Apex.Actor import run
from Models.DistributedModels.Apex.PrioritizedReplayBuffer import Memory
from Models.DistributedModels.Apex.Server import Server


from utils.tool import read_config, AutoProxy

multiprocessing.managers.AutoProxy = AutoProxy

"""
def choose_epsilon_min():
	return np.random.choice(4*[0.05] + 3*[0.01] + 3*[0.1])
"""

def runApeX():
	config = read_config("./Configs/configApeX.yml")
	epsilons = config["min_epsilons"]
	N_ACTORS = config['n_actors']
   	
	BaseManager.register('ReplayBM', Memory)
	BaseManager.register('LearnerBM', Learner)
	BaseManager.register('ServerBM', Server)
	manager = BaseManager()
	manager.start()
	server = manager.ServerBM()
	replay = manager.ReplayBM(config)


	learner = manager.LearnerBM(config)
	
	processes = [Process(target=run, args=(replay, learner, server, config, epsilons[p], p)) for p in range(N_ACTORS) ]

	p_learner = Process(target=learner.train_nn,  args=(replay, server,))
	
	p_learner.start()
	for p in processes:
	  p.start()
  
	
	p_learner.join()
	for p in processes:
	  p.join()