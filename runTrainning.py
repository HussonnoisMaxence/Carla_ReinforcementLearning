
import Environments.Environment as EnvCarla
import argparse

from Models.NoDistributedModels import ModelDQN, ModelD2QN, ModelD3QN, ModelNoisy, ModelPER, TrainModel
from Models.DistributedModels.Apex.Main import runApeX
from utils.tool import read_config, read_seed, write_seed

if __name__ == "__main__":

	parser = argparse.ArgumentParser(
			description='choose your model')
	parser.add_argument('-model', type=str, default='DQN', help='DQN/D2QN/D3QN/Noisy/PER/ApeX')
	parser.add_argument('-seed', type=int, default=-1, help='save/init')
	

	args = parser.parse_args()

	#seeds
	if args.seed == -1:
		config = read_config("./Configs/config" + args.model + ".yml")
		write_seed(config["seed"])
	else:
		write_seed(args.seed)
	print("seed:", read_seed())

	#Models
	if args.model == "ApeX":
		runApeX()
	elif args.model == "DQN":
		env = EnvCarla.CarEnv(0)
		agent = ModelDQN.Agent()
		TrainModel.run(config, agent, env)
			
	elif args.model == "D2QN":
		env = EnvCarla.CarEnv(0)
		agent = ModelD2QN.Agent()
		TrainModel.run(config, agent, env)

	elif args.model == "D3QN":
		env = EnvCarla.CarEnv(0)
		agent = ModelD3QN.Agent()
		TrainModel.run(config, agent, env)

	elif args.model == "Noisy":
		env = EnvCarla.CarEnv(0)
		agent = ModelNoisy.Agent()
		TrainModel.run(config, agent, env)

	elif args.model == "PER":
		agent = ModelPER.Agent()
		env = EnvCarla.CarEnv(0)
		TrainModel.run(config, agent, env)

	else:
		exit()

		
	


	
		


