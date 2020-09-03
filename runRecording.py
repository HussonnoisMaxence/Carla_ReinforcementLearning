import glob
import os
import sys
try:
	sys.path.append(glob.glob('./Environments/carla-*%d.%d-%s.egg' % (
		sys.version_info.major,
		sys.version_info.minor,
		'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
	pass
import carla
import pathlib
import argparse

parser = argparse.ArgumentParser(
			description='choose your model')
parser.add_argument('-namefile', type=str, default='1081-score27')

args = parser.parse_args()
for count,letter in enumerate(args.namefile):
	if letter == "-":
		countL = count 
		break;


record_name = args.namefile
id_actor = int(args.namefile[:countL])
print(id_actor)
client = carla.Client('localhost', 2000)
client.set_timeout(5.0)
# Once we have a client we can retrieve the world that is currently running.
world = client.get_world()
world = client.load_world("Town02") 
world.set_weather(carla.WeatherParameters.ClearNoon)
client.replay_file(str(pathlib.Path().absolute())+"/recordes/"+str(record_name)+".log", 0, 400, id_actor)

#print(client.show_recorder_file_info("C:/Users/mhuss/Desktop/stgit/recordes/"+str(record_name)+".log", False))
client.set_replayer_time_factor(time_factor=2)

