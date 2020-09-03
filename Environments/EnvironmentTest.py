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

import random
import time
import numpy as np
import cv2
import math

from utils.tool import read_config, rgb_to_gray
import pathlib

config = read_config("./Configs/confEnvironmentTest.yml")

SECONDS_PER_EPISODE = config['TIME_WP']
#Image's size
IMG_WIDTH = config['IMG_WIDTH']
IMG_HEIGHT = config['IMG_HEIGHT']	

SHOW_CAM = config['SHOW_PREVIEW']


#Distance between two waypoints
NAV_I = config['NAV_I']
START_POSITION = config['position']
TRAJECTORY = config['trajectory']

#Norme
VMAX = config['VMAX']
SMAXX = config['SMAXX']
SMAXY = config['SMAXY']
AMAX = config['AMAX']



class CarEnv:

	front_camera = None
	def __init__(self, num, port, testName):

		print(num,  port)
		self.client = carla.Client(config['host'], port)
		self.client.set_timeout(5.0)
		

		# Once we have a client we can retrieve the world that is currently running.
		self.world = self.client.get_world()
		self.world = self.client.load_world(config['town']) 
		self.world.set_weather(carla.WeatherParameters.ClearNoon)
		# The world contains the list blueprints that we can use for adding new 
		# actors into the simulation.
		blueprint_library = self.world.get_blueprint_library()

		# Now let's filter all the blueprints of type 'vehicle' and choose the model tesla
		self.actor_list = []
		self.collision_hist = []
		self.line_hist = []
		self.model_3 = blueprint_library.filter(config['model_car'])[0]
		self.vehicle = None
		if self.vehicle is not None:
			transform = self.world.get_map().get_spawn_points()[START_POSITION]
			self.vehicle = self.world.try_spawn_actor(self.model_3, transform)
		while self.vehicle is None: #fix the fail of spawing the vehicle
			transform = self.world.get_map().get_spawn_points()[START_POSITION]
			self.vehicle = self.world.try_spawn_actor(self.model_3, transform)
		self.actor_list.append(self.vehicle)

		self.client.start_recorder(str(pathlib.Path().absolute())+"/recordes/"+str(self.vehicle.id)+"-"+testName+".log")
		self.id_actor = None

	def reset(self):

		self.id_actor =  self.vehicle.id
		
		#Creating the RGB cam
		rgb_cam = self.world.get_blueprint_library().find('sensor.camera.rgb')
		rgb_cam.set_attribute('image_size_x', f'{IMG_WIDTH}')
		rgb_cam.set_attribute('image_size_y', f'{IMG_HEIGHT}')
		rgb_cam.set_attribute('fov', '110')

		transform = carla.Transform(carla.Location(x=2.5, z=0.7))
		sensor = self.world.spawn_actor(rgb_cam, transform, attach_to=self.vehicle)
		
		self.actor_list.append(sensor)
		sensor.listen(lambda data: self.process_img(data))

		###
		self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0)) #Apparently it speed up the processes to get things started, to be confirmed

		time.sleep(1) # sleep to get things started and to not detect a collision when the car spawns/falls from sky.

		#Waiting that the camera is up
		while self.front_camera is None:
			time.sleep(0.01)

			
		#Creating the Navigation and waypoints
		maps = self.world.get_map()
		self.waypoint = maps.get_waypoint(self.vehicle.get_location(), project_to_road = True, lane_type=carla.LaneType.Driving)
		self.waypoint = self.waypoint.next(NAV_I)[TRAJECTORY[0]]
		self.waypoint_reached_counter = 0 #counter of waypoint reached
		self.waypoints_time_counter = 0 #counter of waypoint reached after episode time
		
		#Creating the collission sensor
		colsensor = self.world.get_blueprint_library().find('sensor.other.collision')
		colsensor = self.world.spawn_actor(colsensor, transform, attach_to=self.vehicle)
		self.actor_list.append(colsensor)
		colsensor.listen(lambda event: self.collision_data(event))
		
		linesensor = self.world.get_blueprint_library().find('sensor.other.lane_invasion')
		linesensor = self.world.spawn_actor(linesensor, transform, attach_to=self.vehicle)
		self.actor_list.append(linesensor)
		linesensor.listen(lambda event: self.line_data(event))
		#Create clock
		self.episode_start = time.time()

		self.vehicle.apply_control(carla.VehicleControl(brake=0.0, throttle=0.0)) #again ?
		
		#Observation the image, the car's localisation, the waypoint localisation and the car's speed
		obs = (self.front_camera,[
			   self.vehicle.get_location().x/SMAXX,
			   self.vehicle.get_location().y/SMAXY,
			   self.waypoint.transform.location.x/SMAXX,
			   self.waypoint.transform.location.y/SMAXY,
			   self.vehicle.get_velocity().x/VMAX,
			   self.vehicle.get_velocity().y/VMAX,
			   self.vehicle.get_angular_velocity().x/AMAX,
			   self.vehicle.get_angular_velocity().y/AMAX])
		
		return obs

	#Sensors functions
	def collision_data(self, event):
		self.collision_hist.append(event)

	def line_data(self, event):
		self.line_hist.append(event.crossed_lane_markings[0])

	def process_img(self, image):
		i = np.array(image.raw_data)
		i2 = i.reshape((IMG_HEIGHT, IMG_WIDTH, 4))
		rgb_weights = [0.2989, 0.5870, 0.1140]
		i4 = np.dot(i2[...,:3], rgb_weights)
		self.front_camera = i4/255.0
		
	def compute_d(self):
		vloc = self.vehicle.get_location()
		wloc = self.waypoint.transform.location
		wnp = np.array([wloc.x, wloc.y,wloc.z])
		vnp = np.array([vloc.x, vloc.y,vloc.z])
		return np.linalg.norm(vnp - wnp)
	
	def new_waypoint(self, d):
		reached = False
		if d<config['radius_wp']:
			self.waypoint_reached_counter += 1
			self.waypoint = self.waypoint.next(NAV_I)[TRAJECTORY[self.waypoint_reached_counter%len(TRAJECTORY)]]
			reached = True
		return reached

	def road_exit(self):
		out = False
		if len(self.line_hist) != 0:
			for el in self.line_hist:
				if el.type.name == "NONE":
					print('True')
					out = True
		return out

	def reward_function(self, action):
		#Distance waypoint reward
		waypoint_reached = self.new_waypoint(self.compute_d())
		if len(self.collision_hist) != 0: 
			done = True       
			reward = -1


		elif self.waypoint_reached_counter == len(TRAJECTORY):
			done = True
			reward = 1
			
		elif waypoint_reached :
			done = False
			reward = 1
		else:
			done = False
			reward = 0
				
		if self.episode_start + SECONDS_PER_EPISODE < time.time():
			if self.waypoints_time_counter==self.waypoint_reached_counter:
				done = True
			else:
				self.waypoints_time_counter = self.waypoint_reached_counter
				self.episode_start = time.time()

				done = False
		#print("Action:",action,"reward:", reward)
		if done:
			print("end")
		return reward, done



	def step(self, action):
		
		# Forward at different speed
		if action == 0: #accélération
			self.vehicle.apply_control(carla.VehicleControl(throttle=config['throttle'], steer=config['steer']))	
		if action == 1: #accélération
			self.vehicle.apply_control(carla.VehicleControl(throttle=config['throttle'], steer=-config['steer']))	
		if action == 2: #Avancé ligne droite vitesse constante
			self.vehicle.apply_control(carla.VehicleControl(throttle=config['throttle'], steer=0))	
		
		time.sleep(0.7)
		reward, done = self.reward_function(action)
		obs = (self.front_camera,[
			   self.vehicle.get_location().x/SMAXX,
			   self.vehicle.get_location().y/SMAXY,
			   self.waypoint.transform.location.x/SMAXX,
			   self.waypoint.transform.location.y/SMAXY,
			   self.vehicle.get_velocity().x/VMAX,
			   self.vehicle.get_velocity().y/VMAX,
			   self.vehicle.get_angular_velocity().x/AMAX,
			   self.vehicle.get_angular_velocity().y/AMAX
			   ])
		
		return obs, reward, done, None

	def close_env(self):
		for actor in self.actor_list:
			actor.destroy()

		self.client.stop_recorder()


	def show_recorder(self):
		self.client.replay_file("C:/Users/mhuss/Desktop/stgit/y.log", 0, 30, self.id_actor)
		self.client.set_replayer_time_factor(time_factor=1)