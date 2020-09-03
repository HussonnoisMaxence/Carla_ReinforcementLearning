# Carla_ReinforcementLearning


## Table of contents
* [Contributeurs](#contributeurs)
* [Prérequis](#Prérequis)
* [Quick start](#Quickstart)
* [Run Ape-X](#RunApe-X)
* [Run Test](#RunTest)
* [Run Records](#RunRecords)

## Contributeurs
- HUSSONNOIS Maxence 
- JUN KIM Jae yun

## Prérequis :

- Ubuntu
-	Anaconda
- Simulateur Carla 0.9.9

## QuickStart

### Carla installation

Add the CARLA 0.9.9 repository to the system.
```
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 92635A407F7A020C
sudo add-apt-repository "deb [arch=amd64 trusted=yes] http://dist.carla.org/carla-0.9.9/ all main"
```
Install CARLA and check for the installation in the /opt/ folder
```
sudo apt-get update
sudo apt-get install carla-simulator
cd /opt/carla-simulator
```


### Launch the simulator
Moove into the carla folder

```
cd /opt/carla-simulator
./bin/CarlaUE4.sh -server -carla-rpc-port=2000
```

### Launch the trainning of an agent with a reinforcement learning algorithm
Get in the github repository
Create the environment virtual. (It is just for the first time)
```
conda env create -f ev.yml
```
Activate the environment virtual
```
conda activate ev
```
Launch the python script

```
python runTrainning.py -model DQN
```

### Track progress with tensorboard
To launch tensorboard with the server:

Before connecting to the server
```
ssh -N -F -L localhost:16006:localhost:6006 server@adresse
```
Connect to the server
Get in the github repository, activate the environment virtual
```
conda activate ev
tensorboard --logdir runs --port 6006
```
Open your navigator to the localhost 6006 OR 16006 if you are connected from the server




## RunApe-X
1. Launch two simulation with port 2000 and 2004

2. In another terminal

Get in the github repository, activate the environment virtual, and launch the python script
```
conda activate ev
python runTranning.py -model ApeX
```
## RunTest
1. Launch one simulation with port 2000 

2. In another terminal

Get in the github repository, activate the environment virtual, and launch the python script
```
conda activate ev
python runTest.py
```
## RunRecords
1. Launch one simulation with port 2000 

2. In another terminal

Get in the github repository, activate the environment virtual, and launch the python script
```
conda activate ev
python runRecording.py
```
