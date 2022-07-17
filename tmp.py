import pybullet
import time
import pybullet_data 
import os

physicsClient = pybullet.connect(pybullet.GUI)

pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())

pybullet.resetSimulation()

# planeID = pybullet.loadURDF("bed.urdf")
directory = os.path.join('assistive_gym/assistive_gym/envs/assets/')
# plane = pybullet.loadURDF(os.path.join(directory, 'bed', 'bed.urdf'), physicsClientId=physicsClient)
plane = pybullet.loadURDF(os.path.join(directory, 'stretch', 'stretch.urdf'), physicsClientId=physicsClient)

# robot = pybullet.loadURDF("/home/rchi/Documents/Tool1/washing_tool1.urdf")

for i in range(1000000000000000):
	pybullet.stepSimulation(physicsClientId=physicsClient)

