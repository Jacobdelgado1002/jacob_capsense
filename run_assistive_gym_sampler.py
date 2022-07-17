# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 11:01:10 2021

@author: Simulation
"""

import os, gym
import numpy as np
import assistive_gym
# import assistive_gym.envs.validation_envs
from numpngw import write_png, write_apng
#from IPython.display import display, Image
import matplotlib
import matplotlib.pyplot as plt
import scipy.io as spio
import util_shared
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
#from IPython.display import clear_output
import cape
import pybullet as p

# arm_filename = 'arm_touchread_sensitive_ikea_6sensorsnewimproved_pi8_15cm_iteration_%d_movement_%s.pkl'
# leg_filename = 'leg_touchread_sensitive_ikea_6sensorsnewimproved_pi8_15cm_iteration_%d_movement_%s.pkl'
# directory = 'trainingdata'

# def load_data(participants=range(2, 14), arm=True, limb_segment='wrist', trajectory=None):
#     # Load collected capacitance data and ground truth pose
#     seg = {'wrist': 0, 'forearm': 1, 'upperarm': 2, 'ankle': 0, 'shin': 1, 'knee': 2}[limb_segment]
#     files = [os.path.join('participant_%d' % p, (arm_filename if arm else leg_filename) % (seg, wildcard)) for p in participants for wildcard in (['?', '??', '???'] if trajectory is None else [str(trajectory)])]
#     capacitance, pos_orient, times = util_shared.load_data_from_file(files=files, directory=directory)
#     print(np.shape(capacitance), np.shape(pos_orient))
#     pos_y, pos_z, angle_y, angle_z = pos_orient[:, 0], pos_orient[:, 1], pos_orient[:, 2], pos_orient[:, 3]
#     # return capacitance, pos_y, pos_z, angle_y, angle_z
#     return capacitance, pos_orient

# c_real, po = load_data(participants=[8], arm=True, limb_segment='forearm', trajectory=27)

c_real = np.array([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]])
po = np.array([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]])

# load mesh data for capacitive simulation
data = spio.loadmat('mesh_data_3stripe_electrodes.mat', squeeze_me=True)
nr_elements = data['NrElements']
nr_nodes = data['NrNodes']
inzid = data['inzid']
nodes = data['node']
roi_nodes = data['AllNodes']
electrode_nodes = data['elecNodes']
centroids = data['s']
nr_electrodes = 4
max_distance = 50
max_length = 50
nr_pixels = 64

# initialize capacitive simulation
cap = cape.CAPE(nr_electrodes, electrode_nodes, nr_elements, nr_nodes,
                roi_nodes, max_distance, max_length,
               nr_pixels, nodes, inzid, centroids)

# initialiize FEM matrices
cap.assembleSystem()

# assign boundary conditions to the problem -> first electrode
bnd_roi = np.zeros((roi_nodes.size, 1))
bnd_electrode = np.ones((electrode_nodes[0].size, 1))
bnd_vector = np.concatenate((roi_nodes, electrode_nodes[0]))
bnd_vals = np.concatenate((bnd_roi, bnd_electrode))

# compute boundary vector and matrix
K1, B1 = cap.generateBoundaryMatrices(bnd_vector, bnd_vals)
cap.K_full = cap.K_full + K1

# compute clusters based on mesh
cap.computeClusters()


#np.set_printoptions(suppress=True, precision=3)

# position on the human lib where the robot touches the forarm (startposition for measurements)
init_pos =  [-0.38, -0.84, 1.10]
init_orient = [0, np.pi/2, 0]

positions = np.insert(po, 0, [init_pos[0], init_pos[2], 0, 0], 0)

nr_points = positions.shape[0]

env = gym.make('ValidationSawyer-v1')
env.render()

env._max_episode_steps = 10000

observation = env.reset()
frames = []
nr_runs = 100
rgb_global = [None]*nr_runs
depth_cap = [None]*nr_runs
rgb_cap = [None]*nr_runs

plt.ion()
done = 0
counter = 0
loop_counter=0
while counter < nr_points:
    print("Go to next position\n")
    done = 0
    pos, orient = env.tool.get_base_pos_orient()#env.robot.get_pos_orient(env.robot.right_end_effector)
    if counter > 0:
        target_pos = [init_pos[0], init_pos[1]+positions[counter, 0], init_pos[2]+positions[counter, 1]]
        target_orient = [np.pi/2, np.pi/2-positions[counter,2], positions[counter,3]]
    else:
        target_pos = [init_pos[0], init_pos[1], 0.05+positions[counter, 1]] #0.02 accounts for sensor offset at tcp
        target_orient = [np.pi/2, np.pi/2, 0]

    target_joint_angles = env.robot.ik(env.robot.right_end_effector, target_pos, target_orient,
                                       env.robot.right_arm_ik_indices, max_iterations=1000, use_current_as_rest=True)

    while done == 0:
        loop_counter = loop_counter + 1

        # Step the simulation forward. Have the robot take a random action.
        current_joint_angles = env.robot.get_joint_angles(env.robot.right_arm_joint_indices)
        #print(current_joint_angles)

        pos, orient = env.tool.get_base_pos_orient()#env.robot.get_pos_orient(env.robot.right_end_effector)
        rot_mat = p.getMatrixFromQuaternion(orient)

        eye_pos = [pos[0],pos[1],pos[2]-0.03]
        view_vector = np.dot(np.reshape(np.array(rot_mat),(3,3)),eye_pos)
        rgb_cap_img, depth_img = env.setup_cap_sensor(camera_eye=eye_pos, camera_target=[-view_vector[0], -view_vector[1], view_vector[2]*np.cos(np.pi)])

        depth_cap[counter] = np.array(depth_img)
        rgb_cap[counter] = np.array(rgb_cap_img)

        cap.depth_data = depth_cap[counter]
        cap.rgb_data = 50 * np.ones((64, 64, 4))
        cap.meshMasking()
        cap.solveSystem(K1, B1)
        print("Capacitance: ", cap.cap_vector[counter])

        action = (target_joint_angles - current_joint_angles)*2
        observation, reward, done, info = env.step(action)

        if np.linalg.norm(action) < 0.1:
            counter = counter + 1
            done = 1