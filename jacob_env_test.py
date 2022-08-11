import os, gym
from tkinter.tix import Tree
import numpy as np
import assistive_gym
from numpngw import write_png, write_apng
#from IPython.display import display, Image
import matplotlib
import matplotlib.pyplot as plt
import scipy.io as spio
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
#from IPython.display import clear_output
import cape
import meshio
# import warnings
# warnings.filterwarnings("error")


from assistive_gym.envs.env import AssistiveEnv
from assistive_gym.envs.agents.stretch import Stretch
from assistive_gym.envs.jacob_bathing_envs import ChestCleaningStretchMeshEnv
from assistive_gym.envs.agents.human_mesh import HumanMesh
import pybullet as p
import time

# use meshio to read the vtk file produced by gmsh
msh = meshio.read('/home/rchi/Documents/Tool1/Washing_tool1/tool1.vtk', file_format="vtk")

# used for debugging
# print(msh)
# tetra = msh.get_cells_type("tetra")
# print(tetra.shape[0])
# exit()

# get all the nodes by using the meshio .points attribute 
nodes = msh.points
# print(nodes)
# exit()

# get the inzid by using the meshio .get_cells_type("tetra") attriv\bute. This return all the tetrahedrals in the mesh
inzid = msh.get_cells_type("tetra")

# num_cells, points_cell = np.shape(inzid.data)
# print(num_cells)
# exit()
# inzid = msh.cells[3][1]

# get the number of nodes
nr_nodes = nodes.shape[0]

# get the number of elements
nr_elements = inzid.shape[0]

# six electrodes

# get the region of interest nodes for creating the bounding matrix. These are the nodes at the surface of the mesh. 
# They will have a z axis equal to 0, 50 or 0.40 in our case
# For other cases, the 50 would be dimensions of the area of interest in cm and the 0.40 would be the height of the electrodes.

# naive way of getting roi nodes. use when debugging
# roi_nodes = np.arange(0,2123)

roi_nodes = []
for idx, node in enumerate(nodes):
    if(0 in node or 50 in node or 0.40 in node):
        roi_nodes.append(idx)

roi_nodes =  np.asarray(roi_nodes)

# used for debugging. Allows one to view the roi_nodes and compare with vtk file
# for i in roi_nodes:
#     print(nodes[i])
# # exit()

# get the nodes that encompass the boundaries for the electrode nodes
# TODO: fix code bellow
# the electrode nodes can be defined manually by uncommenting the code bellow.. 
# This can be done by looking at your mesh and adding the nodes that bound your electrodes in their respective lists

electrode_nodes = []
# elec_1 = np.array([9, 10, 11, 12, 251], dtype=np.uint8)
# elec_2 = np.array([13, 14, 15, 16, 252], dtype=np.uint8)
# elec_3 = np.array([5, 6, 7, 8, 250], dtype=np.uint8)
# elec_4 = np.array([17, 18, 19, 20, 253], dtype=np.uint8)
# elec_5 = np.array([1, 2, 3, 4, 249], dtype=np.uint8)
# elec_6 = np.array([21, 22, 23, 24, 254], dtype=np.uint8)
# electrode_nodes.append(elec_1)
# electrode_nodes.append(elec_2)
# electrode_nodes.append(elec_3)
# electrode_nodes.append(elec_4)
# electrode_nodes.append(elec_5)
# electrode_nodes.append(elec_6)

# get the electrode nodes automatically by iterating over all nodes adding the nodes that contain a z axis
# representing the height you chose for your electrodes. In this case, the height is 0.40 cm.
# WARNING: might not work for other file formats that are not vtk.
# vtk files define the electrode nodes of each electrode sequentially. Thus, the following code
# adds the first 4 consecutive nodes with a z axis of 0.40 as one electrode and afterwards adds the first of the
# center nodes to this electrode (in this case 248). Then, it repeats this process until it has computed 6 different electrodes
elec_counter = 0
electrode = []
elec_center_counter = 248
for idx, sublist in enumerate(nodes):

    if len(electrode_nodes) == 6:
        break

    if elec_counter == 4:
        elec_counter = 0
        electrode.append(elec_center_counter)
        elec_center_counter = elec_center_counter + 1
        electrode_add = np.array(electrode)
        electrode_nodes.append(electrode_add)
        electrode = []
        # print(electrode_nodes)
        # exit()


    if(sublist[2] == 0.40):
        electrode.append(idx)
        elec_counter = elec_counter + 1

electrode_nodes = np.asarray(electrode_nodes)


# calculate the centroids for each element. Each element is a list of 4 nodes where each nodes contains an x,y,z
centroids = []

for sublist in(inzid):
    val = np.array([0, 0, 0])    
    for index in sublist: # there will be 4 elements in the sublist
        val = val + nodes[index] # x, y,z position of the element in the sublist
    val = val / 4 # this is the centroid of the tethedra
    centroids.append(val)

centroids = np.asarray(centroids)

# number of electrodes
nr_electrodes = 6
#max distance to measure
max_distance = 50
max_length = 50
# number of pixels for the camera
nr_pixels = 64

# initialize capacitive simulation
cap = cape.CAPE(nr_electrodes, electrode_nodes, nr_elements, nr_nodes,
                roi_nodes, max_distance, max_length,
               nr_pixels, nodes, inzid, centroids)

# initialiize FEM matrices
cap.assembleSystem()

# assign boundary conditions to the problem -> first electrode
bnd_roi = np.zeros((roi_nodes.size, 1))

# Here I was trying to include all 6 electrodes to plot their individual self capacitance but I'm not sure if this is correct***********
total_size = 0

for electrode in electrode_nodes:
    total_size += electrode.size
    
bnd_electrode = np.ones((total_size, 1))

bnd_vector = roi_nodes
for electrode in electrode_nodes:
    bnd_vector = np.concatenate((bnd_vector, electrode))

bnd_vals = np.concatenate((bnd_roi, bnd_electrode))

# original code assign boundary conditions to the problem -> first electrode
# bnd_roi = np.zeros((roi_nodes.size, 1))
# bnd_electrode = np.ones((electrode_nodes[0].size, 1))
# bnd_vector = np.concatenate((roi_nodes, electrode_nodes[0]))
# bnd_vals = np.concatenate((bnd_roi, bnd_electrode))


# compute boundary vector and matrix
K1, B1 = cap.generateBoundaryMatrices(bnd_vector, bnd_vals)
cap.K_full = cap.K_full + K1

# compute clusters based on mesh
cap.computeClusters()


#np.set_printoptions(suppress=True, precision=3)

# create environment with normal model
# env = gym.make('ChestCleaningStretch-v1')

#create environment with SMPLX model
# env = ChestCleaningStretchMeshEnv()
# env = gym.make('ChestCleaningStretch-v1')
env = gym.make('ChestCleaningStretchMesh-v1')
env.render()

env._max_episode_steps = 10000

# Setup a global camera in the environment for scene capturing
# env.setup_camera(camera_eye=[-0.6, -0.4, 2], camera_target=[0.2, 0.2, 0], fov=50, camera_width=512, camera_height=512)
env.setup_camera(camera_eye=[-0.6, -0.4, 2], camera_target=[0.2, 0.2, 0], fov=50, camera_width=512, camera_height=512)
nr_runs = 100
observation = env.reset()
frames = []
rgb_global = [None]*nr_runs
depth_cap = [None]*nr_runs
rgb_cap = [None]*nr_runs


# run the simulation with the stretch movement and plot the change in capacitance and distance
capacitance_list = []
distance_list = []
for i in range(nr_runs):
    if i % 50 == 0:
        print(i)
        print()
        print()

    # Step the simulation forward. Have the robot take a random action.
    # observation, reward, done, info = env.step(env.action_space.sample())
    pos, orient = env.tool.get_base_pos_orient()#env.robot.get_pos_orient(env.robot.right_end_effector)
    current_pos, current_orient = env.robot.get_pos_orient(env.robot.right_end_effector)

    pos, orient = env.robot.get_pos_orient(env.robot.left_end_effector)
    pos_real, orient_real = env.robot.convert_to_realworld(pos, orient)

    # render image from global camera
    global_img, _ = env.get_camera_image_depth()
    rgb_global[i] = np.array(global_img)

    # Setup local camera for capacitive sensor
    rgb_cap_img, depth_img = env.setup_cap_sensor(camera_eye=[pos[0]+0.05, pos[1], pos[2]-0.05], camera_target=[pos[0], pos[1], -pos[2]])
    depth_cap[i] = np.array(depth_img)
    rgb_cap[i] = np.array(rgb_cap_img)

    cap.depth_data = depth_cap[i]
    cap.rgb_data = 50 * np.ones((64, 64, 4))
    cap.meshMasking()
    cap.solveSystem(K1, B1)
    print("Capacitance: ", cap.cap_vector[i])
    capacitance_list.append(cap.cap_vector[i])
    distance = np.linalg.norm(current_pos - env.target_cur_pos)
    distance_list.append(distance)



    action = np.array([0, 0, 0, 0, 0])
    # for trajectory #1 (straight-movement through arm)
    if current_pos[1] < env.target_cur_pos[1]:
        action = np.array([0.1, 0.1, 0, 0, 0])
    elif current_pos[2] < env.target_cur_pos[2]:
        action = np.array([0, 0, 1, 0, 0])
    else:
        action = np.array([0, 0, 0, 1, 0])

    observation, reward, done, info = env.step(action)

np.save('./tmp_capacitance.npy', np.array(capacitance_list))
np.save('./tmp_distance.npy', np.array(distance_list))

fig, ax = plt.subplots()
x1 = []
x2 = []
y1 = distance_list
x1.extend(range(len(distance_list)))
y2 = capacitance_list
x2.extend(range(len(capacitance_list)))
# ax.plot(x1, y1, label='distance', c = 'r')
ax.plot(x2, y2, label='capacitance', c = 'b')
# plt.plot(x1, y1)
# plt.plot(x2, y2)
plt.title("Capacitance vs Time (s)")
plt.xlabel('Time (s)')
plt.ylabel('Capacitance')
# plt.legend()
plt.show()
exit()


# the code bellow can be ignored as it was just used for trying to debug previous issues********************************

# done = 0
# counter = 0
# loop_counter=0
# elbow_pos, elbow_orient = env.human.get_pos_orient(env.human.right_elbow)
# wrist_pos, wrist_orient = env.human.get_pos_orient(env.human.right_wrist)
# # target_pos, target_orient = (elbow_pos + wrist_pos) / 2, (elbow_orient + wrist_orient) / 2
# target_pos, target_orient = np.divide(elbow_pos + wrist_pos, 2), np.divide(elbow_orient + wrist_orient, 2)
# print("This is the elbow position:", elbow_pos)
# print("This is the wrist position:", wrist_pos)
# print("This is the target position:", target_pos)
# # exit()
# flag = False
# while done == 0 and loop_counter < 200000: 
#     loop_counter = loop_counter + 1

#     # Step the simulation forward. Have the robot take a random action.
#     current_joint_angles = env.robot.get_joint_angles(env.robot.right_arm_joint_indices)
#     #print(current_joint_angles)

#     pos, orient = env.tool.get_base_pos_orient()#env.robot.get_pos_orient(env.robot.right_end_effector)
#     current_pos, current_orient = env.robot.get_pos_orient(env.robot.right_end_effector)
#     rot_mat = p.getMatrixFromQuaternion(orient)

#     eye_pos = [pos[0],pos[1],pos[2]-0.03]
#     view_vector = np.dot(np.reshape(np.array(rot_mat),(3,3)),eye_pos)
#     rgb_cap_img, depth_img = env.setup_cap_sensor(camera_eye=eye_pos, camera_target=[-view_vector[0], -view_vector[1], view_vector[2]*np.cos(np.pi)])

#     depth_cap[counter] = np.array(depth_img)
#     rgb_cap[counter] = np.array(rgb_cap_img)

    
#     cap.depth_data = depth_cap[counter]
#     cap.rgb_data = 50 * np.ones((64, 64, 4))
#     cap.meshMasking()
#     cap.solveSystem(K1, B1)
#     print("Capacitance: ", cap.cap_vector[counter])

#     # action = (target_joint_angles - current_joint_angles)*2
#     # print("target joint anlges shape: ", target_joint_angles.shape)
#     # print(target_joint_angles)
#     # print("current joint angle shape: ", current_joint_angles.shape)
#     # print(current_joint_angles)
#     # print("This is action:")
#     # print(action)
#     # print(type(action))
#     # exit()

#     # ac = np.array([1.86939636, 0.23742873, -3.14021474], dtype=float)
#     # print("Array")
#     # print(ac)
#     # print(type(ac))
#     # action = np.array([1.0, 0, 0]) 
#     # action = np.array([0.0, 0, *action])
    
#     # take random actions to test
#     # action = np.random.rand(5)
#     # action = np.array([0, 0, 1, 1, 1, 1, 1, 1])
#     # action = np.random.rand(10)
#     # action[[0, 1]] = 0 
    
    
#     # action = np.array([0.0, 0, *action])
#     print(loop_counter)
#     print(current_pos)
#     print(env.target_cur_pos)
#     # exit()
    
#     action = np.array([0, 0, 0, 0, 0])
#     # for trajectory #1 (straight-movement through arm)
#     if current_pos[1] < env.target_cur_pos[1]:
#         action = np.array([0.1, 0.1, 0, 0, 0])
#     elif current_pos[2] < env.target_cur_pos[2]:
#         action = np.array([0, 0, 1, 0, 0])
#     else:
#         action = np.array([0, 0, 0, 1, 0])

#     # for trajectory #2 (vertical movement)
#     # if not flag:
#     #     if current_pos[1] < env.target_cur_pos[1]:
#     #         action = np.array([0.1, 0.1, 0, 0, 0])
#     #     elif current_pos[2] < 1.00:
#     #         action = np.array([0, 0, 1, 0, 0])
#     #     elif current_pos[0] < env.target_cur_pos[0]:
#     #         action = np.array([0, 0, 0, 0.5, 0])
#     #     else:
#     #         time.sleep(2)
#     #         flag = True

#     # if flag and current_pos[2] > env.target_cur_pos[2]:
#     #     action = np.array([0, 0, -0.1, 0, 0])

#     # for trajectory #3 (horizontal movement)
#     # if current_pos[2] < env.target_cur_pos[2]:
#     #     action = np.array([0, 0, 1, 0, 0])
#     # elif current_pos[0] < env.target_cur_pos[0]:
#     #     action = np.array([0, 0, 0, 1, 0])
#     # elif current_pos[1] < (env.target_cur_pos[1] + 0.05):
#     #     action = np.array([1, 1, 0, 0, 0])
#     # else:
#     #     time.sleep(2)
#     #     flag = True

#     observation, reward, done, info = env.step(action)

#     if np.linalg.norm(action) < 0.1:
#         counter = counter + 1
#         done = 1
        
# while done == 0 and loop_counter < 200000: 
#     loop_counter = loop_counter + 1

#     # Step the simulation forward. Have the robot take a random action.
#     current_joint_angles = env.robot.get_joint_angles(env.robot.right_arm_joint_indices)
#     #print(current_joint_angles)

#     pos, orient = env.tool.get_base_pos_orient()#env.robot.get_pos_orient(env.robot.right_end_effector)
#     rot_mat = p.getMatrixFromQuaternion(orient)

#     eye_pos = [pos[0],pos[1],pos[2]-0.03]
#     view_vector = np.dot(np.reshape(np.array(rot_mat),(3,3)),eye_pos)
#     rgb_cap_img, depth_img = env.setup_cap_sensor(camera_eye=eye_pos, camera_target=[-view_vector[0], -view_vector[1], view_vector[2]*np.cos(np.pi)])

#     depth_cap[counter] = np.array(depth_img)
#     rgb_cap[counter] = np.array(rgb_cap_img)

#     cap.depth_data = depth_cap[counter]
#     cap.rgb_data = 50 * np.ones((64, 64, 4))
#     cap.meshMasking()
#     cap.solveSystem(K1, B1)
#     print("Capacitance: ", cap.cap_vector[counter])

#     # action = (target_joint_angles - current_joint_angles)*2
#     # print("target joint anlges shape: ", target_joint_angles.shape)
#     # print(target_joint_angles)
#     # print("current joint angle shape: ", current_joint_angles.shape)
#     # print(current_joint_angles)
#     # print("This is action:")
#     # print(action)
#     # print(type(action))
#     # exit()

#     # ac = np.array([1.86939636, 0.23742873, -3.14021474], dtype=float)
#     # print("Array")
#     # print(ac)
#     # print(type(ac))
#     # action = np.array([1.0, 0, 0]) 
#     # action = np.array([0.0, 0, *action])
    
#     # take random actions to test
#     # action = np.random.rand(5)
#     # action = np.array([0, 0, 1, 1, 1, 1, 1, 1])
#     # action = np.random.rand(10)
#     # action[[0, 1]] = 0 
    
    
#     # action = np.array([0.0, 0, *action])
#     print(loop_counter)
#     if loop_counter < 100:
#         action = np.array([0, 0, 1, 0, 0])
#     else:
#         action = np.array([0, 0, 0, 1, 0])

#     observation, reward, done, info = env.step(action)

#     if np.linalg.norm(action) < 0.1:
#         counter = counter + 1
#         done = 1
# done = 0
# counter = 0
# loop_counter=0
# print("The number of points is: ", nr_points)
