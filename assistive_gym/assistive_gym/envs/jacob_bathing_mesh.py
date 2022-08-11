import os
from gym import spaces
import numpy as np
import pybullet as p

from .jacob_bathing import ChestCleaningEnv
from .agents import furniture
from .agents.furniture import Furniture

class ChestCleaningMeshEnv(ChestCleaningEnv):
    def __init__(self, robot, human):
        # super(ChestCleaningMeshEnv, self).__init__(robot=robot, human=human)
        super(ChestCleaningEnv, self).__init__(robot=robot, human=human, task='chest_cleaning', 
        obs_robot_len=(14 + len(robot.controllable_joint_indices) - (len(robot.wheel_joint_indices) if robot.mobile else 0)), 
        obs_human_len=(15 + len(human.controllable_joint_indices)))
        self.general_model = True
        # Parameters for personalized human participants
        self.gender = 'male'
        # self.body_shape_filename = '%s_1.pkl' % self.gender
        self.human_height = 1.6

    def reset(self):
        super(ChestCleaningEnv, self).reset()
        self.build_assistive_env('wheelchair')
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0, physicsClientId=self.id)
        self.furniture.set_on_ground()
        if self.robot.wheelchair_mounted:
            wheelchair_pos, wheelchair_orient = self.furniture.get_base_pos_orient()
            self.robot.set_base_pos_orient(wheelchair_pos + np.array(self.robot.toc_base_pos_offset[self.task]), [0, 0, -np.pi/2.0])

        if self.general_model:
            # Randomize the human body shape
            # gender = self.np_random.choice(['male', 'female'])
            gender = 'male'
            # body_shape = self.np_random.randn(1, self.human.num_body_shape)
            body_shape = self.np_random.uniform(-2, 5, (1, self.human.num_body_shape))
            # human_height = self.np_random.uniform(1.59, 1.91) if gender == 'male' else self.np_random.uniform(1.47, 1.78)
            human_height = self.np_random.uniform(1.5, 1.9)
        else:
            gender = self.gender
            body_shape = self.body_shape_filename
            human_height = self.human_height

        # Randomize human pose
        joint_angles = [(self.human.j_left_hip_x, -90), (self.human.j_right_hip_x, -90), (self.human.j_left_knee_x, 70), (self.human.j_right_knee_x, 70), (self.human.j_left_shoulder_z, -45), (self.human.j_right_shoulder_z, 0), (self.human.j_left_elbow_y, -90), (self.human.j_right_elbow_y, 90)]
        # u = self.np_random.uniform
        # joint_angles += [(self.human.j_waist_x, u(-30, 45)), (self.human.j_waist_y, u(-45, 45)), (self.human.j_waist_z, u(-30, 30)), (self.human.j_lower_neck_x, u(-30, 30)), (self.human.j_lower_neck_y, u(-30, 30)), (self.human.j_lower_neck_z, u(-10, 10)), (self.human.j_upper_neck_x, u(-45, 45)), (self.human.j_upper_neck_y, u(-30, 30)), (self.human.j_upper_neck_z, u(-30, 30))]
        # joint_angles += [(self.human.j_waist_x, u(-20, 30)), (self.human.j_waist_y, u(-45, 0)), (self.human.j_waist_z, u(0, 30)), (self.human.j_lower_neck_x, u(-30, 30)), (self.human.j_lower_neck_y, u(-30, 30)), (self.human.j_lower_neck_z, u(-10, 10)), (self.human.j_upper_neck_x, u(-30, 30)), (self.human.j_upper_neck_y, u(-30, 30)), (self.human.j_upper_neck_z, u(-30, 30))]
        joint_angles += [(j, self.np_random.uniform(-10, 10)) for j in (self.human.j_waist_x, self.human.j_waist_y, self.human.j_waist_z, self.human.j_lower_neck_x, self.human.j_lower_neck_y, self.human.j_lower_neck_z, self.human.j_upper_neck_x, self.human.j_upper_neck_y, self.human.j_upper_neck_z)]
        self.human.init(self.directory, self.id, self.np_random, gender=gender, height=human_height, body_shape=body_shape, joint_angles=joint_angles, position=[0, 0, 0], orientation=[0, 0, 0])

        # Place human in chair
        chair_seat_position = np.array([0, 0.15, 0.40])
        self.human.set_base_pos_orient(self.furniture.get_base_pos_orient()[0] + chair_seat_position - self.human.get_vertex_positions(self.human.bottom_index), [0, 0, 0, 1])

        print("Time to generate targets")
        self.generate_target()

        # p.resetDebugVisualizerCamera(cameraDistance=1.10, cameraYaw=40, cameraPitch=-45, cameraTargetPosition=[-0.2, 0, 0.75], physicsClientId=self.id)

        elbow_pos, elbow_orient = self.human.get_pos_orient(self.human.right_elbow)
        wrist_pos, wrist_orient = self.human.get_pos_orient(self.human.right_wrist)
        target_pos, _ = np.divide(elbow_pos + wrist_pos, 2), np.divide(elbow_orient + wrist_orient, 2)
        # self.target_cur_pos = target_pos + np.array([0, 0, 0.08])
        self.target_cur_pos = target_pos + np.array([0, 0, 0.05])

        # Open gripper to hold the tool
        self.robot.set_gripper_open_position(self.robot.right_gripper_indices, self.robot.gripper_pos[self.task], set_instantly=True)
        # # Initialize the tool in the robot's gripper
        self.tool.init(self.robot, self.task, self.directory, self.id, self.np_random, right=True, mesh_scale=[0.08]*3) #0.08

        if not self.robot.mobile:
            self.robot.set_gravity(0, 0, 0)
        self.human.set_gravity(0, 0, 0)
        self.tool.set_gravity(0, 0, 0)

        # p.setPhysicsEngineParameter(numSubSteps=4, numSolverIterations=10, physicsClientId=self.id)


        # Enable rendering
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)


        self.init_env_variables()
        return self._get_obs()

    def generate_target(self):
        # Set targets
        # self.targets_pos_on_upperarm = self.util.capsule_points(p1=np.array([0, 0, 0]), p2=np.array([0, 0, -self.upperarm_length]), radius=self.upperarm_radius, distance_between_points=0.03)
        # self.targets_pos_on_forearm = self.util.capsule_points(p1=np.array([0, 0, 0]), p2=np.array([0, 0, -self.forearm_length]), radius=self.forearm_radius, distance_between_points=0.03)
        
        # self.targets_upperarm = self.create_spheres(radius=0.01, mass=0.0, batch_positions=[[0, 0, 0]]*len(self.targets_pos_on_upperarm), visual=True, collision=False, rgba=[0, 1, 1, 1])
        # self.targets_forearm = self.create_spheres(radius=0.01, mass=0.0, batch_positions=[[0, 0, 0]]*len(self.targets_pos_on_forearm), visual=True, collision=False, rgba=[0, 1, 1, 1])
        # self.total_target_count = len(self.targets_pos_on_upperarm) + len(self.targets_pos_on_forearm)
        # if self.human.gender == 'male':
        #     self.upperarm, self.upperarm_length, self.upperarm_radius = self.human.right_shoulder, 0.279, 0.043
        #     self.forearm, self.forearm_length, self.forearm_radius = self.human.right_elbow, 0.257, 0.033
        # else:
        #     self.upperarm, self.upperarm_length, self.upperarm_radius = self.human.right_shoulder, 0.264, 0.0355
        #     self.forearm, self.forearm_length, self.forearm_radius = self.human.right_elbow, 0.234, 0.027

        # self.target_on_arm = self.util.point_on_capsule(p1=np.array([0, 0, 0]), p2=np.array([0, 0, -self.forearm_length]), radius=self.forearm_radius, theta_range=(0, np.pi*2))
        # arm_pos, arm_orient = self.human.get_pos_orient(self.forearm)
        # target_pos, target_orient = p.multiplyTransforms(arm_pos, arm_orient, self.target_on_arm, [0, 0, 0, 1], physicsClientId=self.id)

        # self.target = self.create_sphere(radius=0.01, mass=0.0, pos=target_pos, visual=True, collision=False, rgba=[0, 1, 1, 1])
        # elbow_pos, _ = self.human.get_pos_orient(self.human.right_elbow)
        # shoulder_pos, _  = self.human.get_pos_orient(self.human.right_wrist)
        # midpoint = (elbow_pos + shoulder_pos) / 2
        # midpoint = midpoint

        elbow_pos, elbow_orient = self.human.get_pos_orient(self.human.right_elbow)
        wrist_pos, wrist_orient = self.human.get_pos_orient(self.human.right_wrist)
        # target_pos, target_orient = (elbow_pos + wrist_pos) / 2, (elbow_orient + wrist_orient) / 2
        target_pos, _ = np.divide(elbow_pos + wrist_pos, 2), np.divide(elbow_orient + wrist_orient, 2)
        # vertex_index = self.human.right_arm_center
        # self.target_pos = self.human.get_vertex_positions(vertex_index)
        self.target_pos = target_pos + np.array([0, 0, 0.03])
        print("This is the target position mesh:" , target_pos)
        self.target = self.create_sphere(radius=0.01, mass=0.0, pos=self.target_pos, visual=True, collision=False, rgba=[0, 1, 1, 1])

    def update_targets(self):
        pass