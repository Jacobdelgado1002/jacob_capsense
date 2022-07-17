import numpy as np
import pybullet as p

from .env import AssistiveEnv
from .agents import furniture
from .agents.furniture import Furniture
from gym.utils import seeding

class ChestCleaningEnv(AssistiveEnv):
    def __init__(self, robot, human):
        super(ChestCleaningEnv, self).__init__(robot=robot, human=human, 
            task='chest_cleaning', 
            # obs_robot_len=(17 + len(robot.controllable_joint_indices) - (len(robot.wheel_joint_indices) if robot.mobile else 0)), 
            obs_robot_len=(3 + 4 + 3  + len(robot.controllable_joint_indices) - (len(robot.wheel_joint_indices) if robot.mobile else 0)), 
            obs_human_len=(18 + len(human.controllable_joint_indices)))
    
    def step(self, action):
        if self.human.controllable:
            action = np.concatenate([action['robot'], action['human']])
        # Execute the action. Step the simulator forward
        print("before take_step")
        self.take_step(action)

        obs = self._get_obs()

        # Get human preferences
        end_effector_velocity = np.linalg.norm(self.robot.get_velocity(self.robot.left_end_effector))
        preferences_score = self.human_preferences(end_effector_velocity=end_effector_velocity)

        # Define our reward function
        reward_distance = -min(self.tool.get_closest_points(self.human, distance=5.0)[-1])
        reward_action = -np.linalg.norm(action) # Penalize actions
        # reward_new_contact_points = self.new_contact_points # Reward new contact points on a person

        reward = self.config('distance_weight')*reward_distance + self.config('action_weight')*reward_action + preferences_score

        # if self.gui and self.tool_force_on_human > 0:
        #     print('Task success:', self.task_success, 'Force at tool on human:', self.tool_force_on_human, reward_new_contact_points)

        # info = {'task_success': int(self.task_success >= (self.total_target_count*self.config('task_success_threshold'))), 'action_robot_len': self.action_robot_len, 'action_human_len': self.action_human_len, 'obs_robot_len': self.obs_robot_len, 'obs_human_len': self.obs_human_len}
        info = {'task_success': int(self.task_success >= (self.config('task_success_threshold'))), 'action_robot_len': self.action_robot_len, 'action_human_len': self.action_human_len, 'obs_robot_len': self.obs_robot_len, 'obs_human_len': self.obs_human_len}

        

        done = self.iteration >= 200

        print("taking step is done")
        if not self.human.controllable:
            return obs, reward, done, info
        else:
            # Co-optimization with both human and robot controllable
            return obs, {'robot': reward, 'human': reward}, {'robot': done, 'human': done, '__all__': done}, {'robot': info, 'human': info}
    
    # def get_total_force(self):
    #     total_force_on_human = np.sum(self.robot.get_contact_points(self.human)[-1])
    #     tool_force = np.sum(self.tool.get_contact_points()[-1])
    #     tool_force_on_human = 0
    #     new_contact_points = 0
    #     for linkA, linkB, posA, posB, force in zip(*self.tool.get_contact_points(self.human)):
    #         total_force_on_human += force
    #         if linkA in [1]:
    #             tool_force_on_human += force
    #             # Only consider contact with human upperarm, forearm, hand
    #             if linkB < 0 or linkB > len(self.human.all_joint_indices):
    #                 continue

    #             indices_to_delete = []
    #             for i, (target_pos_world, target) in enumerate(zip(self.targets_pos_upperarm_world, self.targets_upperarm)):
    #                 if np.linalg.norm(posB - target_pos_world) < 0.025:
    #                     # The robot made contact with a point on the person's arm
    #                     new_contact_points += 1
    #                     self.task_success += 1
    #                     target.set_base_pos_orient(self.np_random.uniform(1000, 2000, size=3), [0, 0, 0, 1])
    #                     indices_to_delete.append(i)
    #             self.targets_pos_on_upperarm = [t for i, t in enumerate(self.targets_pos_on_upperarm) if i not in indices_to_delete]
    #             self.targets_upperarm = [t for i, t in enumerate(self.targets_upperarm) if i not in indices_to_delete]
    #             self.targets_pos_upperarm_world = [t for i, t in enumerate(self.targets_pos_upperarm_world) if i not in indices_to_delete]

    #             indices_to_delete = []
    #             for i, (target_pos_world, target) in enumerate(zip(self.targets_pos_forearm_world, self.targets_forearm)):
    #                 if np.linalg.norm(posB - target_pos_world) < 0.025:
    #                     # The robot made contact with a point on the person's arm
    #                     new_contact_points += 1
    #                     self.task_success += 1
    #                     target.set_base_pos_orient(self.np_random.uniform(1000, 2000, size=3), [0, 0, 0, 1])
    #                     indices_to_delete.append(i)
    #             self.targets_pos_on_forearm = [t for i, t in enumerate(self.targets_pos_on_forearm) if i not in indices_to_delete]
    #             self.targets_forearm = [t for i, t in enumerate(self.targets_forearm) if i not in indices_to_delete]
    #             self.targets_pos_forearm_world = [t for i, t in enumerate(self.targets_pos_forearm_world) if i not in indices_to_delete]

    #     return tool_force, tool_force_on_human, total_force_on_human, new_contact_points

    def _get_obs(self, agent=None):
        print("begining of get_obs")
        tool_pos, tool_orient = self.tool.get_pos_orient(1)
        tool_pos_real, tool_orient_real = self.robot.convert_to_realworld(tool_pos, tool_orient)
        robot_joint_angles = self.robot.get_joint_angles(self.robot.controllable_joint_indices)
        # Fix joint angles to be in [-pi, pi]
        robot_joint_angles = (np.array(robot_joint_angles) + np.pi) % (2*np.pi) - np.pi
        if self.robot.mobile:
            # Don't include joint angles for the wheels
            robot_joint_angles = robot_joint_angles[len(self.robot.wheel_joint_indices):]
        shoulder_pos = self.human.get_pos_orient(self.human.right_shoulder)[0]
        elbow_pos = self.human.get_pos_orient(self.human.right_elbow)[0]
        wrist_pos = self.human.get_pos_orient(self.human.right_wrist)[0]
        shoulder_pos_real, _ = self.robot.convert_to_realworld(shoulder_pos)
        elbow_pos_real, _ = self.robot.convert_to_realworld(elbow_pos)
        wrist_pos_real, _ = self.robot.convert_to_realworld(wrist_pos)
        # self.tool_force, self.tool_force_on_human, self.total_force_on_human, self.new_contact_points = self.get_total_force()
        # robot_obs = np.concatenate([tool_pos_real, tool_orient_real, robot_joint_angles, shoulder_pos_real, elbow_pos_real, wrist_pos_real, [self.tool_force]]).ravel()
        robot_obs = np.concatenate([tool_pos_real, tool_orient_real, robot_joint_angles, wrist_pos_real]).flatten()
        if agent == 'robot':
            return robot_obs
        if self.human.controllable:
            human_joint_angles = self.human.get_joint_angles(self.human.controllable_joint_indices)
            tool_pos_human, tool_orient_human = self.human.convert_to_realworld(tool_pos, tool_orient)
            shoulder_pos_human, _ = self.human.convert_to_realworld(shoulder_pos)
            elbow_pos_human, _ = self.human.convert_to_realworld(elbow_pos)
            wrist_pos_human, _ = self.human.convert_to_realworld(wrist_pos)
            human_obs = np.concatenate([tool_pos_human, tool_orient_human, human_joint_angles, shoulder_pos_human, elbow_pos_human, wrist_pos_human]).ravel()
            if agent == 'human':
                return human_obs
            # Co-optimization with both human and robot controllable
            return {'robot': robot_obs, 'human': human_obs}
        print("done of get_obs")
        
        return robot_obs

    def reset(self):
        # print("before reset!")
        super(ChestCleaningEnv, self).reset()
        # self.seed(seeding.create_seed())
        self.build_assistive_env('wheelchair')
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)

        self.prev_target_contact_pos = np.zeros(3)
        if self.robot.wheelchair_mounted:
            wheelchair_pos, wheelchair_orient = self.furniture.get_base_pos_orient()
            self.robot.set_base_pos_orient(wheelchair_pos + np.array(self.robot.toc_base_pos_offset[self.task]), [0, 0, -np.pi/2.0])

        # Set joint angles for human joints (in degrees)
        joints_positions = [(self.human.j_right_shoulder_x, 90), (self.human.j_right_elbow, 90), (self.human.j_left_elbow, -90), (self.human.j_right_hip_x, -90), (self.human.j_right_knee, 80), (self.human.j_left_hip_x, -90), (self.human.j_left_knee, 80)]
        joints_positions += [(self.human.j_right_shoulder_x, self.np_random.uniform(-90, 90)), (self.human.j_right_elbow, self.np_random.uniform(-90, 90)), (self.human.j_left_elbow, self.np_random.uniform(-90, 90))]
        self.human.setup_joints(joints_positions, use_static_joints=True, reactive_force=None)

        # import time
        # time.sleep(2)
        # self.reset()

        # # Update robot and human motor gains
        # self.robot.motor_gains = self.human.motor_gains = 0.025


        #  # * Setup human in the air, with legs and arms slightly seperated
        # joints_positions = [(self.human.j_right_shoulder_x, 90), (self.human.j_right_elbow, 90), (self.human.j_left_elbow, -90), (self.human.j_right_hip_x, -90), (self.human.j_right_knee, 80), (self.human.j_left_hip_x, -90), (self.human.j_left_knee, 80)]
        # self.human.setup_joints(joints_positions, use_static_joints=False, reactive_force=None)
        # # self.human.set_base_pos_orient([0, -0.2, 1.1], [-np.pi/2.0, 0, np.pi])

        # # * Add small variation to the body pose
        # motor_indices, motor_positions, motor_velocities, motor_torques = self.human.get_motor_joint_states()
        # # print(motor_positions)
        # self.human.set_joint_angles(motor_indices, motor_positions+self.np_random.uniform(-0.2, 0.2, size=len(motor_indices)))
        # # self.increase_pose_variation()
        # # * Increase friction of joints so human doesn't fail around exessively as they settle
        # # print([p.getDynamicsInfo(self.human.body, joint)[1] for joint in self.human.all_joint_indices])
        # self.human.set_whole_body_frictions(spinning_friction=2)
        

        shoulder_pos = self.human.get_pos_orient(self.human.right_shoulder)[0]
        elbow_pos = self.human.get_pos_orient(self.human.right_elbow)[0]
        wrist_pos = self.human.get_pos_orient(self.human.right_wrist)[0]

        # Initialize the tool in the robot's gripper
        # self.tool.init(self.robot, self.task, self.directory, self.id, self.np_random, right=False, mesh_scale=[1]*3)

        target_ee_pos = np.array([-0.6, 0.2, 1]) + self.np_random.uniform(-0.05, 0.05, size=3)
        target_ee_orient = self.get_quaternion(self.robot.toc_ee_orient_rpy[self.task])
        # base_position = self.init_robot_pose(target_ee_pos, target_ee_orient, [(target_ee_pos, target_ee_orient)], [(shoulder_pos, None), (elbow_pos, None), (wrist_pos, None)], arm='left', tools=[self.tool], collision_objects=[self.human, self.furniture], wheelchair_enabled=False)
        # self.init_robot_pose(target_ee_pos, target_ee_orient, [(target_ee_pos, target_ee_orient), (self.target_pos, None)], [(self.target_pos, target_ee_orient)], arm='right', tools=[self.tool], collision_objects=[self.human, self.table, self.furniture])
        # self.robot.skip_pose_optimization---------------------------------
        # if self.robot.wheelchair_mounted:
        #     # Load a nightstand in the environment for mounted arms
        #     self.nightstand = Furniture()
        #     self.nightstand.init('nightstand', self.directory, self.id, self.np_random)
        #     self.nightstand.set_base_pos_orient(np.array([-0.9, 0.7, 0]) + base_position, [0, 0, 0, 1])

        # Open gripper to hold the tool
        self.robot.set_gripper_open_position(self.robot.left_gripper_indices, self.robot.gripper_pos[self.task], set_instantly=True)

        
        self.generate_targets()

        if not self.robot.mobile:
            self.robot.set_gravity(0, 0, 0)
        self.human.set_gravity(0, 0, 0)
        self.tool.set_gravity(0, 0, 0)
        
        # distance from each electrode

        # print(self.tool.get_pos_orient(0))
        # print(self.tool.get_pos_orient(1))
        # print(self.tool.get_pos_orient(2))
        # print(self.tool.get_pos_orient(3))
        # print(self.tool.get_pos_orient(4))
        # print(self.tool.get_pos_orient(5))
        # print(self.tool.get_base_pos_orient())

        # target_total = self.targets_pos_forearm_world + self.targets_pos_upperarm_world
        # print(self.tool.controllable_joint_indices)
        # for i in (self.tool.controllable_joint_indices):
        #     dists = np.linalg.norm(target_total - self.tool.get_pos_orient(i)[0], axis=1)
        #     print(min(dists))
            
        # print(min(dists))
        # distance = np.linalg.norm(self.tool.get_pos_orient(1)[:,1:6] self.human ) 
        # print("The shortest distance is: ", -min(self.tool.get_closest_points(self.human, distance=5.0)[-1]))
        # print("The shortest distance is: ", min(dists))

        # Enable rendering

        self.init_env_variables()
        # print("reset is done!")
        return self._get_obs()

    def generate_targets(self):
        self.target_indices_to_ignore = []
        if self.human.gender == 'male':
            self.upperarm, self.upperarm_length, self.upperarm_radius = self.human.right_shoulder, 0.279, 0.043
            self.forearm, self.forearm_length, self.forearm_radius = self.human.right_elbow, 0.257, 0.033
        else:
            self.upperarm, self.upperarm_length, self.upperarm_radius = self.human.right_shoulder, 0.264, 0.0355
            self.forearm, self.forearm_length, self.forearm_radius = self.human.right_elbow, 0.234, 0.027

        self.target_on_arm = self.util.point_on_capsule(p1=np.array([0, 0, 0]), p2=np.array([0, 0, -self.forearm_length]), radius=self.forearm_radius, theta_range=(0, np.pi*2))
        arm_pos, arm_orient = self.human.get_pos_orient(self.forearm)
        target_pos, target_orient = p.multiplyTransforms(arm_pos, arm_orient, self.target_on_arm, [0, 0, 0, 1], physicsClientId=self.id)

        self.target = self.create_sphere(radius=0.01, mass=0.0, pos=target_pos, visual=True, collision=False, rgba=[0, 1, 1, 1])

        # self.targets_pos_on_upperarm = self.util.capsule_points(p1=np.array([0, 0, 0]), p2=np.array([0, 0, -self.upperarm_length]), radius=self.upperarm_radius, distance_between_points=0.03)
        # self.targets_pos_on_forearm = self.util.capsule_points(p1=np.array([0, 0, 0]), p2=np.array([0, 0, -self.forearm_length]), radius=self.forearm_radius, distance_between_points=0.03)
        
        # head_pos, head_orient = self.human.get_pos_orient(self.human.head)
        # target_pos, target_orient = p.multiplyTransforms(head_pos, head_orient, self.mouth_pos, [0, 0, 0, 1], physicsClientId=self.id)
        # self.target = self.create_sphere(radius=0.01, mass=0.0, pos=target_pos, collision=False, rgba=[0, 1, 0, 1])

        # self.targets_upperarm = self.create_spheres(radius=0.01, mass=0.0, batch_positions=[[0, 0, 0]]*len(self.targets_pos_on_upperarm), visual=True, collision=False, rgba=[0, 1, 1, 1])
        # self.targets_forearm = self.create_spheres(radius=0.01, mass=0.0, pos=targets_pos_on_forearm, visual=True, collision=False, rgba=[0, 1, 1, 1])

        # self.total_target_count = len(self.targets_pos_on_forearm)
        self.update_targets()

    def update_targets(self):
        # upperarm_pos, upperarm_orient = self.human.get_pos_orient(self.upperarm)
        # self.targets_pos_upperarm_world = []
        # for target_pos_on_arm, target in zip(self.targets_pos_on_upperarm, self.targets_upperarm):
        #     target_pos = np.array(p.multiplyTransforms(upperarm_pos, upperarm_orient, target_pos_on_arm, [0, 0, 0, 1], physicsClientId=self.id)[0])
        #     self.targets_pos_upperarm_world.append(target_pos)
        #     target.set_base_pos_orient(target_pos, [0, 0, 0, 1])

        # forearm_pos, forearm_orient = self.human.get_pos_orient(self.forearm)
        # self.targets_pos_forearm_world = []
        # for target_pos_on_arm, target in zip(self.targets_pos_on_forearm, self.targets_forearm):
        #     target_pos = np.array(p.multiplyTransforms(forearm_pos, forearm_orient, target_pos_on_arm, [0, 0, 0, 1], physicsClientId=self.id)[0])
        #     self.targets_pos_forearm_world.append(target_pos)
        #     target.set_base_pos_orient(target_pos, [0, 0, 0, 1])
        # head_pos, head_orient = self.human.get_pos_orient(self.human.head)
        # target_pos = ([0, 0, -self.upperarm_length]) 
        # self.target_pos = np.array(target_pos)
        # self.target.set_base_pos_orient(self.target_pos, [0, 0, 0, 1])
        arm_pos, arm_orient = self.human.get_pos_orient(self.forearm)
        target_pos, target_orient = p.multiplyTransforms(arm_pos, arm_orient, self.target_on_arm, [0, 0, 0, 1], physicsClientId=self.id)
        self.target_pos = np.array(target_pos)
        self.target.set_base_pos_orient(self.target_pos, [0, 0, 0, 1])

