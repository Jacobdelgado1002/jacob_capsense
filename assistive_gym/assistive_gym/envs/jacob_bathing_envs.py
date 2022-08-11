from .jacob_bathing import ChestCleaningEnv
from .jacob_bathing_mesh import ChestCleaningMeshEnv
from .agents import pr2, baxter, sawyer, jaco, stretch, panda, human, human_mesh
from .agents.pr2 import PR2
from .agents.baxter import Baxter
from .agents.sawyer import Sawyer
from .agents.jaco import Jaco
from .agents.stretch import Stretch
from .agents.panda import Panda
from .agents.human import Human
from .agents.human_mesh import HumanMesh
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune.registry import register_env

robot_arm = 'left'
human_controllable_joint_indices = human.right_arm_joints
class ChestCleaningPR2Env(ChestCleaningEnv):
    def __init__(self):
        super(ChestCleaningPR2Env, self).__init__(robot=PR2(robot_arm), human=Human(human_controllable_joint_indices, controllable=False))

class ChestCleaningBaxterEnv(ChestCleaningEnv):
    def __init__(self):
        super(ChestCleaningBaxterEnv, self).__init__(robot=Baxter(robot_arm), human=Human(human_controllable_joint_indices, controllable=False))

class ChestCleaningSawyerEnv(ChestCleaningEnv):
    def __init__(self):
        super(ChestCleaningSawyerEnv, self).__init__(robot=Sawyer(robot_arm), human=Human(human_controllable_joint_indices, controllable=False))

class ChestCleaningJacoEnv(ChestCleaningEnv):
    def __init__(self):
        super(ChestCleaningJacoEnv, self).__init__(robot=Jaco(robot_arm), human=Human(human_controllable_joint_indices, controllable=False))

class ChestCleaningStretchEnv(ChestCleaningEnv):
    def __init__(self):
        super(ChestCleaningStretchEnv, self).__init__(robot=Stretch('wheel_'+robot_arm), human=Human(human_controllable_joint_indices, controllable=False))

class ChestCleaningPandaEnv(ChestCleaningEnv):
    def __init__(self):
        super(ChestCleaningPandaEnv, self).__init__(robot=Panda(robot_arm), human=Human(human_controllable_joint_indices, controllable=False))

class ChestCleaningPR2HumanEnv(ChestCleaningEnv, MultiAgentEnv):
    def __init__(self):
        super(ChestCleaningPR2HumanEnv, self).__init__(robot=PR2(robot_arm), human=Human(human_controllable_joint_indices, controllable=True))
register_env('assistive_gym:ChestCleaningPR2Human-v1', lambda config: ChestCleaningPR2HumanEnv())

class ChestCleaningBaxterHumanEnv(ChestCleaningEnv, MultiAgentEnv):
    def __init__(self):
        super(ChestCleaningBaxterHumanEnv, self).__init__(robot=Baxter(robot_arm), human=Human(human_controllable_joint_indices, controllable=True))
register_env('assistive_gym:ChestCleaningBaxterHuman-v1', lambda config: ChestCleaningBaxterHumanEnv())

class ChestCleaningSawyerHumanEnv(ChestCleaningEnv, MultiAgentEnv):
    def __init__(self):
        super(ChestCleaningSawyerHumanEnv, self).__init__(robot=Sawyer(robot_arm), human=Human(human_controllable_joint_indices, controllable=True))
register_env('assistive_gym:ChestCleaningSawyerHuman-v1', lambda config: ChestCleaningSawyerHumanEnv())

class ChestCleaningJacoHumanEnv(ChestCleaningEnv, MultiAgentEnv):
    def __init__(self):
        super(ChestCleaningJacoHumanEnv, self).__init__(robot=Jaco(robot_arm), human=Human(human_controllable_joint_indices, controllable=True))
register_env('assistive_gym:ChestCleaningJacoHuman-v1', lambda config: ChestCleaningJacoHumanEnv())

class ChestCleaningStretchHumanEnv(ChestCleaningEnv, MultiAgentEnv):
    def __init__(self):
        super(ChestCleaningStretchHumanEnv, self).__init__(robot=Stretch('wheel_'+robot_arm), human=Human(human_controllable_joint_indices, controllable=True))
register_env('assistive_gym:ChestCleaningStretchHuman-v1', lambda config: ChestCleaningStretchHumanEnv())

class ChestCleaningPandaHumanEnv(ChestCleaningEnv, MultiAgentEnv):
    def __init__(self):
        super(ChestCleaningPandaHumanEnv, self).__init__(robot=Panda(robot_arm), human=Human(human_controllable_joint_indices, controllable=True))
register_env('assistive_gym:ChestCleaningPandaHuman-v1', lambda config: ChestCleaningPandaHumanEnv())

class ChestCleaningPR2MeshEnv(ChestCleaningMeshEnv):
    def __init__(self):
        super(ChestCleaningPR2MeshEnv, self).__init__(robot=PR2(robot_arm), human=HumanMesh())

class ChestCleaningBaxterMeshEnv(ChestCleaningMeshEnv):
    def __init__(self):
        super(ChestCleaningBaxterMeshEnv, self).__init__(robot=Baxter(robot_arm), human=HumanMesh())

class ChestCleaningSawyerMeshEnv(ChestCleaningMeshEnv):
    def __init__(self):
        super(ChestCleaningSawyerMeshEnv, self).__init__(robot=Sawyer(robot_arm), human=HumanMesh())

class ChestCleaningJacoMeshEnv(ChestCleaningMeshEnv):
    def __init__(self):
        super(ChestCleaningJacoMeshEnv, self).__init__(robot=Jaco(robot_arm), human=HumanMesh())

class ChestCleaningStretchMeshEnv(ChestCleaningMeshEnv):
    def __init__(self):
        super(ChestCleaningStretchMeshEnv, self).__init__(robot=Stretch('wheel_'+robot_arm), human=HumanMesh())
        # super(ChestCleaningStretchMeshEnv, self).__init__(robot=Stretch(robot_arm), human=HumanMesh())

class ChestCleaningPandaMeshEnv(ChestCleaningMeshEnv):
    def __init__(self):
        super(ChestCleaningPandaMeshEnv, self).__init__(robot=Panda(robot_arm), human=HumanMesh())

