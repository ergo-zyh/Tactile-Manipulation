import random
import numpy as np
import robosuite as suite
from robosuite.controllers import load_controller_config

from PIL import Image
from IPython.display import display

from robosuite.utils.transform_utils import *
from transform import *

# ['Lift', 'Stack', 'NutAssembly', 'NutAssemblySingle', 'NutAssemblySquare', 'NutAssemblyRound', 'PickPlace', 'PickPlaceSingle', 'PickPlaceMilk', 'PickPlaceBread', 'PickPlaceCereal', 'PickPlaceCan', 'Door', 'Wipe', 'TwoArmLift', 'TwoArmPegInHole', 'TwoArmHandover']
# print(suite.ALL_ENVIRONMENTS)

# ['JOINT_VELOCITY', 'JOINT_TORQUE', 'JOINT_POSITION', 'OSC_POSITION', 'OSC_POSE', 'IK_POSE']
# print(suite.ALL_CONTROLLERS)

# ['Sawyer', 'Baxter', 'Panda', 'Jaco', 'Kinova3', 'IIWA', 'UR5e'])
# print(suite.ALL_ROBOTS)

# 'baxter'
# print(suite.robots.BIMANUAL_ROBOTS)

options = dict()
options["env_name"] = "TwoArmPegInHole"
options["robots"] = [
    random.choice(["Sawyer", "Panda"]),
    random.choice(["Sawyer", "Panda"])
]

options["controller_configs"] = [
    load_controller_config(default_controller="OSC_POSE"),
    load_controller_config(default_controller="OSC_POSE"),
]

options["env_configuration"] = "single-arm-parallel"

options["robots"]

env = suite.make(
    **options,
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    camera_names="frontview",
)

frontview = env.sim.render(height=256, width=256, camera_name="frontview")[::-1]
agentview = env.sim.render(height=256, width=256, camera_name="agentview")[::-1]
display(Image.fromarray(frontview), Image.fromarray(agentview))

for r in env.robots:
    print(r.name)
    print(r.controller.name)
    print(r.controller_config)    
    print(r.controller.qpos_index)
    print(r.controller.qvel_index)
    print(r.controller.joint_index)
    print()

