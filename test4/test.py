import math
import time
import numpy as np
import robosuite as suite
import matplotlib.pyplot as plt
from robosuite.controllers import joint_vel
from robosuite.controllers.controller_factory import load_controller_config, controller_factory
from robosuite.models.grippers.gripper_model import GripperModel
from scipy.spatial.transform import rotation
from robosuite.wrappers import GymWrapper

from robosuite.models.grippers import GRIPPER_MAPPING
from model.panda_gripper import PandaGripper
from FrankaEnv import FrankaEnv
from utils import *

import matplotlib.pyplot as plt

# add custom gripper
GRIPPER_MAPPING["PandaGripper"] = PandaGripper

# show force data
show_data = False

# params for impedence control
control_freq = 500

controller_config = load_controller_config(custom_fpath='/home/zyh/tactile_manipulation/tactile_manipulation/test4/pose_control.json')
# controller_config = load_controller_config(default_controller="OSC_POSE")

# user_controller_config = controller_factory(
#     "JOINT_VELOCITY",
#     input
# )

env = suite.make(
        "FrankaEnv",
        robots=['Panda'],
        gripper_types="PandaGripper", 
        controller_configs=controller_config,
        use_object_obs=True,
        use_camera_obs=False,
        has_renderer=True,
        render_camera=None,
        has_offscreen_renderer=False,
        control_freq=control_freq,
        horizon=10000,
    )

obs = env.reset()
# env.render()

panda = env.robots[0]
panda._visualize_grippers(True)
# print(dir(panda.sim))
# print(panda.get_sensor_measurement('gripper0_unit_1_force'))

# 'robot0_joint1', 'robot0_joint2', 'robot0_joint3', 'robot0_joint4', 'robot0_joint5', 'robot0_joint6', 'robot0_joint7', 
# 'gripper0_finger_joint1', 'gripper0_finger_joint2', 'sphere_joint0'

# ============= set initial joints =============
# panda.sim.data.set_joint_qpos('robot0_joint1',1)
# panda.sim.data.set_joint_qpos('gripper0_finger_joint1',0.04)
# panda.sim.data.set_joint_qpos('gripper0_finger_joint2',-0.04)
# panda.sim.forward()

# ============= get recent joints =============
# print(panda.sim.data.get_joint_qpos('sphere_joint0'))

# 'gripper0_force_ee', 'gripper0_torque_ee', 
# 'gripper0_unit_1_force', 'gripper0_unit_2_force', 'gripper0_unit_3_force', 'gripper0_unit_4_force', 'gripper0_unit_5_force', 'gripper0_unit_6_force', 'gripper0_unit_7_force', 'gripper0_unit_8_force', 'gripper0_unit_9_force'
# ============= get sensor data =============
# print(panda.get_sensor_measurement('gripper0_unit_1_force'))

# end-effector target   eef_pos: [-0.472451602 -0.0003.62638892  0.600286243]   eef_quat: [-0.50022181  0.49950866  0.50009951 -0.5001697 ]

# before 
# ee_state [-0.47245160156968813, -0.00036263889212142185, 0.6002862430456579, 1.2096464996424185, -1.207921953777475, -1.2093507451873091]
# after 
# ee_state [-0.47245160156968813, -0.00036263889212142185, 0.6002862430456579, -1.2078945137033221, -1.2093232726556435, 1.2094930081999449]
# print(obs['robot0_eef_pos'])
# print(obs['robot0_eef_quat'])
ee_state = list(obs['robot0_eef_pos']) + list(quat2rotvec(obs['robot0_eef_quat']))
# print(ee_state)
# exit()
# ee_state = np.array(ee_state)
# print(ee_state)

n_actions = 200
actions = 0.01 * np.random.uniform(low=-1., high=1., size=(n_actions, env.action_spec[0].shape[0]))

# from robosuite.controllers.ik import InverseKinematicsController
# controller = InverseKinematicsController(panda.sim,)
# print(dir(controller))

# sensor data
t = 0
while True:
    t += 1
    env.render()

    # print( np.array(list(obs['robot0_eef_pos']) + list(quat2rotvec(obs['robot0_eef_quat']))))

    action = ee_state + list([0.04,-0.04])
    # action = ee_state + list([0.003,-0.003])
    # print('ee_state', ee_state)
    
    # self.sim.data.ctrl[:] = action
    obs, _,_,_ = env.step(actions[t])
    
    
    # if t >= 400:
    #     break
    # con_set = env.get_contacts(env.object)
    
    # print(t)
    # panda.sim.data.set_joint_qpos('gripper0_finger_joint1', 0.04 - t *0.0001)
    # panda.sim.data.set_joint_qpos('gripper0_finger_joint2',-0.04 + t *0.0001)
    # panda.sim.forward()
    
    # if not con_set:
    #     print(t)
    #     panda.sim.data.set_joint_qpos('gripper0_finger_joint1', 0.04 - t *0.0001)
    #     panda.sim.data.set_joint_qpos('gripper0_finger_joint2',-0.04 + t *0.0001)
    #     panda.sim.forward()
    # else:
    #     break

    # print(panda.get_sensor_measurement('gripper0_unit_1_force'))
    # print(panda.get_sensor_measurement('gripper0_unit_2_force'))
    # print(panda.get_sensor_measurement('gripper0_unit_3_force'))
    # print(panda.get_sensor_measurement('gripper0_unit_4_force'))
    # print(panda.get_sensor_measurement('gripper0_unit_5_force'))
    # print(panda.get_sensor_measurement('gripper0_unit_6_force'))
    # print(panda.get_sensor_measurement('gripper0_unit_7_force'))
    # print(panda.get_sensor_measurement('gripper0_unit_8_force'))
    # print(panda.get_sensor_measurement('gripper0_unit_9_force'))
    print('\n')



# Robot

# Controller
# InverseKinematicsController
