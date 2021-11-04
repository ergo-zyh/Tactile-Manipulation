import numpy as np
import robosuite as suite
import time
from robosuite.controllers.controller_factory import load_controller_config, controller_factory

controller_config = load_controller_config(default_controller="OSC_POSE")

# create environment instance
env = suite.make(
    env_name="Lift", # try with other tasks like "Stack" and "Door"
    robots="Panda",  # try with other robots like "Sawyer" and "Jaco"
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    controller_configs=controller_config
)

# reset the environment
env.reset()
# action 第6、7：夹持距离
action = np.random.randn(env.robots[0].dof - 1) # sample random action
n_actions = 200
actions = 0.1* np.random.uniform(low=-1., high=1., size=(n_actions, env.action_spec[0].shape[0]))
# action = [ 0.25853008, -1.29166783, -0.83451379, -1.19240517, -0.66191375, -0.80748019,  0.40211025, 0.40211025]
# [ 0.25853008 -1.29166783 -0.83451379 -1.19240517 -0.66191375 -0.80748019 0.40211025  0.40211025]

for i in range(200):

    # action = np.random.randn(env.robots[0].dof) # sample random action
    # x = np.random.randn(1)
    # action[0] += x * 0.1
    # action[5] = action[6]

    # actions[i][2] = 0
    obs, reward, done, info = env.step(actions[i])  # take action in the environment
    print('action', actions[i])
    # print('obs', obs) # joint_pos_cos, joint_pos_sin, joint_vel, eef_pose, eef_quat, gripper_qpos, gripper_qvel, cube_pos, cube_quat, gripper_to_cube_pos, proprio_state
    # print('reward', reward)
    # print('info', info)
    env.render()  # render on display    