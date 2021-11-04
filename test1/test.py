from Cython.Compiler.Nodes import IfClauseNode
import mujoco_py as mp
import os
import math


model = mp.load_model_from_path('/home/zyh/tactile_manipulation/tactile_manipulation/test1/robot_description/franka/franka.xml')
sim = mp.MjSim(model)
viewer = mp.MjViewer(sim)
t = 0

# print(dir(sim.data))

# 控制器
# 设置初始值
  
sim.data.set_joint_qpos("panda_joint1", 1)
sim.data.set_joint_qpos("panda_joint2", 1)
sim.data.set_joint_qpos("panda_joint3", 1)
sim.data.set_joint_qpos("panda_joint4", 0)
sim.data.set_joint_qpos("panda_joint5", 0)
sim.data.set_joint_qpos("panda_joint6", 0)
sim.data.set_joint_qpos("panda_joint7", 0)
sim.data.set_joint_qpos("panda_finger_joint1", 0.0)
sim.data.set_joint_qpos("panda_finger_joint2", 0.0)
sim.forward()

sim_state = sim.get_state()
print(sim_state)
while True:
    # print(sim.data.sensordata)
    # sim.data.ctrl[7] = -0.01
    t += 1
    sim.step()
    
    viewer.render()
    if t > 100 and os.getenv('TESTING') is not None:
        break