import numpy as np
import robosuite as suite

from PIL import Image
from IPython.display import display

print(suite.ALL_ENVIRONMENTS)
print(suite.ALL_ROBOTS)
print(suite.ALL_GRIPPERS)
print(suite.ALL_CONTROLLERS)

env = suite.make(
    env_name="Lift",
    robots="Sawyer",
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
)

obs = env.reset()

# frontview = Image.fromarray(obs['frontview_image'][::-1])
# display(frontview)

# agentview = Image.fromarray(obs['agentview_image'][::-1])
# display(agentview)

low, high = env.action_spec # get action limits
for i in range(100):
    action = np.random.uniform(low, high) # sample random action
    obs, reward, done, _ = env.step(action)
    env.render()  # render on display

# display(Image.fromarray(obs['frontview_image'][::-1]))

print('number of bodies:', env.sim.model.nbody)
print('number of joints:', env.sim.model.njnt)
print('number of generalized coordinates:', env.sim.model.nq)
print('number of degrees of freedom:', env.sim.model.nv)
print('number of degrees of freedom:', env.sim.model.nu)
print('number of activation states:', env.sim.model.na)

print(env.sim.model.body_names)

body_id = 1
print(env.sim.model.body_names[body_id])
print(env.sim.data.body_xpos[body_id])
print(env.sim.data.body_xquat[body_id])

body_id = -1
print()
print(env.sim.model.body_names[body_id])
print('Frame origin:\n', env.sim.data.body_xpos[body_id])
print('\nRotation quaternion:\n', env.sim.data.body_xquat[body_id])
print('\nRotation matrix:\n', env.sim.data.body_xmat[body_id].reshape(3,3))

# simple example of coordinate transformations
import robosuite.utils.transform_utils as T

print(T.quat2mat(T.convert_quat(np.array(env.sim.data.body_xquat[body_id]), to="xyzw")))

print(env.sim.data.body_xmat[body_id].reshape(3, 3))

print(env.sim.data.body_xpos[body_id])

# get information of all bodies
for i in range(env.sim.model.nbody):
    name = env.sim.model.body_names[i]
    body_id = env.sim.model.body_name2id(name)
    body_xpos = env.sim.data.body_xpos[body_id]
    print(body_id, name, body_xpos)