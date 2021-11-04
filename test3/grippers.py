import robosuite as suite
from robosuite.controllers import load_controller_config

import random
from PIL import Image
from IPython.display import display

suite.ALL_GRIPPERS

def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

for robot in ['Sawyer', 'Panda', 'Jaco', 'Kinova3', 'IIWA', 'UR5e']:

    for gripper in suite.ALL_GRIPPERS:

        options = dict()
        options["env_name"] = "Lift"
        options["robots"] = [robot]

        options["controller_configs"] = [
            load_controller_config(default_controller="OSC_POSE"),
        ]

        options["gripper_types"] = None
        options["gripper_types"] = gripper

        env = suite.make(
            **options,
            has_renderer=False,
            ignore_done=True,
            use_camera_obs=True,
            camera_names="frontview",
        )

        print(robot, gripper, '(DoF: %d)' % env.action_dim)

        frontview = env.sim.render(height=256, width=256, camera_name="frontview")[::-1]
        agentview = env.sim.render(height=256, width=256, camera_name="agentview")[::-1]
        im_frontview = Image.fromarray(frontview)
        im_agentview = Image.fromarray(agentview)
        display(get_concat_h(im_frontview, im_agentview))

