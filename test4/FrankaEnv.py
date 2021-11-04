from sys import modules
import numpy as np
import robosuite as suite
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models import arenas
from robosuite.models.tasks import ManipulationTask
from robosuite.models.arenas import EmptyArena

from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler

from robosuite.models.objects import BoxObject

class FrankaEnv(SingleArmEnv):
    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        mount_types="default",
        gripper_types="default",
        initialization_noise="default",
        reward_scale=1.0,
        use_object_obs=False,
        use_camera_obs=True,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera=None,
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
    ):
        
        self.reward_scale = reward_scale
        self.use_object_obs = use_object_obs 

        super().__init__(
            robots, 
            env_configuration=env_configuration, 
            controller_configs=controller_configs, 
            mount_types=mount_types, 
            gripper_types=gripper_types, 
            initialization_noise=initialization_noise, 
            use_camera_obs=use_camera_obs, 
            has_renderer=has_renderer, 
            has_offscreen_renderer=has_offscreen_renderer, 
            render_camera=render_camera, 
            render_collision_mesh=render_collision_mesh, 
            render_visual_mesh=render_visual_mesh, 
            render_gpu_device_id=render_gpu_device_id, 
            control_freq=control_freq, 
            horizon=horizon, 
            ignore_done=ignore_done, 
            hard_reset=hard_reset, 
            camera_names=camera_names, 
            camera_heights=camera_heights, 
            camera_widths=camera_widths, 
            camera_depths=camera_depths)
    
    # 载入环境必要模型，创建arena，robot，object
    def _load_model(self):
        super()._load_model()
    
        arena = EmptyArena()

        # self.object = BoxObject(
        #     name = "box",
        #     size = [0.02, 0.02, 0.02],
        #     density = 100000,
        #     friction = [1, 0.005, 0.0001],
        #     rgba = [0, 0.5, 0.5, 1],
        #     solref = [1000 -500],           # [0.004, 1]
        #     solimp = [0.9, 0.95, 0.001],
        #     material = "default"
        #     # joints=[dict(type="free", damping="0")]
        # )

        self.object = BoxObject(
            name = "box",
            size = [0.02, 0.02, 0.02],
            rgba = [0, 0.5, 0.5, 1]
        )

        self.model = ManipulationTask(
            mujoco_arena=arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.object,
        )

    # Reset Env states
    def _reset_internal(self):
        super()._reset_internal()
        
        # Reset robot pose
        init_qpos = np.array([2.94, -0.574, 0.103, -2.687, -0.218, 3.704, -0.544, 0.04, -0.04])
        num_joints = len(self.robots[0]._ref_joint_pos_indexes)
        self.sim.data.qpos[np.array(list(self.robots[0]._ref_joint_pos_indexes) + list([num_joints,num_joints+1]))] = init_qpos
        
        # Reset all object positions using initializer sampler if we're not directly loading from an xml

        # Reset object pose
        self.box_id = self.sim.model.body_name2id(self.object.root_body)
        eef_pos_xy = self.sim.data.site_xpos[self.robots[0].eef_site_id][:2]
        # self.sim.data.set_joint_qpos(self.box.joints[0], [eef_pos_xy[0], eef_pos_xy[1]+0.1,1,0,0,0,1])
        # self.sim.data.set_joint_qpos(self.object.joints[0], [-0.473, 0, 0.6, 0, 0, 0, 1])
        self.sim.data.set_joint_qpos(self.object.joints[0], [-0.473, 0, 2, 0, 0, 0, 1])


    # 初始化obs,用到装饰器@sensor
    def _setup_observables(self):
        observables = super()._setup_observables()

        if self.use_object_obs:
            pf = self.robots[0].robot_model.naming_prefix
            modality = "object"

            @sensor(modality=modality)
            def obj_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.box_id])

            @sensor(modality=modality)
            def obj_qpos(obs_cache):
                return np.array(self.sim.data.body_xquat[self.box_id])
            
            @sensor(modality=modality)
            def obj_to_eef_pos(obs_cache):
                return obs_cache["obj_pos"] - obs_cache[f"{pf}eef_pos"] if\
                    "obj_pos" in obs_cache and f"{pf}eef_pos" in obs_cache else np.zeros(3)

            sensors = [obj_pos, obj_qpos, obj_to_eef_pos]
            names = [s.__name__ for s in sensors]

            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq
                )
        return observables


    def reward(self, action):
        return 1.0
        
    def step(self, action):
        # con_set = self.get_contacts(self.fly_obj)
        
        # if not con_set:
        #     # vel follow
        #     pass
        # else:
        #     impedence_control()
        # # add fly object traj
        return super().step(action)

    def reset(self):

        return super().reset()
        # add fly object traj