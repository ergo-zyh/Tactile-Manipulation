from gym.envs.registration import register


register(
    id='franka',
    entry_point='gymEnv.envs.franka:FrankaReacherEnv',
)

