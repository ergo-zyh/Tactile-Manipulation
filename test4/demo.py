from robosuite import controllers
from robosuite.models.grippers import gripper_model
from model.panda_gripper import PandaGripper

gripper_model = PandaGripper()
print(gripper_model)


# from robosuite.controllers.ik import InverseKinematicsController
# controller = InverseKinematicsController(sim,)
# print(dir(controller))
