from openteach.ros_links.stretch import DexArmControl 
from .robot import RobotWrapper
from openteach.utils.network import ZMQKeypointSubscriber
import numpy as np
import time

class Stretch(RobotWrapper):
    def __init__(self, ip,record_type=None):
        self._controller = DexArmControl(ip=ip,record_type=record_type)
        self._data_frequency = 90

    @property
    def recorder_functions(self):
        return {
            'joint_states': self.get_joint_state_from_socket,
            'cartesian_states': self.get_cartesian_state_from_socket,
            'gripper_states': self.get_gripper_state_from_socket,
            'actual_cartesian_states': self.get_robot_actual_cartesian_position,
            'actual_joint_states': self.get_robot_actual_joint_position,
            'actual_gripper_states': self.get_gripper_state,
            'commanded_cartesian_state': self.get_cartesian_commanded_position
        }

    @property
    def name(self):
        return 'stretch'

    @property
    def data_frequency(self):
        return self._data_frequency

    # State information functions
    def get_joint_velocity(self):
        pass

    def get_joint_torque(self):
        pass

    def get_arm_cartesian_state(self):
        return self._controller.get_arm_cartesian_state()

    def get_base_state(self):
        return self._controller.get_base_state()
    
    def get_lift_state(self):
        return self._controller.get_lift_state()

    def get_end_of_the_arm_state(self):
        return self._controller.get_end_of_the_arm_state()
    
    def reset(self):
        return self._controller._init_control()

    # Movement functions
    def home(self):
        return self._controller.home_arm()

    def move_coords(self, cartesian_coords, duration=3):
        self._controller.move_arm_cartesian(cartesian_coords, duration=duration)

    def arm_control(self, cartesian_coords):
        self._controller.arm_control(cartesian_coords)

    def get_pose(self):
        # self.robot_init_rotation = self.get_end_of_the_arm_state()
        # self.robot_init_base = self.get_base_state()
        # self.robot_init_lift = self.get_lift_state()
        # self.robot_init_arm = self.get_arm_cartesian_state()
        # self.robot_init_translation = np.array([self.robot_init_base,self.robot_init_lift,self.robot_init_arm])
        # self.robot_init_H = np.block([[self.robot_init_rotation, self.robot_init_translation], [0, 0, 0, 1]])
        pass 

    def base_control(self, base_coords):
        self._controller.base_control(base_coords)

    def lift_control(self, lift_coords):
        self._controller.lift_control(lift_coords)

    def move_velocity(self, input_velocity_values, duration): 
        pass

    def set_gripper_state(self , gripper_state):
        self._controller.set_gripper_status(gripper_state)
    
    def get_gripper_state(self):
        gripper_state_dict= self._controller.get_gripper_state()
        return gripper_state_dict