from openteach.ros_links.bimanual import DexArmControl
from .robot import RobotWrapper
from openteach.utils.network import ZMQKeypointSubscriber
import numpy as np
import time

class BimanualLeft(RobotWrapper):
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
        return 'left_xarm'

    @property
    def data_frequency(self):
        return self._data_frequency

    # State information functions
    def get_joint_state(self):
        return self._controller.get_arm_joint_state()
    
    def get_joint_velocity(self):
        pass

    def get_joint_torque(self):
        pass

    def get_cartesian_state(self):
        return self._controller.get_cartesian_state()

    def get_joint_position(self):
        return self._controller.get_arm_position()
    
    def get_cartesian_position(self):
        return self._controller.get_arm_cartesian_coords()

    def reset(self):
        return self._controller._init_xarm_control()
    
    def get_pose(self):
        return self._controller.get_arm_pose()

    # Movement functions
    def home(self):
        return self._controller.home_arm()

    def move(self, input_angles):
        self._controller.move_arm_joint(input_angles)

    def move_coords(self, cartesian_coords, duration=3):
        self._controller.move_arm_cartesian(cartesian_coords, duration=duration)

    def arm_control(self, cartesian_coords):
        self._controller.arm_control(cartesian_coords)

    def move_velocity(self, input_velocity_values, duration):
        pass


    def get_gripper_state_from_socket(self):
        self._gripper_state_subscriber = ZMQKeypointSubscriber(
                host = '10.19.216.156', 
                port = 8115,
                topic = 'gripper_left'
            )
        gripper_state = self._gripper_state_subscriber.recv_keypoints()
        gripper_state_dict= dict(
            gripper_position = np.array(gripper_state, dtype=np.float32),
            timestamp = time.time()
        )
        return gripper_state_dict
    

    def get_cartesian_state_from_socket(self):
        self.cartesian_state_subscriber = ZMQKeypointSubscriber(
                host = '10.19.216.156', 
                port = 8116,
                topic = 'cartesian'
            )
        cartesian_state = self.cartesian_state_subscriber.recv_keypoints()
        cartesian_state_dict= dict(
            cartesian_position = np.array(cartesian_state, dtype=np.float32),
            timestamp = time.time()
        )
        return cartesian_state_dict
    
    def get_joint_state_from_socket(self):
        self._joint_state_subscriber = ZMQKeypointSubscriber(
                host = '10.19.216.156', 
                port = 8117,
                topic = 'joint'
            )
        joint_state = self._joint_state_subscriber.recv_keypoints()
        # gripper_state = self._controller.robot.get_gripper_position()[1]
        joint_state_dict= dict(
            joint_position = np.array(joint_state, dtype=np.float32),
            timestamp = time.time()
        )
        #self._controller.set_gripper_status(gripper_state)
        return joint_state_dict
    
    def get_cartesian_commanded_position(self):
        self.cartesian_state_subscriber = ZMQKeypointSubscriber(
                host = '10.19.216.156', 
                port = 8121,
                topic = 'cartesian'
            )
        cartesian_state = self.cartesian_state_subscriber.recv_keypoints()
        # gripper_state = self._controller.robot.get_gripper_position()[1]
        cartesian_state_dict= dict(
            commanded_cartesian_position = np.array(cartesian_state, dtype=np.float32),
            timestamp = time.time()
        )
        #self._controller.set_gripper_status(gripper_state)
        return cartesian_state_dict
    
    def get_robot_actual_cartesian_position(self):
        cartesian_state=self.get_cartesian_position()
        cartesian_dict = dict(
            cartesian_position = np.array(cartesian_state, dtype=np.float32),
            timestamp = time.time()
        )
        
        return cartesian_dict
    
   
    def get_robot_actual_joint_position(self):
        joint_state_dict=self._controller.get_arm_joint_state()
        return joint_state_dict
    
    def get_gripper_state(self):
        gripper_state_dict= self._controller.get_gripper_state()
        return gripper_state_dict

    

