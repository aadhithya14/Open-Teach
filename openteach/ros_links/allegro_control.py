import rospy
import numpy as np
import time

from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from allegro_hand.controller import AllegroController
from copy import deepcopy as copy


ALLEGRO_JOINT_STATE_TOPIC = '/allegroHand/joint_states'
ALLEGRO_COMMANDED_JOINT_STATE_TOPIC = '/allegroHand/commanded_joint_states'



ALLEGRO_ORIGINAL_HOME_VALUES = [
    0, -0.17453293, 0.78539816, 0.78539816,           # Index
    0, -0.17453293,  0.78539816,  0.78539816,         # Middle
    0.08726646, -0.08726646, 0.87266463,  0.78539816, # Ring 
    1.04719755,  0.43633231,  0.26179939, 0.78539816  # Thumb
]



class DexArmControl():
    def __init__(self, record_type=None, robot_type='both'):

        # if pub_port is set to None it will mean that
        # this will only be used for listening to franka and not commanding
        try:
            rospy.init_node("dex_arm", disable_signals = True, anonymous = True)
        except:
            pass
    
        if robot_type == 'allegro':
            self._init_allegro_hand_control()
    # Controller initializers
    def _init_allegro_hand_control(self):
        self.allegro = AllegroController()

        self.allegro_joint_state = None
        rospy.Subscriber(
            ALLEGRO_JOINT_STATE_TOPIC, 
            JointState, 
            self._callback_allegro_joint_state, 
            queue_size = 1
        )

        self.allegro_commanded_joint_state = None
        rospy.Subscriber(
            ALLEGRO_COMMANDED_JOINT_STATE_TOPIC, 
            JointState, 
            self._callback_allegro_commanded_joint_state, 
            queue_size = 1
        )

   

    # Rostopic callback functions
    def _callback_allegro_joint_state(self, joint_state):
        self.allegro_joint_state = joint_state

    def _callback_allegro_commanded_joint_state(self, joint_state):
        self.allegro_commanded_joint_state = joint_state

    # State information functions
    def get_hand_state(self):
        if self.allegro_joint_state is None:
            return None

        raw_joint_state = copy(self.allegro_joint_state)

        joint_state = dict(
            position = np.array(raw_joint_state.position, dtype = np.float32),
            velocity = np.array(raw_joint_state.velocity, dtype = np.float32),
            effort = np.array(raw_joint_state.effort, dtype = np.float32),
            timestamp = raw_joint_state.header.stamp.secs + (raw_joint_state.header.stamp.nsecs * 1e-9)
        )
        return joint_state

    def get_commanded_hand_state(self):
        if self.allegro_commanded_joint_state is None:
            return None

        raw_joint_state = copy(self.allegro_commanded_joint_state)

        joint_state = dict(
            position = np.array(raw_joint_state.position, dtype = np.float32),
            velocity = np.array(raw_joint_state.velocity, dtype = np.float32),
            effort = np.array(raw_joint_state.effort, dtype = np.float32),
            timestamp = raw_joint_state.header.stamp.secs + (raw_joint_state.header.stamp.nsecs * 1e-9)
        )
        return joint_state
        
    def get_hand_position(self):
        if self.allegro_joint_state is None:
            return None

        return np.array(self.allegro_joint_state.position, dtype = np.float32)

    def get_hand_velocity(self):
        if self.allegro_joint_state is None:
            return None

        return np.array(self.allegro_joint_state.velocity, dtype = np.float32)

    def get_hand_torque(self):
        if self.allegro_joint_state is None:
            return None

        return np.array(self.allegro_joint_state.effort, dtype = np.float32)

    def get_commanded_hand_joint_position(self):
        if self.allegro_commanded_joint_state is None:
            return None

        return np.array(self.allegro_commanded_joint_state.position, dtype = np.float32)

    
    # Movement functions
    def move_hand(self, allegro_angles):
        self.allegro.hand_pose(allegro_angles)

    def home_hand(self):
        self.allegro.hand_pose(ALLEGRO_ORIGINAL_HOME_VALUES)

    def reset_hand(self):
        self.home_hand()

    
    # Full robot commands
    def move_robot(self, allegro_angles, arm_angles):
        self.allegro.hand_pose(allegro_angles)

    def home_robot(self):
        self.home_hand()
        # For now we're using cartesian values
