import rospy
import numpy as np
import time

from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from allegro_hand.controller import AllegroController
from franka_arm.controller import FrankaController
from copy import deepcopy as copy

from deoxys.utils import transform_utils

from franka_arm.constants import *
from franka_arm.utils import generate_cartesian_space_min_jerk

ALLEGRO_JOINT_STATE_TOPIC = '/allegroHand/joint_states'
ALLEGRO_COMMANDED_JOINT_STATE_TOPIC = '/allegroHand/commanded_joint_states'

FRANKA_HOME = [-1.5208185 ,  1.5375434 ,  1.4714179 , -1.8101345 ,  0.01227421, 1.8809032 ,  0.67484516]



ALLEGRO_ORIGINAL_HOME_VALUES = [
    0, -0.17453293, 0.78539816, 0.78539816,           # Index
    0, -0.17453293,  0.78539816,  0.78539816,         # Middle
    0.08726646, -0.08726646, 0.87266463,  0.78539816, # Ring 
    1.04719755,  0.43633231,  0.26179939, 0.78539816  # Thumb
]
ALLEGRO_HOME_VALUES = ALLEGRO_ORIGINAL_HOME_VALUES


class DexArmControl():
    def __init__(self, record_type=None, robot_type='both'):

        # if pub_port is set to None it will mean that
        # this will only be used for listening to franka and not commanding
        try:
            rospy.init_node("dex_arm", disable_signals = True, anonymous = True)
        except:
            pass
    
        if robot_type == 'both':
            self._init_allegro_hand_control()
            self._init_franka_arm_control(record_type)
        elif robot_type == 'allegro':
            self._init_allegro_hand_control()
        elif robot_type == 'franka':
            self._init_franka_arm_control(record_type)

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

    def _init_franka_arm_control(self, record_type=None):

        self.franka = FrankaController(record_type)

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

    def get_arm_osc_position(self):
        current_pos, current_axis_angle = copy(self.franka.get_osc_position())
        current_pos = np.array(current_pos, dtype=np.float32).flatten()
        current_axis_angle = np.array(current_axis_angle, dtype=np.float32).flatten()

        osc_position = np.concatenate(
            [current_pos, current_axis_angle],
            axis=0
        )
        
        return osc_position

    def get_arm_cartesian_state(self):
        current_pos, current_quat = copy(self.franka.get_cartesian_position())

        cartesian_state = dict(
            position = np.array(current_pos, dtype=np.float32).flatten(),
            orientation = np.array(current_quat, dtype=np.float32).flatten(),
            timestamp = time.time()
        )

        return cartesian_state


    def get_arm_joint_state(self):
        joint_positions = copy(self.franka.get_joint_position())
        # print('joint_position: {}'.format(joint_positions))

        joint_state = dict(
            position = np.array(joint_positions, dtype=np.float32),
            timestamp = time.time()
        )

        return joint_state
    
    def get_arm_pose(self):
        pose = copy(self.franka.get_pose())

        pose_state = dict(
            position = np.array(pose, dtype=np.float32),
            timestamp = time.time()
        )

        return pose_state

    def get_arm_position(self):

        joint_state = self.get_arm_joint_state()
        return joint_state['position']


    def get_arm_velocity(self):
        raise ValueError('get_arm_velocity() is being called - Arm Velocity cannot be collected in Franka arms, this method should not be called')

    def get_arm_torque(self):
        raise ValueError('get_arm_torque() is being called - Arm Torques cannot be collected in Franka arms, this method should not be called')


    def get_arm_cartesian_coords(self):
        current_pos, current_quat = copy(self.franka.get_cartesian_position())

        current_pos = np.array(current_pos, dtype=np.float32).flatten()
        current_quat = np.array(current_quat, dtype=np.float32).flatten()

        cartesian_coord = np.concatenate(
            [current_pos, current_quat],
            axis=0
        )

        return cartesian_coord

    # Movement functions
    def move_hand(self, allegro_angles):
        self.allegro.hand_pose(allegro_angles)

    def home_hand(self):
        self.allegro.hand_pose(ALLEGRO_HOME_VALUES)

    def reset_hand(self):
        self.home_hand()

    def move_arm_joint(self, joint_angles):
        self.franka.joint_movement(joint_angles)

    def move_arm_cartesian(self, cartesian_pos, duration=3):
        # Moving
        start_pose = self.get_arm_cartesian_coords()
        poses = generate_cartesian_space_min_jerk(
            start = start_pose, 
            goal = cartesian_pos, 
            time_to_go = duration,
            hz = self.franka.control_freq
        )

        for pose in poses:
            self.arm_control(pose)

        # Debugging the pose difference 
        last_pose = self.get_arm_cartesian_coords()
        pose_error = cartesian_pos - last_pose
        debug_quat_diff = transform_utils.quat_multiply(last_pose[3:], transform_utils.quat_inverse(cartesian_pos[3:]))
        angle_diff = 180*np.linalg.norm(transform_utils.quat2axisangle(debug_quat_diff))/np.pi
        print('Absolute Pose Error: {}, Angle Difference: {}'.format(
            np.abs(pose_error[:3]), angle_diff
        ))

    def arm_control(self, cartesian_pose):
        self.franka.cartesian_control(cartesian_pose=cartesian_pose)

    def home_arm(self):
        self.move_arm_cartesian(FRANKA_HOME_CART, duration=5)

    def reset_arm(self):
        self.home_arm()

    # Full robot commands
    def move_robot(self, allegro_angles, arm_angles):
        self.franka.joint_movement(arm_angles, False)
        self.allegro.hand_pose(allegro_angles)

    def home_robot(self):
        self.home_hand()
        self.home_arm() # For now we're using cartesian values
