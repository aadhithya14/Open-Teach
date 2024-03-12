import numpy as np
from copy import deepcopy as copy
from .allegro_kdl import AllegroKDL
from openteach.ros_links.allegro_control import DexArmControl 
from openteach.constants import *
from openteach.utils.files import get_yaml_data, get_path_in_package
from openteach.robot.robot import RobotWrapper

class AllegroHand(RobotWrapper):
    def __init__(self, **kwargs):
        self._controller = DexArmControl(robot_type='allegro')

        # For robot configurations
        self._kdl_solver = AllegroKDL()
        self._joint_limit_config = get_yaml_data(get_path_in_package("robot/allegro/configs/allegro_link_info.yaml"))['links_info']

        self._data_frequency = 300

    @property
    def name(self):
        return 'allegro'

    @property
    def recorder_functions(self):
        return {
            'joint_states': self.get_joint_state, 
            'commanded_joint_states': self.get_commanded_joint_state
        }

    @property
    def data_frequency(self):
        return self._data_frequency

    # State information functions
    def get_joint_state(self):
        return self._controller.get_hand_state()

    def get_commanded_joint_state(self):
        return self._controller.get_commanded_hand_state()

    def get_joint_position(self):
        return self._controller.get_hand_position()

    def get_joint_velocity(self):
        return self._controller.get_hand_velocity()

    def get_joint_torque(self):
        return self._controller.get_hand_torque()

    def get_commanded_joint_position(self):
        return self._controller.get_commanded_hand_joint_position()


    # Getting random position initializations for the fingers
    def _get_finger_limits(self, finger_type):
        finger_min = np.array(self._joint_limit_config[finger_type]['joint_min'])
        finger_max = np.array(self._joint_limit_config[finger_type]['joint_max'])
        return finger_min, finger_max

    def _get_thumb_random_angles(self):
        thumb_low_limit, thumb_high_limit = self._get_finger_limits('thumb')

        random_angles = np.zeros((ALLEGRO_JOINTS_PER_FINGER))
        for idx in range(ALLEGRO_JOINTS_PER_FINGER - 1): # ignoring the base
            random_angles[idx + 1] = 0.5 * (thumb_low_limit[idx + 1] + (np.random.rand() * (thumb_high_limit[idx + 1] - thumb_low_limit[idx + 1])))

        return random_angles

    def _get_finger_random_angles(self, finger_type):
        if finger_type == 'thumb':
            return self._get_thumb_random_angles()

        finger_low_limit, finger_high_limit = self._get_finger_limits(finger_type)

        random_angles = np.zeros((ALLEGRO_JOINTS_PER_FINGER))
        for idx in range(ALLEGRO_JOINTS_PER_FINGER - 1): # ignoring the base
            random_angles[idx + 1] = 0.8 * (finger_low_limit[idx + 1] + (np.random.rand() * (finger_high_limit[idx + 1] - finger_low_limit[idx + 1])))

        random_angles[0] = -0.1 + (np.random.rand() * 0.2) # Base angle
        return random_angles

    def set_random_position(self):
        random_angles = []
        for finger_type in ['index', 'middle', 'ring', 'thumb']:
            random_angles.append(self._get_finger_random_angles(finger_type))

        target_angles = np.hstack(random_angles)
        self.move(target_angles)

    # Kinematics functions
    def get_fingertip_coords(self, joint_positions):
        return self._kdl_solver.get_fingertip_coords(joint_positions)

    def _get_joint_state_from_coord(self, index_tip_coord, middle_tip_coord, ring_tip_coord, thumb_tip_coord):
        return self._kdl_solver.get_joint_state_from_coord(
            index_tip_coord, 
            middle_tip_coord, 
            ring_tip_coord, 
            thumb_tip_coord,
            seed = self.get_joint_position()
        )

    # Movement functions
    def home(self):
        self._controller.home_hand()

    def move_coords(self, fingertip_coords):
        desired_angles = self._get_joint_state_from_coord(
            index_tip_coord = fingertip_coords[0:3],
            middle_tip_coord = fingertip_coords[3:6],
            ring_tip_coord = fingertip_coords[6:9],
            thumb_tip_coord = fingertip_coords[9:]
        )

        self._controller.move_hand(desired_angles)

    def move(self, angles):
        self._controller.move_hand(angles)