from openteach.ros_links.franka_allegro_control import DexArmControl 
from .robot import RobotWrapper

class FrankaArm(RobotWrapper):
    def __init__(self, record_type=None):
        self._controller = DexArmControl(record_type=record_type, robot_type='franka')
        self._data_frequency = 50

    @property
    def recorder_functions(self):
        return {
            'joint_states': self.get_joint_state,
            'cartesian_states': self.get_cartesian_state
        }

    @property
    def name(self):
        return 'franka'

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
        return self._controller.get_arm_cartesian_state()

    def get_joint_position(self):
        return self._controller.get_arm_position()
    
    def get_cartesian_position(self):
        return self._controller.get_arm_cartesian_coords()

    def get_osc_position(self):
        return self._controller.get_arm_osc_position()
    
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