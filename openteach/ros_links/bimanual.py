#import rospy
import numpy as np
import time
from copy import deepcopy as copy
from xarm import XArmAPI
from enum import Enum
import math

from openteach.constants import SCALE_FACTOR
from scipy.spatial.transform import Rotation as R
from openteach.constants import *

class RobotControlMode(Enum):
    CARTESIAN_CONTROL = 0
    SERVO_CONTROL = 1

#Wrapper for XArm
class Robot(XArmAPI):
    def __init__(self, ip="192.168.86.230", is_radian=True):
        super(Robot, self).__init__(
            port=ip, is_radian=is_radian, is_tool_coord=False)
        self.set_gripper_enable(True)
        self.ip = ip

    def clear(self):
        self.clean_error()
        self.clean_warn()
        # self.motion_enable(enable=False)
        self.motion_enable(enable=True)

    def set_mode_and_state(self, mode: RobotControlMode, state: int = 0):
        self.set_mode(mode.value)
        self.set_state(state)
        self.set_gripper_mode(0)  # Gripper is always in position control.

    def reset(self):
        # Clean error
        self.clear()
        print("SLow reset working")
        self.set_mode_and_state(RobotControlMode.CARTESIAN_CONTROL, 0)
        status = self.set_servo_angle(angle=ROBOT_HOME_JS, wait=True, is_radian=True, speed=math.radians(50))
        # self.set_mode_and_state(RobotControlMode.SERVO_CONTROL, 0)
        # status = self.set_servo_cartesian_aa(
        #             ROBOT_HOME_POSE_AA, wait=False, relative=False, mvacc=200, speed=50)
        assert status == 0, "Failed to set robot at home joint position"
        self.set_mode_and_state(RobotControlMode.SERVO_CONTROL, 0)
        self.set_gripper_position(800.0, wait=True)
        time.sleep(0.1)



class DexArmControl():
    def __init__(self,ip,  record_type=None):

        # if pub_port is set to None it will mean that
        # this will only be used for listening to franka and not commanding
        # try:
        #     rospy.init_node("dex_arm", disable_signals = True, anonymous = True)
        # except:
        #     pass
    
       
        #self._init_franka_arm_control(record)
        self.robot =Robot(ip, is_radian=True) 

        self.desired_cartesian_pose = None

    # Controller initializers
    def _init_xarm_control(self):
       
        self.robot.reset()
        
        status, home_pose = self.robot.get_position_aa()
        assert status == 0, "Failed to get robot position"
        home_affine = self.robot_pose_aa_to_affine(home_pose)
        # Initialize timestamp; used to send messages to the robot at a fixed frequency.
        last_sent_msg_ts = time.time()

        # Initialize the environment state action tuple.
        
    

   

    # Rostopic callback functions
   
    # State information functions
   
    def get_arm_pose(self):
        status, home_pose = self.robot.get_position_aa()
        home_affine = self.robot_pose_aa_to_affine(home_pose)
        return home_affine

    def get_arm_position(self):
        joint_state =np.array(self.robot.get_servo_angle()[1])
        return joint_state

    def get_arm_velocity(self):
        raise ValueError('get_arm_velocity() is being called - Arm Velocity cannot be collected in Franka arms, this method should not be called')

    def get_arm_torque(self):
        raise ValueError('get_arm_torque() is being called - Arm Torques cannot be collected in Franka arms, this method should not be called')

    def get_arm_cartesian_coords(self):
        status, home_pose = self.robot.get_position_aa()
        return home_pose

    def get_gripper_state(self):
        gripper_position=self.robot.get_gripper_position()
        gripper_pose= dict(
            position = np.array(gripper_position[1], dtype=np.float32).flatten(),
            timestamp = time.time()
        )
        return gripper_pose

    def move_arm_joint(self, joint_angles):
        self.robot.set_servo_angle(joint_angles, wait=True, is_radian=True, mvacc=80, speed=10)

    def move_arm_cartesian(self, cartesian_pos, duration=3):
        self.robot.set_servo_cartesian_aa(
                    cartesian_pos, wait=False, relative=False, mvacc=200, speed=50)

    def set_desired_cartesian_pose(self, cartesian_pose):
        self.desired_cartesian_pose = cartesian_pose

    def arm_control(self, cartesian_pose):
        if self.robot.has_error:
            self.robot.clear()
            # self.robot.set_mode_and_state(1)
            self.robot.set_mode_and_state(RobotControlMode.SERVO_CONTROL, 0)
        self.robot.set_servo_cartesian_aa(
                    cartesian_pose, wait=False, relative=False, mvacc=200, speed=50)
    
    def continue_control(self):
        if self.desired_cartesian_pose is None:
            return
        
        curr_cartesian_pose = self.get_arm_cartesian_coords()

        pos = curr_cartesian_pose[:3]
        delta = self.desired_cartesian_pose[:3] - pos
        delta = np.clip(delta, -2, 2)
        pos = curr_cartesian_pose[:3] + delta
        next_cartesian_pose = np.concatenate([pos, self.desired_cartesian_pose[3:]])
        self.arm_control(next_cartesian_pose)

        # self.robot.continue_move()
        
    def get_arm_joint_state(self):
        joint_positions =np.array(self.robot.get_servo_angle()[1])
        joint_state = dict(
            position = np.array(joint_positions, dtype=np.float32),
            timestamp = time.time()
        )
        return joint_state
        
    def get_cartesian_state(self):
        status,current_pos=self.robot.get_position_aa() 
        cartesian_state = dict(
            position = np.array(current_pos[0:3], dtype=np.float32).flatten(),
            orientation = np.array(current_pos[3:], dtype=np.float32).flatten(),
            timestamp = time.time()
        )

        return cartesian_state

    def home_arm(self):
        self.move_arm_cartesian(BIMANUAL_RIGHT_HOME, duration=5)

    def reset_arm(self):
        self.home_arm()

    # Full robot commands
    def move_robot(self,arm_angles):
        self.robot.set_servo_angle(angle=arm_angles,is_radian=True)
        
    def home_robot(self):
        self.home_arm() # For now we're using cartesian values

    def set_gripper_status(self, position):
        self.robot.set_gripper_position(position)

    def robot_pose_aa_to_affine(self,pose_aa: np.ndarray) -> np.ndarray:
        """Converts a robot pose in axis-angle format to an affine matrix.
        Args:
            pose_aa (list): [x, y, z, ax, ay, az] where (x, y, z) is the position and (ax, ay, az) is the axis-angle rotation.
            x, y, z are in mm and ax, ay, az are in radians.
        Returns:
            np.ndarray: 4x4 affine matrix [[R, t],[0, 1]]
        """

        rotation = R.from_rotvec(pose_aa[3:]).as_matrix()
        translation = np.array(pose_aa[:3]) / SCALE_FACTOR

        return np.block([[rotation, translation[:, np.newaxis]],
                        [0, 0, 0, 1]])