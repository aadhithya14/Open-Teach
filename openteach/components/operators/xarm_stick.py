import numpy as np
import time

from openteach.constants import *
from openteach.utils.timer import FrequencyTimer
from openteach.utils.network import ZMQKeypointSubscriber , ZMQKeypointPublisher
from openteach.utils.vectorops import *
from openteach.utils.files import *
from openteach.robot.xarm_stick import XArm
from scipy.spatial.transform import Rotation, Slerp
from .operator import Operator
from scipy.spatial.transform import Rotation as R
from numpy.linalg import pinv
# Scale factor is used to convert from m to mm and mm to m.
from openteach.constants import H_R_V, H_R_V_star, GRIPPER_OPEN, GRIPPER_CLOSE, SCALE_FACTOR


def get_relative_affine(init_affine, current_affine):
    """ Returns the relative affine from the initial affine to the current affine.
        Args:
            init_affine: Initial affine
            current_affine: Current affine
        Returns:
            Relative affine from init_affine to current_affine
    """
    # Relative affine from init_affine to current_affine in the VR controller frame.
    H_V_des = pinv(init_affine) @ current_affine

    # Transform to robot frame.
    # Flips axes
    relative_affine_rot = (pinv(H_R_V) @ H_V_des @ H_R_V)[:3, :3]
    # Translations flips are mirrored.
    relative_affine_trans = (pinv(H_R_V_star) @ H_V_des @ H_R_V_star)[:3, 3]

    # Homogeneous coordinates
    relative_affine = np.block(
        [[relative_affine_rot, relative_affine_trans.reshape(3, 1)], [0, 0, 0, 1]])

    return relative_affine


np.set_printoptions(precision=2, suppress=True)
# Rotation should be filtered when it's being sent
class Filter:
    def __init__(self, state, comp_ratio=0.6):
        self.pos_state = state[:3]
        self.ori_state = state[3:7]
        self.comp_ratio = comp_ratio

    def __call__(self, next_state):
        self.pos_state = self.pos_state[:3] * self.comp_ratio + next_state[:3] * (1 - self.comp_ratio)
        ori_interp = Slerp([0, 1], Rotation.from_rotvec(
            np.stack([self.ori_state, next_state[3:7]], axis=0)),)
        self.ori_state = ori_interp([1 - self.comp_ratio])[0].as_rotvec()
        return np.concatenate([self.pos_state, self.ori_state])


class XArmOperator(Operator):
    def __init__(
        self,
        host, 
        controller_state_port,
        gripper_port=None,
        cartesian_publisher_port = None,
        joint_publisher_port = None,
        cartesian_command_publisher_port = None):

        self.notify_component_start('xArm stick operator')
        
        # Subscribe controller state
        self._controller_state_subscriber = ZMQKeypointSubscriber(
            host=host,
            port=controller_state_port,
            topic='controller_state'
        )

        # # Subscribers for the transformed hand keypoints
        self._transformed_arm_keypoint_subscriber = None
        self._transformed_hand_keypoint_subscriber = None

        # Initalizing the robot controller
        self._robot = XArm(ip=RIGHT_ARM_IP, host_address=host)
        self.robot.reset()

        # Gripper and cartesian publisher
        self.gripper_publisher = ZMQKeypointPublisher(
            host=host,
            port=gripper_port
        )

        self.cartesian_publisher = ZMQKeypointPublisher(
            host=host,
            port=cartesian_publisher_port
        )

        self.joint_publisher = ZMQKeypointPublisher(
            host=host,
            port=joint_publisher_port
        )

        self.cartesian_command_publisher = ZMQKeypointPublisher(
            host=host,
            port=cartesian_command_publisher_port
        )    

        # Get the initial pose of the robot
        home_pose=np.array(self.robot.get_cartesian_position())
        self.robot_init_H = self.robot_pose_aa_to_affine(home_pose)
        self._timer = FrequencyTimer(BIMANUAL_VR_FREQ)

        # Class Variables
        self.resolution_scale =1
        self.arm_teleop_state = ARM_TELEOP_STOP
        self.is_first_frame= True
        self.prev_gripper_flag=0
        self.prev_pause_flag=0
        self.pause_cnt=0
        self.gripper_correct_state=GRIPPER_OPEN
        self.gripper_flag=1
        self.pause_flag=1
        self.gripper_cnt=0

        self.start_teleop = False
        self.init_affine = None

    
    @property
    def timer(self):
        return self._timer

    @property
    def robot(self):
        return self._robot

    def return_real(self):
        return True

    @property
    def transformed_hand_keypoint_subscriber(self):
        return self._transformed_hand_keypoint_subscriber

    @property
    def transformed_arm_keypoint_subscriber(self):
        return self._transformed_arm_keypoint_subscriber
    
    @property
    def controller_state_subscriber(self):
        return self._controller_state_subscriber
        
    # Convert robot pose in axis-angle format to affine matrix
    def robot_pose_aa_to_affine(self, pose_aa: np.ndarray) -> np.ndarray:
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

    def affine_to_robot_pose_aa(self, affine: np.ndarray) -> np.ndarray:
        """Converts an affine matrix to a robot pose in axis-angle format.
        Args:
            affine (np.ndarray): 4x4 affine matrix [[R, t],[0, 1]]
        Returns:
            list: [x, y, z, ax, ay, az] where (x, y, z) is the position and (ax, ay, az) is the axis-angle rotation.
            x, y, z are in mm and ax, ay, az are in radians.
        """
        translation = affine[:3, 3] * SCALE_FACTOR
        rotation = R.from_matrix(affine[:3, :3]).as_rotvec()
        return np.concatenate([translation, rotation])

    # Apply retargeted angles to the robot
    def _apply_retargeted_angles(self, log=False):
       
       # Get the controller state
        self.controller_state = self.controller_state_subscriber.recv_keypoints()
        
        if self.is_first_frame:
            self.robot.home()
            time.sleep(2)
            self.home_pose = self.robot._controller.robot.get_position_aa()[1]
            self.home_affine = self.robot_pose_aa_to_affine(self.home_pose)
            self.is_first_frame = False

        if self.controller_state.right_a:
            # Pressing A button calibrates first frame and starts teleop for right robot.
            self.start_teleop = True
            self.init_affine = self.controller_state.right_affine
        if self.controller_state.right_b:
            # Pressing B button stops teleop. And resets calibration frames to None  for right robot.
            self.start_teleop = False
            self.init_affine = None
            self.home_pose = self.robot._controller.robot.get_position_aa()[1]
            self.home_affine = self.robot_pose_aa_to_affine(self.home_pose)

        
        # Relative transform
        if self.start_teleop:
            relative_affine = get_relative_affine(self.init_affine, self.controller_state.right_affine)
        else:
            relative_affine = np.zeros((4,4))
            relative_affine[3, 3] = 1

        # Gripper
        gripper_state = None
        if self.controller_state.right_index_trigger > 0.5:
            gripper_state = GRIPPER_CLOSE
        elif self.controller_state.right_hand_trigger > 0.5:
            gripper_state = GRIPPER_OPEN
        if gripper_state is not None and gripper_state != self.gripper_correct_state:
            self.robot.set_gripper_state(gripper_state * 800)
            self.gripper_correct_state = gripper_state
            
        if self.start_teleop:
            home_translation = self.home_affine[:3, 3]
            home_rotation = self.home_affine[:3, :3]

            # Target
            target_translation = home_translation + relative_affine[:3, 3]
            target_rotation = home_rotation @ relative_affine[:3, :3]
            
            target_affine = np.block([[target_rotation, target_translation.reshape(-1,1)], [0, 0, 0, 1]])

            # If this target pose is too far from the current pose, move it to the closest point on the boundary.
            target_pose = self.affine_to_robot_pose_aa(target_affine).tolist()
            current_pose = self.robot._controller.robot.get_position_aa()[1]
            delta_translation = np.array(
                    target_pose[:3] - np.array(current_pose[:3]))
            
            # When using servo commands, the maximum distance the robot can move is 10mm; clip translations accordingly.
            delta_translation = np.clip(delta_translation,
                                        a_min=ROBOT_SERVO_MODE_STEP_LIMITS[0],
                                        a_max=ROBOT_SERVO_MODE_STEP_LIMITS[1])

            # a_min and a_max are the boundaries of the robot's workspace; clip absolute position to these boundaries.
            des_translation = delta_translation + np.array(current_pose[:3])
            des_translation = np.clip(des_translation,
                                        a_min=ROBOT_WORKSPACE[0],
                                        a_max=ROBOT_WORKSPACE[1]).tolist()

            des_rotation = target_pose[3:]
            des_pose = des_translation + des_rotation
        else:
            des_pose = self.home_pose

        # We save the states here during teleoperation as saving directly at 90Hz seems to be too fast for XArm.
        self.gripper_publisher.pub_keypoints(self.gripper_correct_state,"gripper")
        position=self.robot.get_cartesian_position()
        joint_position= self.robot.get_joint_position()
        self.cartesian_publisher.pub_keypoints(position,"cartesian")
        self.joint_publisher.pub_keypoints(joint_position,"joint")
        self.cartesian_command_publisher.pub_keypoints(des_pose,"cartesian")

        if self.start_teleop:
            self.robot.arm_control(des_pose)
