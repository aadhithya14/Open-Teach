import numpy as np
import matplotlib.pyplot as plt
import zmq

from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

from copy import deepcopy as copy
from asyncio import threads
from openteach.constants import *
from openteach.utils.timer import FrequencyTimer
from openteach.utils.network import ZMQKeypointSubscriber, ZMQKeypointPublisher
from openteach.utils.vectorops import *
from openteach.utils.files import *

from openteach.robot.robot import RobotWrapper
from scipy.spatial.transform import Rotation, Slerp
from .operator import Operator
from scipy.spatial.transform import Rotation as R
from numpy.linalg import pinv



np.set_printoptions(precision=2, suppress=True)

# Filter for removing noise in the teleoperation
class Filter:
    def __init__(self, state, comp_ratio=0.6):
        self.pos_state = state[:3]
        self.ori_state = state[3:7]
        self.comp_ratio = comp_ratio

    def __call__(self, next_state):
        self.pos_state = self.pos_state[:3] * self.comp_ratio + next_state[:3] * (1 - self.comp_ratio)
        ori_interp = Slerp([0, 1], Rotation.from_rotvec(
            np.stack([self.ori_state, next_state[3:6]], axis=0)),)
        self.ori_state = ori_interp([1 - self.comp_ratio])[0].as_rotvec()
        return np.concatenate([self.pos_state, self.ori_state])

#Template arm operator class
class TemplateArmOperator(Operator):
    def __init__(
        self,
        host, transformed_keypoints_port,
        use_filter=False,
        arm_resolution_port = None, 
        gripper_port =None,
        cartesian_publisher_port = None,
        joint_publisher_port = None,
        cartesian_command_publisher_port = None):

        self.notify_component_start('Bimanual arm operator')
        # Transformed Arm Keypoint Subscriber
        self._transformed_arm_keypoint_subscriber = ZMQKeypointSubscriber(
            host=host,
            port=transformed_keypoints_port,
            topic='transformed_hand_frame'
        )
        # Transformed Hand Keypoint Subscriber
        self._transformed_hand_keypoint_subscriber = ZMQKeypointSubscriber(
            host=host,
            port=transformed_keypoints_port,
            topic='transformed_hand_coords'
        )
        # Gripper Publisher
        self.gripper_publisher = ZMQKeypointPublisher(
            host=host,
            port=gripper_port
        )
        # Cartesian Publisher
        self.cartesian_publisher = ZMQKeypointPublisher(
            host=host,
            port=cartesian_publisher_port
        )
        # Joint Publisher
        self.joint_publisher = ZMQKeypointPublisher(
            host=host,
            port=joint_publisher_port
        )
        # Cartesian Command Publisher
        self.cartesian_command_publisher = ZMQKeypointPublisher(
            host=host,
            port=cartesian_command_publisher_port
        )
        # Arm Resolution Subscriber
        self._arm_resolution_subscriber = ZMQKeypointSubscriber(
            host= host,
            port= arm_resolution_port,
            topic = 'button'
        )
        # Define Robot object
        self._robot = RobotWrapper()
        self.robot.reset()
        
        # Get the initial pose of the robot
        home_pose=np.array(self.robot.get_cartesian_position())
        self.robot_init_H = self.robot_pose_aa_to_affine(home_pose)
        self._timer = FrequencyTimer(BIMANUAL_VR_FREQ) 

        # Use the filter
        self.use_filter = use_filter
        if use_filter:
            robot_init_cart = self._homo2cart(self.robot_init_H)
            self.comp_filter = Filter(robot_init_cart, comp_ratio=0.8)
            
        # Class variables
        self.gripper_flag=1
        self.pause_flag=1
        self.prev_pause_flag=0
        self.is_first_frame= True
        self.gripper_cnt=0
        self.prev_gripper_flag=0
        self.pause_cnt=0
        self.gripper_correct_state=1
        self.resolution_scale =1
        self.arm_teleop_state = ARM_TELEOP_STOP

    @property
    def timer(self):
        return self._timer

    @property
    def robot(self):
        return self._robot

    @property
    def transformed_hand_keypoint_subscriber(self):
        return self._transformed_hand_keypoint_subscriber

    @property
    def transformed_arm_keypoint_subscriber(self):
        return self._transformed_arm_keypoint_subscriber
        
    
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

        return np.block([[rotation, translation[:, np.newaxis]],[0, 0, 0, 1]])
    
    #Function to differentiate between real and simulated robot
    def return_real(self):
        return True

    # Function Gets the transformed hand frame
    def _get_hand_frame(self):
        data = None  # Initialize with a default value
        for i in range(10):
            data = self.transformed_arm_keypoint_subscriber.recv_keypoints(flags=zmq.NOBLOCK)
            if data is not None:
                break 
        if data is None:
            return None
        return np.asanyarray(data).reshape(4, 3)
    
    # Function to get the resolution scale mode
    def _get_resolution_scale_mode(self):
        data = self._arm_resolution_subscriber.recv_keypoints()
        res_scale = np.asanyarray(data).reshape(1)[0] # Make sure this data is one dimensional
        return res_scale
    
    # Function to get the arm teleop state from the hand keypoints
    def _get_arm_teleop_state_from_hand_keypoints(self):
        pause_state ,pause_status,pause_right =self.get_pause_state_from_hand_keypoints()
        pause_status =np.asanyarray(pause_status).reshape(1)[0] 

        return pause_state,pause_status,pause_right

    # Function to turn a frame to a homogeneous matrix
    def _turn_frame_to_homo_mat(self, frame):
        t = frame[0]
        R = frame[1:]

        homo_mat = np.zeros((4, 4))
        homo_mat[:3, :3] = np.transpose(R)
        homo_mat[:3, 3] = t
        homo_mat[3, 3] = 1

        return homo_mat

    # Function to turn homogenous matrix to cartesian vector
    def _homo2cart(self, homo_mat):
        # Here we will use the resolution scale to set the translation resolution
        t = homo_mat[:3, 3]
        R = Rotation.from_matrix(
            homo_mat[:3, :3]).as_rotvec(degrees=False)

        cart = np.concatenate(
            [t, R], axis=0
        )
        return cart
    

    # Get the scaled cartesian pose
    def _get_scaled_cart_pose(self, moving_robot_homo_mat):
        # Get the cart pose without the scaling
        unscaled_cart_pose = self._homo2cart(moving_robot_homo_mat)

        # Get the current cart pose
        home_pose = self.robot.get_cartesian_position()
        home_pose_array = np.array(home_pose)  # Convert tuple to numpy array
        current_homo_mat = self.robot_pose_aa_to_affine(home_pose_array)
        current_cart_pose = home_pose_array#self._homo2cart(current_homo_mat)

        # Get the difference in translation between these two cart poses
        diff_in_translation = unscaled_cart_pose[:3] - current_cart_pose[:3]
        scaled_diff_in_translation = diff_in_translation * self.resolution_scale
        
        scaled_cart_pose = np.zeros(6)
        scaled_cart_pose[3:] = unscaled_cart_pose[3:] # Get the rotation directly
        scaled_cart_pose[:3] = current_cart_pose[:3] + scaled_diff_in_translation # Get the scaled translation only

        return scaled_cart_pose

    # Reset Teleoperation and make the current frame as initial frame
    def _reset_teleop(self):
        print('****** RESETTING TELEOP ****** ')
        home_pose = self.robot.get_cartesian_position()
        home_pose_array = np.array(home_pose)  # Convert tuple to numpy array
        self.robot_init_H = self.robot_pose_aa_to_affine(home_pose_array)
        first_hand_frame = self._get_hand_frame()
        while first_hand_frame is None:
            first_hand_frame = self._get_hand_frame()
        self.hand_init_H = self._turn_frame_to_homo_mat(first_hand_frame)
        self.hand_init_t = copy(self.hand_init_H[:3, 3])
        self.is_first_frame = False
        print("Resetting complete")
        return first_hand_frame

    # Function to get gripper state from hand keypoints
    def get_gripper_state_from_hand_keypoints(self):
        transformed_hand_coords= self._transformed_hand_keypoint_subscriber.recv_keypoints()
        distance = np.linalg.norm(transformed_hand_coords[OCULUS_JOINTS['pinky'][-1]]- transformed_hand_coords[OCULUS_JOINTS['thumb'][-1]])
        thresh = 0.03
        gripper_fl =False
        if distance < thresh:
            self.gripper_cnt+=1
            if self.gripper_cnt==1:
                self.prev_gripper_flag = self.gripper_flag
                self.gripper_flag = not self.gripper_flag 
                gripper_fl=True
        else: 
            self.gripper_cnt=0
        gripper_state = np.asanyarray(self.gripper_flag).reshape(1)[0]
        status= False  
        if gripper_state!= self.prev_gripper_flag:
            status= True
        return gripper_state , status , gripper_fl 
   
    # Toggle the robot to pause/resume using ring/middle finger pinch, both finger modes are supported to avoid any hand pose noise issue
    def get_pause_state_from_hand_keypoints(self):
        transformed_hand_coords= self._transformed_hand_keypoint_subscriber.recv_keypoints()
        ring_distance = np.linalg.norm(transformed_hand_coords[OCULUS_JOINTS['ring'][-1]]- transformed_hand_coords[OCULUS_JOINTS['thumb'][-1]])
        middle_distance = np.linalg.norm(transformed_hand_coords[OCULUS_JOINTS['middle'][-1]]- transformed_hand_coords[OCULUS_JOINTS['thumb'][-1]])
        thresh = 0.03 
        pause_right= True
        if ring_distance < thresh  or middle_distance < thresh:
            self.pause_cnt+=1
            if self.pause_cnt==1:
                self.prev_pause_flag=self.pause_flag
                self.pause_flag = not self.pause_flag       
        else:
            self.pause_cnt=0
        pause_state = np.asanyarray(self.pause_flag).reshape(1)[0]
        pause_status= False  
        if pause_state!= self.prev_pause_flag:
            pause_status= True 
        return pause_state , pause_status , pause_right

    # Function to apply retargeted angles
    def _apply_retargeted_angles(self, log=False):

        # See if there is a reset in the teleop state
        new_arm_teleop_state,pause_status,pause_right = self._get_arm_teleop_state_from_hand_keypoints()
        if self.is_first_frame or (self.arm_teleop_state == ARM_TELEOP_STOP and new_arm_teleop_state == ARM_TELEOP_CONT):
            moving_hand_frame = self._reset_teleop() # Should get the moving hand frame only once
        else:
            moving_hand_frame = self._get_hand_frame()
        self.arm_teleop_state = new_arm_teleop_state
        arm_teleoperation_scale_mode = self._get_resolution_scale_mode()

        if arm_teleoperation_scale_mode == ARM_HIGH_RESOLUTION:
            self.resolution_scale = 1
        elif arm_teleoperation_scale_mode == ARM_LOW_RESOLUTION:
            self.resolution_scale = 0.6


        if moving_hand_frame is None: 
            return # It means we are not on the arm mode yet instead of blocking it is directly returning
        
        self.hand_moving_H = self._turn_frame_to_homo_mat(moving_hand_frame)

        # Transformation code
        H_HI_HH = copy(self.hand_init_H) # Homo matrix that takes P_HI to P_HH - Point in Inital Hand Frame to Point in Home Hand Frame
        H_HT_HH = copy(self.hand_moving_H) # Homo matrix that takes P_HT to P_HH
        H_RI_RH = copy(self.robot_init_H) # Homo matrix that takes P_RI to P_RH
       
        H_HT_HI = np.linalg.pinv(H_HI_HH) @ H_HT_HH # Homo matrix that takes P_HT to P_HI

        # This matrix will change accordingly on how you want the rotations to be.
        H_R_V= np.array([[0 , 0, -1, 0], 
                        [0 , -1, 0, 0],
                        [-1, 0, 0, 0],
                        [0, 0 ,0 , 1]])

        # This matrix will change accordingly on how you want the translation to be.
        H_T_V = np.array([[0, 0 ,1, 0],
                         [0 ,1, 0, 0],
                         [-1, 0, 0, 0],
                        [0, 0, 0, 1]])

        H_HT_HI_r=(pinv(H_R_V) @ H_HT_HI @ H_R_V)[:3,:3] 
        H_HT_HI_t=(pinv(H_T_V) @ H_HT_HI @ H_T_V)[:3,3]
         
        relative_affine = np.block(
        [[ H_HT_HI_r,  H_HT_HI_t.reshape(3, 1)], [0, 0, 0, 1]])
        target_translation = H_RI_RH[:3,3] + relative_affine[:3,3]

        target_rotation = H_RI_RH[:3, :3] @ relative_affine[:3,:3]
        H_RT_RH = np.block(
                    [[target_rotation, target_translation.reshape(-1, 1)], [0, 0, 0, 1]])

        self.robot_moving_H = copy(H_RT_RH)
        final_pose = self._get_scaled_cart_pose(self.robot_moving_H)
        final_pose[0:3]=final_pose[0:3]*1000

        # Apply the filter
        if self.use_filter:
            final_pose = self.comp_filter(final_pose)

        gripper_state,status_change, gripper_flag =self.get_gripper_state_from_hand_keypoints()
        if gripper_flag ==1 and status_change is True:
            self.gripper_correct_state=gripper_state
            self.robot.set_gripper_state(self.gripper_correct_state*800)
        
        # We save the states here during teleoperation as saving directly at 90Hz seems to be too fast for XArm.
        self.gripper_publisher.pub_keypoints(self.gripper_correct_state,"gripper_right")
        position=self.robot.get_cartesian_position()
        joint_position= self.robot.get_joint_position()
        self.cartesian_publisher.pub_keypoints(position,"cartesian")
        self.joint_publisher.pub_keypoints(joint_position,"joint")
        self.cartesian_command_publisher.pub_keypoints(final_pose, "cartesian")
        
        if self.arm_teleop_state == ARM_TELEOP_CONT and gripper_flag == False:
            self.robot.arm_control(final_pose)

       
        


