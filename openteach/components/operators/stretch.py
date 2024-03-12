import numpy as np
import matplotlib.pyplot as plt
import zmq

from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

from copy import deepcopy as copy
from openteach.constants import *
from openteach.utils.timer import FrequencyTimer
from openteach.utils.network import ZMQKeypointSubscriber
from openteach.utils.vectorops import *
from openteach.utils.files import *
#from openteach.robot.stretch import Stretch
from scipy.spatial.transform import Rotation, Slerp
from .operator import Operator
#from stretch_visual_servoing.normalized_velocity_control import NormalizedVelocityControl
from openteach.utils.publisher import ImitiationPolicyPublisher
from openteach.utils.subscriber import ImagePolicySubscriber
import torch


np.set_printoptions(precision=2, suppress=True)
# Filter to smooth out the arm cartesian state
class Filter:
    def __init__(self, state, comp_ratio=0.6):
        self.pos_state = state[:3]
        self.ori_state = state[3:7]
        self.comp_ratio = comp_ratio

    def __call__(self, next_state):
        self.pos_state = self.pos_state[:3] * self.comp_ratio + next_state[:3] * (1 - self.comp_ratio)
        ori_interp = Slerp([0, 1], Rotation.from_quat(
            np.stack([self.ori_state, next_state[3:7]], axis=0)),)
        self.ori_state = ori_interp([1 - self.comp_ratio])[0].as_quat()
        return np.concatenate([self.pos_state, self.ori_state])

class StretchOperator(Operator):
    def __init__(
        self,
        host,
        transformed_keypoints_port,
        use_filter=False,
        arm_resolution_port = None,
        teleoperation_reset_port = None,
    ):
        self.notify_component_start('stretch arm operator')
        # Subscribers for the transformed hand keypoints
        self._transformed_hand_keypoint_subscriber = ZMQKeypointSubscriber(
            host=host,
            port=transformed_keypoints_port,
            topic='transformed_hand_coords'
        )
        # Subscribers for the transformed arm frame
        self._transformed_arm_keypoint_subscriber = ZMQKeypointSubscriber(
            host=host,
            port=transformed_keypoints_port,
            topic='transformed_hand_frame'
        )

        # Initalizing the robot controller
        #self._robot = Stretch()
        self.resolution_scale = 1 # NOTE: Get this from a socket
        self.arm_teleop_state = ARM_TELEOP_STOP # We will start as the cont

        # Subscribers for the resolution scale and teleop state
        self._arm_resolution_subscriber = ZMQKeypointSubscriber(
            host = host,
            port = arm_resolution_port,
            topic = 'button'
        )

        self._arm_teleop_state_subscriber = ZMQKeypointSubscriber(
            host = host, 
            port = teleoperation_reset_port,
            topic = 'pause'
        )
        self.is_first_frame = True
        self.publisher = ImitiationPolicyPublisher()
        self.subscriber = ImagePolicySubscriber()
        self.subscriber.register_for_uid(self.publisher)
        self.subscriber.register_for_uid(self)

        self.use_filter = use_filter

        self._timer = FrequencyTimer(5)
        self.gripper_cnt=0
        self.prev_gripper_flag=False
        self.gripper_flag=False
        self.pause_cnt=0
        self.gripper_correct_state=1

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

    # Get the hand frame
    def _get_hand_frame(self):
        for i in range(10):
            data = self.transformed_arm_keypoint_subscriber.recv_keypoints(flags=zmq.NOBLOCK)
            if not data is None: break 
        if data is None: return None
        return np.asanyarray(data).reshape(4, 3)
    
    # Get the resolution scale mode (High or Low)
    def _get_resolution_scale_mode(self):
        data = self._arm_resolution_subscriber.recv_keypoints()
        res_scale = np.asanyarray(data).reshape(1)[0] # Make sure this data is one dimensional
        return res_scale  

    # Get the teleop state (Pause or Continue)
    def _get_arm_teleop_state(self):
        reset_stat = self._arm_teleop_state_subscriber.recv_keypoints()
        reset_stat = np.asanyarray(reset_stat).reshape(1)[0] # Make sure this data is one dimensional
        return reset_stat

    # Converts a frame to a homogenous transformation matrix
    def _turn_frame_to_homo_mat(self, frame):
        t = frame[0]
        R = frame[1:]

        homo_mat = np.zeros((4, 4))
        homo_mat[:3, :3] = np.transpose(R)
        homo_mat[:3, 3] = t
        homo_mat[3, 3] = 1

        return homo_mat
    
    # Converts Homogenous Transformation Matrix to Cartesian Coords
    def _homo2cart(self, homo_mat):
        t = homo_mat[:3, 3]
        R = Rotation.from_matrix(
            homo_mat[:3, :3]).as_euler('zxy', degrees=False)

        cart = np.concatenate(
            [t, R]
        )

        return cart
    
    # Gets the Scaled Resolution pose
    def _get_scaled_cart_pose(self, moving_robot_homo_mat):
        # Get the cart pose without the scaling
        unscaled_cart_pose = self._homo2cart(moving_robot_homo_mat)

        # Get the current cart pose
        current_homo_mat = copy(self.robot.get_pose())
        current_cart_pose = self._homo2cart(current_homo_mat)

        # Get the difference in translation between these two cart poses
        diff_in_translation = unscaled_cart_pose[:3] - current_cart_pose[:3]
        scaled_diff_in_translation = diff_in_translation * self.resolution_scale
        # print('SCALED_DIFF_IN_TRANSLATION: {}'.format(scaled_diff_in_translation))
        
        scaled_cart_pose = np.zeros(7)
        scaled_cart_pose[3:] = unscaled_cart_pose[3:] # Get the rotation directly
        scaled_cart_pose[:3] = current_cart_pose[:3] + scaled_diff_in_translation # Get the scaled translation only

        return scaled_cart_pose

    # Reset the teleoperation and get the first frame
    def _reset_teleop(self):
        # Get the rotation components
        first_hand_frame = self._get_hand_frame()
        while first_hand_frame is None:
            first_hand_frame = self._get_hand_frame()
        self.hand_init_H = self._turn_frame_to_homo_mat(first_hand_frame)
        self.hand_init_t = copy(self.hand_init_H[:3, 3])
        self.is_first_frame = False
        return first_hand_frame
    
    # Function to get gripper state from hand keypoints
    def get_gripper_state_from_hand_keypoints(self):
        transformed_hand_coords= self._transformed_hand_keypoint_subscriber.recv_keypoints()
        distance = np.linalg.norm(transformed_hand_coords[OCULUS_JOINTS['index'][-1]]- transformed_hand_coords[OCULUS_JOINTS['thumb'][-1]])
        thresh = 0.03
        gripper_fl =False
        if distance < thresh: # and distance2 < thresh and distance3 < thresh) or (distance < thresh and distance2 < thresh) or (distance < thresh and distance3 < thresh) or (distance2 < thresh and distance3 < thresh):
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

    # Apply the retargeted angles
    def _apply_retargeted_angles(self, log=False):

        # See if there is a reset in the teleop
        new_arm_teleop_state = self._get_arm_teleop_state()
        if self.is_first_frame or (self.arm_teleop_state == ARM_TELEOP_STOP and new_arm_teleop_state == ARM_TELEOP_CONT):
            moving_hand_frame = self._reset_teleop() # Should get the moving hand frame only once
        else:
            moving_hand_frame = self._get_hand_frame() # Should get the hand frame 
        self.arm_teleop_state = new_arm_teleop_state 

        # Get the arm resolution
        arm_teleoperation_scale_mode = self._get_resolution_scale_mode()
        if arm_teleoperation_scale_mode == ARM_HIGH_RESOLUTION:
            self.resolution_scale = 1
        elif arm_teleoperation_scale_mode == ARM_LOW_RESOLUTION:
            self.resolution_scale = 0.6

        if moving_hand_frame is None: 
            return # It means we are not on the arm mode yet instead of blocking it is directly returning
        
        # Get the moving hand frame
        self.hand_moving_H = self._turn_frame_to_homo_mat(moving_hand_frame)

        # Transformation code
        H_HI_HH = copy(self.hand_init_H) # Homo matrix that takes P_HI  to P_HH - Point in Inital Hand Frame to Point in current hand Frame
        H_HT_HH = copy(self.hand_moving_H) # Homo matrix that takes P_HT to P_HH
        #H_RI_RH = copy(self.robot_init_H) # Homo matrix that takes P_RI to P_RH

      
        H_HT_HI = np.linalg.pinv(H_HI_HH) @ H_HT_HH # Homo matrix that takes P_HT to P_HI
      
        H_T_V= [[1,0,0,0],
                [0,1,0,0],
                [0,0,1,0],
                [0,0,0,1]]

        H_R_V= [[-1,0,0,0],
                [0,-1,0,0],
                [0,0,-1,0],
                [0,0,0,1]]
        
        mask =  [[0,0,0],
                [0,0,0],
                [0,0,1]]
        
        # # Find the relative transform and apply it to robot initial position

        H_R_R= (np.linalg.pinv(H_R_V)@H_HT_HI@H_R_V)[:3,:3]
        H_R_T= (np.linalg.pinv(H_T_V)@H_HT_HI@H_T_V)[:3,3] 
        H_RT_RH=np.block([[H_R_R,H_R_T.reshape(3,1)],[np.array([0,0,0]),1]])
        print("Translation is", H_R_T)
        self.robot_moving_H = copy(H_RT_RH)
        final_pose = self.robot_moving_H
        # Use the resolution scale to get the final cart pose
        # Use a Filter
        if self.use_filter:
            final_pose = self.comp_filter(final_pose)
        cart= self._homo2cart(final_pose)
        cart[0]=cart[0]*0.1
        # Move the robot arm
        gripper_state,status_change, gripper_flag=self.get_gripper_state_from_hand_keypoints()
        if gripper_flag ==1 and status_change is True:
            if gripper_state==0:
                self.gripper_correct_state=-1
            else:
                self.gripper_correct_state=1
        
        print("Gripper state",self.gripper_correct_state)
        print("Cartesian state",cart[3:6])
        self.publisher.publish_action(cart, torch.tensor(np.array([self.gripper_correct_state])))

    def stream(self):
        #self.notify_component_start('{} control'.format(self.robot.name))
        print("Start controlling the robot hand using the Oculus Headset.\n")

        # Assume that the initial position is considered initial after 3 seconds of the start
        while True:
            try:
                #if self.robot.get_joint_position() is not None:
                self.timer.start_loop()

                    # Retargeting function
                self._apply_retargeted_angles(log=False)

                self.timer.end_loop()
            except KeyboardInterrupt:
                break

        self.transformed_arm_keypoint_subscriber.stop()
        print('Stopping the teleoperator!')
