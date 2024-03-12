import numpy as np
import matplotlib.pyplot as plt
import zmq

from tqdm import tqdm

from copy import deepcopy as copy
from openteach.constants import *
from openteach.utils.timer import FrequencyTimer
from openteach.utils.network import ZMQKeypointSubscriber, ZMQKeypointPublisher
from openteach.utils.vectorops import *
from openteach.utils.files import *
from scipy.spatial.transform import Rotation, Slerp
from .operator import Operator
from .calibrators.allegro import OculusThumbBoundCalibrator

import robosuite.utils.transform_utils as T
from scipy.spatial.transform import Rotation as R

np.set_printoptions(precision=2, suppress=True)


# Rotation should be filtered when it's being sent
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
	
class LiberoSimOperator(Operator):
	def __init__(
		self,
		host,
		transformed_keypoints_port,
		stream_configs,
		stream_oculus,
		endeff_publish_port,
		endeffpossubscribeport,
		robotposesubscribeport,
		moving_average_limit,
		arm_resolution_port = None,
	):
		self.notify_component_start('libero operator')
		self._host, self._port = host, transformed_keypoints_port
		self._hand_transformed_keypoint_subscriber = ZMQKeypointSubscriber(
			host = self._host,
			port = self._port,
			topic = 'transformed_hand_coords'
		)
		self._arm_transformed_keypoint_subscriber = ZMQKeypointSubscriber(
			host=host,
			port=transformed_keypoints_port,
			topic='transformed_hand_frame'
		)


		# Initalizing the robot controller
		self.resolution_scale = 1 # NOTE: Get this from a socket
		self.arm_teleop_state = ARM_TELEOP_STOP # We will start as the cont
		self.gripper_correct_state =0
		self.pause_flag=0
		self.gripper_flag=1
		self.prev_gripper_flag=0
		self.prev_pause_flag=0
		self.pause_cnt=0


		self._arm_resolution_subscriber = ZMQKeypointSubscriber(
			host = host,
			port = arm_resolution_port,
			topic = 'button'
		)

		self.end_eff_position_subscriber = ZMQKeypointSubscriber(
			host = host,
			port =  endeffpossubscribeport,
			topic = 'endeff_coords'

		)

		self.end_eff_position_publisher = ZMQKeypointPublisher(
			host = host,
			port = endeff_publish_port
		)

		# robot pose subscriber
		self.robot_pose_subscriber = ZMQKeypointSubscriber(
			host = host,
			port = robotposesubscribeport,
			topic = 'robot_pose'
		)

		# Calibrating to get the thumb bounds
		self._calibrate_bounds()

		self._stream_oculus=stream_oculus
		self.stream_configs=stream_configs
		self._timer = FrequencyTimer(VR_FREQ)
		self._robot='Libero_Sim'
		self.is_first_frame = True
		
		# Frequency timer
		self._timer = FrequencyTimer(VR_FREQ)
		self.direction_counter = 0
		self.current_direction = 0
		# Moving average queues
		self.moving_Average_queue = []
		self.moving_average_limit = moving_average_limit
		self.hand_frames = []
		self.count = 0

	@property
	def timer(self):
		return self._timer

	@property
	def robot(self):
		return self._robot
	
	@property
	def transformed_hand_keypoint_subscriber(self):
		return self._hand_transformed_keypoint_subscriber
	
	@property
	def transformed_arm_keypoint_subscriber(self):
		return self._arm_transformed_keypoint_subscriber
	
	#Calibrate the bounds
	def _calibrate_bounds(self):
		self.notify_component_start('calibration')
		calibrator = OculusThumbBoundCalibrator(self._host, self._port)
		self.hand_thumb_bounds = calibrator.get_bounds()
		print(f'THUMB BOUNDS IN THE OPERATOR: {self.hand_thumb_bounds}')

	# Get the hand frame
	def _get_hand_frame(self):
		for i in range(10):
			data = self.transformed_arm_keypoint_subscriber.recv_keypoints(flags=zmq.NOBLOCK)
			if not data is None: break 
		if data is None: return None
		return np.asanyarray(data).reshape(4, 3)
	
	# Get the resolution scale mode
	def _get_resolution_scale_mode(self):
		data = self._arm_resolution_subscriber.recv_keypoints()
		res_scale = np.asanyarray(data).reshape(1)[0] # Make sure this data is one dimensional
		return res_scale
	
	# Get Homogenous matrix from the frame
	def _turn_frame_to_homo_mat(self, frame):
		t = frame[0]
		R = frame[1:]

		homo_mat = np.zeros((4, 4))
		homo_mat[:3, :3] = np.transpose(R)
		homo_mat[:3, 3] = t
		homo_mat[3, 3] = 1

		return homo_mat

	# Convert Homogenous matrix to cartesian vector 
	def _homo2cart(self, homo_mat):
		# Here we will use the resolution scale to set the translation resolution
		t = homo_mat[:3, 3]
		R = Rotation.from_matrix(
			homo_mat[:3, :3]).as_quat()

		cart = np.concatenate(
			[t, R], axis=0
		)
		return cart
	
	# convert cartesian vector to homogenous matrix
	def cart2homo(self, cart):
		homo=np.zeros((4,4))
		t = cart[0:3]
		R = Rotation.from_quat(cart[3:]).as_matrix()

		homo[0:3,3] = t
		homo[:3,:3] = R
		homo[3,:] = np.array([0,0,0,1])
		return homo

	# Get the scaled cartesian pose	
	def _get_scaled_cart_pose(self, moving_robot_homo_mat):
		# Get the cart pose without the scaling
		unscaled_cart_pose = self._homo2cart(moving_robot_homo_mat)

		# Get the current cart pose
		current_homo_mat = copy(self.robot.get_pose()['position'])
		current_cart_pose = self._homo2cart(current_homo_mat)

		# Get the difference in translation between these two cart poses
		diff_in_translation = unscaled_cart_pose[:3] - current_cart_pose[:3]
		scaled_diff_in_translation = diff_in_translation * self.resolution_scale
		
		scaled_cart_pose = np.zeros(7)
		scaled_cart_pose[3:] = unscaled_cart_pose[3:] # Get the rotation directly
		scaled_cart_pose[:3] = current_cart_pose[:3] + scaled_diff_in_translation # Get the scaled translation only

		return scaled_cart_pose
	 
	# Check if it is real robot or simulation
	def return_real(self):
		return False
			
	# Reset Teleoperation
	def _reset_teleop(self):
		# Just updates the beginning position of the arm
		print('****** RESETTING TELEOP ****** ')
		self.robot_frame=self.end_eff_position_subscriber.recv_keypoints()
		self.robot_init_H=self.cart2homo(self.robot_frame[2:])
		self.robot_moving_H = copy(self.robot_init_H)

		first_hand_frame = self._get_hand_frame()
		while first_hand_frame is None:
			first_hand_frame = self._get_hand_frame()
		self.hand_init_H = self._turn_frame_to_homo_mat(first_hand_frame)
		self.hand_init_t = copy(self.hand_init_H[:3, 3])

		self.is_first_frame = False

		return first_hand_frame
	
	# Get ARm Teleop state from Hand keypoints 
	def _get_arm_teleop_state_from_hand_keypoints(self):
		pause_state ,pause_status,pause_right =self.get_pause_state_from_hand_keypoints()
		pause_status =np.asanyarray(pause_status).reshape(1)[0] 
		return pause_state,pause_status,pause_right
	
	# Get Pause State from Hand Keypoints 
	def get_pause_state_from_hand_keypoints(self):
		transformed_hand_coords= self.transformed_hand_keypoint_subscriber.recv_keypoints()
		ring_distance = np.linalg.norm(transformed_hand_coords[OCULUS_JOINTS['ring'][-1]]- transformed_hand_coords[OCULUS_JOINTS['thumb'][-1]])
		middle_distance = np.linalg.norm(transformed_hand_coords[OCULUS_JOINTS['middle'][-1]]- transformed_hand_coords[OCULUS_JOINTS['thumb'][-1]])
		thresh = 0.04 
		pause_right= True
		if ring_distance < thresh or middle_distance < thresh:
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
	
	# Get Gripper State from Hand Keypoints 
	def get_gripper_state_from_hand_keypoints(self):
		transformed_hand_coords= self.transformed_hand_keypoint_subscriber.recv_keypoints()
		pinky_distance = np.linalg.norm(transformed_hand_coords[OCULUS_JOINTS['pinky'][-1]]- transformed_hand_coords[OCULUS_JOINTS['thumb'][-1]])
		thresh = 0.03
		gripper_fr =False
		if pinky_distance < thresh:
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
		return gripper_state , status , gripper_fr

	# Apply retargeted angles
	def _apply_retargeted_angles(self, log=False):

		# See if there is a reset in the teleop
		new_arm_teleop_state,pause_status,pause_right = self._get_arm_teleop_state_from_hand_keypoints()
		if self.is_first_frame or (self.arm_teleop_state == ARM_TELEOP_STOP and new_arm_teleop_state == ARM_TELEOP_CONT):
			moving_hand_frame = self._reset_teleop() # Should get the moving hand frame only once
		else:
			moving_hand_frame = self._get_hand_frame()
		self.arm_teleop_state = new_arm_teleop_state

		# gripper
		gripper_state,status_change, gripper_flag = self.get_gripper_state_from_hand_keypoints()
		if self.gripper_cnt==1 and status_change is True:
			self.gripper_correct_state= gripper_state
        # if status_change is True:
		if self.gripper_correct_state == GRIPPER_OPEN:
			gripper_state = -1
		elif self.gripper_correct_state == GRIPPER_CLOSE:
			gripper_state = 1

		if moving_hand_frame is None: 
			return # It means we are not on the arm mode yet instead of blocking it is directly returning
		
		self.hand_moving_H = self._turn_frame_to_homo_mat(moving_hand_frame)

		# Transformation code
		H_HI_HH = copy(self.hand_init_H) # Homo matrix that takes P_HI to P_HH - Point in Inital Hand Frame to Point in Home Hand Frame
		H_HT_HH = copy(self.hand_moving_H) # Homo matrix that takes P_HT to P_HH
		H_RI_RH = copy(self.robot_init_H) # Homo matrix that takes P_RI to P_RH

		H_HT_HI = np.linalg.pinv(H_HI_HH) @ H_HT_HH # Homo matrix that takes P_HT to P_HI
		
		#####################################################################################
		H_R_V= np.array([[0 , 0, 1, 0], 
						[0 , 1, 0, 0],
						[-1, 0, 0, 0],
						[0, 0 ,0 , 1]])
		H_T_V = np.array([[0, 0 ,1, 0],
						 [0 ,1, 0, 0],
						 [-1, 0, 0, 0],
						[0, 0, 0, 1]])
	
		H_HT_HI_r=(np.linalg.pinv(H_R_V) @ H_HT_HI @ H_R_V)[:3,:3]
		H_HT_HI_t=(np.linalg.pinv(H_T_V) @ H_HT_HI @ H_T_V)[:3,3]
		
		relative_affine = np.block(
		[[ H_HT_HI_r,  H_HT_HI_t.reshape(3, 1)], [0, 0, 0, 1]])
		
		target_translation = H_RI_RH[:3,3] + relative_affine[:3,3]
		target_rotation = H_RI_RH[:3, :3] @ relative_affine[:3,:3]
		H_RT_RH = np.block(
					[[target_rotation, target_translation.reshape(-1, 1)], [0, 0, 0, 1]])

		curr_robot_pose = self.robot_moving_H
		translation_scale = 50.0 
		T_togo = H_RT_RH[:3, 3] #* translation_scale
		R_togo = H_RT_RH[:3, :3]
		# To use simulation arm with position control use T_togo, R_togo and convert this as a pose and publish as end effector pose and gripper state


		# This part is to send relative pose as actions as Libero expects relative pose.
		T_curr = curr_robot_pose[:3, 3]
		R_curr = curr_robot_pose[:3, :3]
		rel_pos = (T_togo - T_curr) * translation_scale
		rel_rot = np.linalg.pinv(R_curr) @ R_togo
		rel_axis_angle = R.from_matrix(rel_rot).as_rotvec()
		rel_axis_angle = rel_axis_angle * 5.0 
		
		self.robot_moving_H = copy(H_RT_RH)
		action = np.concatenate([rel_pos, rel_axis_angle, [gripper_state]])

		averaged_action = moving_average(
			action,
			self.moving_Average_queue,
			self.moving_average_limit,
		)

		if self.arm_teleop_state == ARM_TELEOP_CONT and gripper_flag == False:
			self.end_eff_position_publisher.pub_keypoints(averaged_action,"endeff_coords")
		else:
			self.end_eff_position_publisher.pub_keypoints(np.concatenate([np.zeros(6),[gripper_state]]),"endeff_coords")


	