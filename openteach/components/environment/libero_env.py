import os
import time
import numpy as np
from openteach.utils.timer import FrequencyTimer
from openteach.utils.network import ZMQCameraPublisher, ZMQCompressedImageTransmitter,ZMQKeypointPublisher,ZMQKeypointSubscriber
from openteach.components.environment.arm_env import Arm_Env
from openteach.constants import *
from openteach.utils.images import rescale_image

import robosuite.utils.transform_utils as T
from libero.libero import benchmark, get_libero_path

from robosuite import load_controller_config
import libero.libero.envs.bddl_utils as BDDLUtils
from libero.libero.envs import *

# Libero Environment class 
class LiberoEnv(Arm_Env):
	def __init__(self,
			 host,
			 camport,
			 timestamppublisherport,
			 endeff_publish_port,
			 endeffpossubscribeport,
			 robotposepublishport,
			 stream_oculus,
			 suite_name,
			 task_name,
	):
		  
		self._timer=FrequencyTimer(VR_FREQ)
		self.host=host
		self.camport=camport
		self.stream_oculus=stream_oculus

		self._stream_oculus = stream_oculus

		#Define ZMQ pub/sub
		#Port for publishing rgb images.
		self.rgb_publisher = ZMQCameraPublisher(
			host = host,
			port = camport
		)
		# for ego-centric view
		self.rgb_publisher_ego = ZMQCameraPublisher(
			host = host,
			port = camport + 1
		)
		
		#Publishing the stream into the oculus.
		if self._stream_oculus:
			self.rgb_viz_publisher = ZMQCompressedImageTransmitter(
				host = host,
				port = camport + VIZ_PORT_OFFSET
			)
		#Publisher for Depth data
		self.depth_publisher = ZMQCameraPublisher(
			host = host,
			port = camport + DEPTH_PORT_OFFSET 
		)
		# for ego-centric view
		self.depth_publisher_ego = ZMQCameraPublisher(
			host = host,
			port = camport + 1 + DEPTH_PORT_OFFSET 
		)

		#Publisher for endeffector Positions 
		self.endeff_publisher = ZMQKeypointPublisher(
			host = host,
			port = endeff_publish_port
		)

		#Publisher for endeffector Velocities
		self.endeff_pos_subscriber = ZMQKeypointSubscriber(
			host = host,
			port = endeffpossubscribeport,
			topic='endeff_coords'
		)

		# Robot pose publisher
		self.robot_pose_publisher = ZMQKeypointPublisher(
			host = host,
			port = robotposepublishport
		)

		#Publisher for timestamps
		self.timestamp_publisher = ZMQKeypointPublisher(
			host=host,
			port=timestamppublisherport
		)

		self.name="Libero_Sim"

		# initialize env
		print("Initializing Environment")
		benchmark_dict = benchmark.get_benchmark_dict()
		task_suite = benchmark_dict[suite_name]()
		# get task id from list of task names
		task_id = task_suite.get_task_names().index(task_name)
		# create environment
		task = task_suite.get_task(task_id)
		task_name = task.name
		task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)

		# Get controller config
		controller_config = load_controller_config(default_controller="OSC_POSE")

		# Create argument configuration
		config = {
			"robots": ["Panda"],
			"controller_configs": controller_config,
		}

		problem_info = BDDLUtils.get_problem_info(task_bddl_file)

		# Create environment
		problem_name = problem_info["problem_name"]
		domain_name = problem_info["domain_name"]
		language_instruction = problem_info["language_instruction"]

		self.env = TASK_MAPPING[problem_name](
			bddl_file_name=task_bddl_file,
			**config,
			has_renderer=True,
			has_offscreen_renderer=False,
			render_camera="agentview",
			ignore_done=True,
			use_camera_obs=False,
			reward_shaping=True,
			control_freq=20,
		)
		seed = np.random.randint(0, 100000)
		self.env.seed(seed)
		position = self.reset()
		self.robot_pose_publisher.pub_keypoints(position, 'robot_pose')
		
	# Reset the environment
	def reset(self):
		self.obs = self.env.reset()
		return self.env.get_robot_state_vector(self.obs) 

	# Get the RGB and Depth Images
	def get_rgb_depth_images(self, camera_name=None):
		if camera_name is None:
			camera_name = 'agentview'
		rgb, depth = self.env.sim.render(width=480, height=480, camera_name=camera_name, depth=True)
		rgb = rgb[::-1, :, ::-1].astype(np.uint8)
		depth = depth[::-1, :].astype(np.uint8)
		time = self.get_time()
		return rgb, depth, time
	
	# Get the time
	def get_time(self):
		return time.time()
	
	# Get the endeffector position
	def get_endeff_position(self):
		return self.env.get_robot_state_vector(self.obs) # [gripper_pos, eef_pos, eef_quat]
			
	@property              
	def timer(self):
		return self._timer
	   			
	# Take action
	def take_action(self):
		action = self.endeff_pos_subscriber.recv_keypoints()
		self.obs, _, _, _ = self.env.step(action)                   

	# Stream the environment
	def stream(self):
		self.notify_component_start('{} environment'.format(self.name))
		
		while True:
			#try:
			self.timer.start_loop() 
			#Get RGB Images and Depth Images
			color_image,depth_image,timestamp=self.get_rgb_depth_images()
			color_image_ego, depth_image_ego, timestamp_ego=self.get_rgb_depth_images(camera_name='robot0_eye_in_hand')
			#Publishes RGB images
			self.rgb_publisher.pub_rgb_image(color_image, timestamp)
			self.rgb_publisher_ego.pub_rgb_image(color_image_ego, timestamp_ego)
			self.timestamp_publisher.pub_keypoints(timestamp,'timestamps')
			#Set this to True        
			if self._stream_oculus:
				self.rgb_viz_publisher.send_image(rescale_image(color_image, 2)) # 128 * 128

			# Publishing the depth images
			self.depth_publisher.pub_depth_image(depth_image, timestamp)
			self.depth_publisher_ego.pub_depth_image(depth_image_ego, timestamp_ego)
			
			#Gets the endeffector position       
			position=self.get_endeff_position()
			#Publishes the endeffector position so that Operator can use.
			self.endeff_publisher.pub_keypoints(position,'endeff_coords')


			#Takes Action
			self.take_action()

			# Publish robot pose
			position = self.get_endeff_position()
			self.robot_pose_publisher.pub_keypoints(position, 'robot_pose')

			self.timer.end_loop()

	

			
