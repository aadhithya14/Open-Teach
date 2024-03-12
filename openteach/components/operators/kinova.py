import zmq
import time
import numpy as np
import matplotlib.pyplot as plt 

from copy import deepcopy as copy
from .operator import Operator
from scipy.spatial.transform import Rotation
from openteach.robot.kinova import KinovaArm
from openteach.utils.files import *
from openteach.utils.vectorops import *
from openteach.utils.network import ZMQKeypointSubscriber
from openteach.utils.timer import FrequencyTimer
from openteach.constants import *
from scipy.spatial.transform import Rotation, Slerp
from scipy.spatial.transform import Rotation as R

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

# This class is used for the kinova arm teleoperation
class KinovaArmOperator(Operator):
    def __init__(
        self,
        host,
        transformed_keypoints_port,
        moving_average_limit,
        use_filter=False,
        arm_resolution_port = None,
        teleoperation_reset_port = None,
    ):
        self.notify_component_start('kinova arm operator')
        self._transformed_arm_keypoint_subscriber = ZMQKeypointSubscriber(
            host=host,
            port=transformed_keypoints_port,
            topic='transformed_hand_frame'
        )
        self._transformed_hand_keypoint_subscriber = ZMQKeypointSubscriber(
            host=host,
            port=transformed_keypoints_port,
            topic='transformed_hand_coords'
        )

        # Initalizing the robot controller
        self._robot = KinovaArm()
        
       
        self.resolution_scale = 1 # NOTE: Get this from a socket
        self.arm_teleop_state = ARM_TELEOP_STOP # We will start as the cont

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
        
        time.sleep(1)
        robot_coords = self.robot.get_cartesian_position()
        self.robot_init_H =  self.cartesian_to_homo(robot_coords)
        self.is_first_frame = True

        self.use_filter = use_filter
        if use_filter:
            robot_init_cart = self._homo2cart(self.robot_init_H)
            self.comp_filter = Filter(robot_init_cart, comp_ratio=0.8)

        # Getting the bounds to perform linear transformation
        bounds_file = get_path_in_package(
            'components/operators/configs/kinova.yaml')
        bounds_data = get_yaml_data(bounds_file)
        self.velocity_threshold = bounds_data['velocity_threshold']

        # Frequency timer
        self._timer = FrequencyTimer(VR_FREQ)

        # Moving average queues
        self.moving_Average_queue = []
        self.moving_average_limit = moving_average_limit

    @property
    def timer(self):
        return self._timer

    @property
    def robot(self):
        return self._robot

    @property
    def transformed_arm_keypoint_subscriber(self):
        return self._transformed_arm_keypoint_subscriber
    
    @property
    def transformed_hand_keypoint_subscriber(self):
        return self._transformed_hand_keypoint_subscriber
    

    # Converts cartesian to Homogenous matrix
    def cartesian_to_homo(self,pose_aa: np.ndarray) -> np.ndarray:
        """Converts a robot pose in axis-angle format to an affine matrix.
        Args:
            pose_aa (list): [x, y, z, ax, ay, az] where (x, y, z) is the position and (ax, ay, az) is the axis-angle rotation.
            x, y, z are in mm and ax, ay, az are in radians.
        Returns:
            np.ndarray: 4x4 affine matrix [[R, t],[0, 1]]
        """        
        rotation = R.from_quat(pose_aa[3:]).as_matrix()
        translation = np.array(pose_aa[:3])
        return np.block([[rotation, translation[:, np.newaxis]],
                        [0, 0, 0, 1]])
    
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
    
    # get the translation vector
    def _get_translation_vector(self,commanded_robot_position, current_robot_position):
        return commanded_robot_position - current_robot_position
    
    # Get the rotation angular displacement
    def _get_rotation_angles(self, robot_target_orientation,current_robot_rotation_values):
        # Calculating the angular displacement between the target hand frame and the current robot frame# 
        target_rotation_state = Rotation.from_quat(robot_target_orientation)
        robot_rotation_state = Rotation.from_quat(current_robot_rotation_values)

        # Calculating the angular displacement between the target hand frame and the current robot frame
        angular_displacement = Rotation.from_matrix(
            np.matmul(robot_rotation_state.inv().as_matrix(),target_rotation_state.as_matrix())
        ).as_rotvec()

        return angular_displacement 

    # Get the displacement vector
    def _get_displacement_vector(self, commanded_robot_position, current_robot_position):
        commanded_robot_pose = np.zeros(6)
        # Transformation from translation
        commanded_robot_pose[:3] = self._get_translation_vector(
            commanded_robot_position=commanded_robot_position[:3],
            current_robot_position = current_robot_position[:3]
        ) * KINOVA_VELOCITY_SCALING_FACTOR

        # Transformation from rotation
        commanded_robot_pose[3:] = self._get_rotation_angles(
            robot_target_orientation=commanded_robot_position[3:],
            current_robot_rotation_values = current_robot_position[3:]
        ) 
        return commanded_robot_pose

    # Converts a frame to a homogenous transformation matrix
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
        t = homo_mat[:3, 3]
        R = Rotation.from_matrix(
            homo_mat[:3, :3]).as_quat()

        cart = np.concatenate(
            [t, R], axis=0
        )

        return cart
    
    # Get the scaled resolution cartesian pose
    def _get_scaled_cart_pose(self, moving_robot_homo_mat):
        # Get the cart pose without the scaling
        unscaled_cart_pose = self._homo2cart(moving_robot_homo_mat)

       
        robot_coords = self.robot.get_cartesian_position()
        current_homo_mat =  copy(self.cartesian_to_homo(robot_coords))
        current_cart_pose = self._homo2cart(current_homo_mat)

        # Get the difference in translation between these two cart poses
        diff_in_translation = unscaled_cart_pose[:3] - current_cart_pose[:3]
        scaled_diff_in_translation = diff_in_translation * self.resolution_scale
        
        scaled_cart_pose = np.zeros(7)
        scaled_cart_pose[3:] = unscaled_cart_pose[3:] # Get the rotation directly
        scaled_cart_pose[:3] = current_cart_pose[:3] + scaled_diff_in_translation # Get the scaled translation only

        return scaled_cart_pose
     
    # Reset the teleoperation
    def _reset_teleop(self):
        print('****** RESETTING TELEOP ****** ')
        robot_coords = self.robot.get_cartesian_position()
        self.robot_init_H =  self.cartesian_to_homo(robot_coords)

        first_hand_frame = self._get_hand_frame()
        while first_hand_frame is None:
            first_hand_frame = self._get_hand_frame()
        self.hand_init_H = self._turn_frame_to_homo_mat(first_hand_frame)
        self.hand_init_t = copy(self.hand_init_H[:3, 3])
        self.is_first_frame = False

        return first_hand_frame
    

    # Apply retargeted angles
    def _apply_retargeted_angles(self, log=False):

        # See if there is a reset in the teleop
        new_arm_teleop_state = self._get_arm_teleop_state()
        if self.is_first_frame or (self.arm_teleop_state == ARM_TELEOP_STOP and new_arm_teleop_state == ARM_TELEOP_CONT):
            moving_hand_frame = self._reset_teleop() # Should get the moving hand frame only once
        else:
            moving_hand_frame = self._get_hand_frame()
        if moving_hand_frame is None: 
            if new_arm_teleop_state == ARM_TELEOP_STOP:
                self.arm_teleop_state = new_arm_teleop_state
            return # It means we are not on the arm mode yet instead of blocking it is directly returning
        current_robot_position = self.robot.get_cartesian_position()

        if current_robot_position is None:
            if new_arm_teleop_state == ARM_TELEOP_STOP:
                self.arm_teleop_state = new_arm_teleop_state
            return
        
        self.arm_teleop_state = new_arm_teleop_state

        arm_teleoperation_scale_mode = self._get_resolution_scale_mode()

        if arm_teleoperation_scale_mode == ARM_HIGH_RESOLUTION:
            self.resolution_scale = 1
        elif arm_teleoperation_scale_mode == ARM_LOW_RESOLUTION:
            self.resolution_scale = 0.6
        
        # Find the moving hand frame
        self.hand_moving_H = self._turn_frame_to_homo_mat(moving_hand_frame)

        # Transformation code
        H_HI_HH = copy(self.hand_init_H) # Homo matrix that takes P_HI to P_HH - Point in Inital Hand Frame to Point in Home Hand Frame
        H_HT_HH = copy(self.hand_moving_H) # Homo matrix that takes P_HT to P_HH
        H_RI_RH = copy(self.robot_init_H) # Homo matrix that takes P_RI to P_RH

        # Find the relative transformation in human hand space.
        H_HT_HI = np.linalg.pinv(H_HI_HH) @ H_HT_HH # Homo matrix that takes P_HT to P_HI

        # Transformation matrix
        H_R_V=  [[0,1,0,0],
                [-1,0,0,0],
                [0,0,1,0],
                [0,0,0,1]]
        
        # Find the relative transform and apply it to robot initial position
        H_R_R= (np.linalg.pinv(H_R_V)@H_HT_HI@H_R_V)[:3,:3]
        H_R_T= (np.linalg.pinv(H_R_V)@H_HT_HI@H_R_V)[:3,3]
        H_F_H=np.block([[H_R_R,H_R_T.reshape(3,1)],[np.array([0,0,0]),1]])
        H_RT_RH = H_RI_RH  @ H_F_H # Homo matrix that takes P_RT to P_RH

        self.robot_moving_H = copy(H_RT_RH)

        final_pose = self._get_scaled_cart_pose(self.robot_moving_H)
        # filter the pose in case needed
        if self.use_filter:
            final_pose = self.comp_filter(final_pose)

        # Calculated velocity
        calculated_velocity = self._get_displacement_vector(final_pose, current_robot_position)
        averaged_velocity = moving_average(
            calculated_velocity,
            self.moving_Average_queue,
            self.moving_average_limit
        )

        # Filter the velocities to make it less oscillatory
        for axis in range(len(averaged_velocity[:3])):
            if abs(averaged_velocity[axis]) < self.velocity_threshold:
                averaged_velocity[axis] = 0

        self.robot.move_velocity(averaged_velocity, 1 / VR_FREQ)


    # NOTE: This is for debugging should remove this when needed
    def stream(self):
        self.notify_component_start('{} control'.format(self.robot.name))
        print("Start controlling the robot hand using the Oculus Headset.\n")

        # Assume that the initial position is considered initial after 3 seconds of the start
        while True:
            try:
                if self.robot.get_joint_position() is not None:
                    self.timer.start_loop()

                    # Retargeting function
                    self._apply_retargeted_angles(log=False)

                    self.timer.end_loop()
            except KeyboardInterrupt:
                break

        self.transformed_arm_keypoint_subscriber.stop()
        print('Stopping the teleoperator!')