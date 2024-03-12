import numpy as np
import matplotlib.pyplot as plt
import zmq

from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

from copy import deepcopy as copy
from shapely.geometry import Point, Polygon 
from shapely.ops import nearest_points
from openteach.constants import *
from openteach.utils.timer import FrequencyTimer
from openteach.utils.network import ZMQKeypointSubscriber, ZMQKeypointPublisher
from openteach.robot.allegro.allegro_retargeters import AllegroKDLControl, AllegroJointControl
from openteach.utils.vectorops import *
from openteach.utils.files import *
#from openteach.robot.franka import FrankaArm
from scipy.spatial.transform import Rotation, Slerp
from .operator import Operator
from .calibrators.allegro import OculusThumbBoundCalibrator


np.set_printoptions(precision=2, suppress=True)

def test_filter(save_data=False):
    if save_data:
        filter = CompStateFilter(np.asarray( [ 0.575466,   -0.17820767,  0.23671454, -0.281564 ,  -0.6797597,  -0.6224841 ,  0.2667619 ]))
        timer = FrequencyTimer(VR_FREQ)

        i = 0
        while True:
            try:
                timer.start_loop()

                rand_pos = np.random.randn(7)
                filtered_pos = filter(rand_pos)

                print('rand_pos: {} - filtered_pos: {}'.format(rand_pos, filtered_pos)) 

                if i == 0:
                    all_poses = np.expand_dims(np.stack([rand_pos, filtered_pos], axis=0), 0)
                else:
                    all_poses = np.concatenate([
                        all_poses,
                        np.expand_dims(np.stack([rand_pos, filtered_pos], axis=0), 0)
                    ], axis=0)

                print('all_poses shape: {}'.format(
                    all_poses.shape
                ))

                i += 1
                timer.end_loop()

            except KeyboardInterrupt:
                np.save('all_poses.npy', all_poses)
                break

    else:
        all_poses = np.load('all_poses.npy')
        fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(10,10))
        pbar = tqdm(total=len(all_poses))
        for i in range(len(all_poses)):
            filtered_pos = all_poses[:i+1,1,:]
            rand_pos = all_poses[:i+1,0,:]

            # print('filtered_pos.shape: {}'.format(filtered_pos.shape))
            for j in range(filtered_pos.shape[1]):
                axs[int(j / 3), j % 3].plot(filtered_pos[:,j], label='Filtered')
                axs[int(j / 3), j % 3].plot(rand_pos[:,j], label='Actual')
                axs[int(j / 3), j % 3].set_title(f'{j}th Axes')
                axs[int(j / 3), j % 3].legend()

            pbar.update(1)
            plt.savefig(os.path.join(f'all_poses/state_{str(i).zfill(3)}.png'))
            fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(10,10))

# Rotation should be filtered when it's being sent
class CompStateFilter:
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
    


class MovingAllegroSimOperator(Operator):
    def __init__(
        self,
        host,
        transformed_keypoints_port,
        finger_configs,
        stream_configs,
        stream_oculus,
        jointanglepublishport,
        jointanglesubscribeport,
        endeff_publish_port,
        endeffpossubscribeport,
        moving_average_limit,
        allow_rotation=False,
        arm_type='main_arm',
        use_filter=False,
        arm_resolution_port = None,
        teleoperation_reset_port = None,
        human_teleop = False
    ):
        self.notify_component_start('franka arm operator')
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

        self.human_teleop= human_teleop
        self.joint_angle_publisher = ZMQKeypointPublisher(
            host = host,
            port = jointanglepublishport
        )

        self.joint_angle_subscriber = ZMQKeypointSubscriber(
            host = host,
            port = jointanglesubscribeport,
            topic= 'current_angles'
        )

        # Initalizing the robot controller
        self.arm_type = arm_type
        self.allow_rotation = allow_rotation
        self.resolution_scale = 1 # NOTE: Get this from a socket
        self.arm_teleop_state = ARM_TELEOP_STOP # We will start as the cont

        self.fingertip_solver = AllegroKDLControl()
        self.finger_joint_solver = AllegroJointControl()



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

        self.end_eff_position_subscriber = ZMQKeypointSubscriber(
            host = host,
            port =  endeffpossubscribeport,
            topic = 'endeff_coords'

        )

        self.end_eff_position_publisher = ZMQKeypointPublisher(
            host = host,
            port = endeff_publish_port
        )

        #Adding Allegro Hand Specific things 

        self.finger_configs = finger_configs
        self.moving_average_queues = {
            'thumb': [],
            'index': [],
            'middle': [],
            'ring': []
        }

        # Calibrating to get the thumb bounds
        self._calibrate_bounds()
        self._stream_oculus=stream_oculus
        self.stream_configs=stream_configs
        
       
        
        # Getting the bounds for the allegro hand
        allegro_bounds_path = get_path_in_package('components/operators/configs/allegro.yaml')
        self.allegro_bounds = get_yaml_data(allegro_bounds_path)

        self._timer = FrequencyTimer(VR_FREQ)

        if self.finger_configs['three_dim']:
            self.thumb_angle_calculator = self._get_3d_thumb_angles
        else:
            self.thumb_angle_calculator = self._get_2d_thumb_angles
        #torch.set_num_threads(1)
       
        self.real=False
        self._robot='Allegro_Moving_Sim'
       
        self.is_first_frame = True
        #print('ROBOT INIT H: \n{}'.format(self.robot_init_H))

        self.use_filter = use_filter
        if use_filter:
            robot_init_cart = self._homo2cart(self.robot_init_H)
            self.comp_filter = CompStateFilter(robot_init_cart, comp_ratio=0.8)

        if allow_rotation:
            self.initial_quat = np.array(
                [-0.27686286, -0.66575766, -0.63895273,  0.26805457])
            self.rotation_axis = np.array([0, 0, 1])

        # Getting the bounds to perform linear transformation
        bounds_file = get_path_in_package(
            'components/operators/configs/franka.yaml')
        bounds_data = get_yaml_data(bounds_file)

        # Bounds for performing linear transformation
        self.corresponding_robot_axes = bounds_data['corresponding_robot_axes']
        self.franka_bounds = bounds_data['robot_bounds']
        self.wrist_bounds = bounds_data['wrist_bounds']

        # Matrices to reorient the end-effector rotation frames
        self.frame_realignment_matrix = np.array(
            bounds_data['frame_realignment_matrix']).reshape(3, 3)
        self.rotation_realignment_matrix = np.array(
            bounds_data['rotation_alignment_matrix']).reshape(3, 3)

        # Frequency timer
        self._timer = FrequencyTimer(VR_FREQ)

        self.direction_counter = 0
        self.current_direction = 0

        # Moving average queues
        self.moving_Average_queue = []
        self.moving_average_limit = moving_average_limit

        self.hand_frames = []

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
    
    def _calibrate_bounds(self):
        self.notify_component_start('calibration')
        calibrator = OculusThumbBoundCalibrator(self._host, self._port)
        self.hand_thumb_bounds = calibrator.get_bounds() # Provides [thumb-index bounds, index-middle bounds, middle-ring-bounds]
        print(f'THUMB BOUNDS IN THE OPERATOR: {self.hand_thumb_bounds}')

    def _get_hand_frame(self):
        for i in range(10):
            data = self.transformed_arm_keypoint_subscriber.recv_keypoints(flags=zmq.NOBLOCK)
            if not data is None: break 
        # print('data: {}'.format(data))
        if data is None: return None
        return np.asanyarray(data).reshape(4, 3)
    
    def _get_resolution_scale_mode(self):
        data = self._arm_resolution_subscriber.recv_keypoints()
        res_scale = np.asanyarray(data).reshape(1)[0] # Make sure this data is one dimensional
        return res_scale
        # return ARM_LOW_RESOLUTION    

    def _get_arm_teleop_state(self):
        reset_stat = self._arm_teleop_state_subscriber.recv_keypoints()
        reset_stat = np.asanyarray(reset_stat).reshape(1)[0] # Make sure this data is one dimensional
        return reset_stat
    
    #Hand Teleoperation Specific Functions
    
    def _get_finger_coords(self):
        raw_keypoints = self.transformed_hand_keypoint_subscriber.recv_keypoints()
        return dict(
            index = np.vstack([raw_keypoints[0], raw_keypoints[OCULUS_JOINTS['index']]]),
            middle = np.vstack([raw_keypoints[0], raw_keypoints[OCULUS_JOINTS['middle']]]),
            ring = np.vstack([raw_keypoints[0], raw_keypoints[OCULUS_JOINTS['ring']]]),
            thumb =  np.vstack([raw_keypoints[0], raw_keypoints[OCULUS_JOINTS['thumb']]])
        )

    def _get_2d_thumb_angles(self, thumb_keypoints, curr_angles):
        for idx, thumb_bounds in enumerate(self.hand_thumb_bounds):
            if coord_in_bound(thumb_bounds[:4], thumb_keypoints[:2]) > -1:
                return self.fingertip_solver.thumb_motion_2D(
                    hand_coordinates = thumb_keypoints, 
                    xy_hand_bounds = thumb_bounds[:4],
                    yz_robot_bounds = self.allegro_bounds['thumb_bounds'][idx]['projective_bounds'],
                    robot_x_val = self.allegro_bounds['x_coord'],
                    moving_avg_arr = self.moving_average_queues['thumb'], 
                    curr_angles = curr_angles
                )
        
        return curr_angles

    def _get_3d_thumb_angles(self, thumb_keypoints, curr_angles):
       
        # We will be using polygon implementations of shapely library to test this
        planar_point = Point(thumb_keypoints)
        planar_thumb_bounds = Polygon(self.hand_thumb_bounds[:4])

        # Get the closest point from the thumb to the point
        # this will return the point if it's inside the bounds
        closest_point = nearest_points(planar_thumb_bounds, planar_point)[0]
        closest_point_coords = [closest_point.x, closest_point.y, thumb_keypoints[2]]
        return self.fingertip_solver.thumb_motion_3D(
            hand_coordinates = closest_point_coords,
            xy_hand_bounds = self.hand_thumb_bounds[:4],
            yz_robot_bounds = self.allegro_bounds['thumb_bounds'][0]['projective_bounds'], # NOTE: We assume there is only one bound now
            z_hand_bound = self.hand_thumb_bounds[4],
            x_robot_bound = self.allegro_bounds['thumb_bounds'][0]['x_bounds'],
            moving_avg_arr = self.moving_average_queues['thumb'], 
            curr_angles = curr_angles
        )

    def _generate_frozen_angles(self, joint_angles, finger_type):
        for idx in range(ALLEGRO_JOINTS_PER_FINGER):
            if idx > 0:
                joint_angles[idx + ALLEGRO_JOINT_OFFSETS[finger_type]] = 0
            else:
                joint_angles[idx + ALLEGRO_JOINT_OFFSETS[finger_type]] = 0

        return joint_angles

    def _clip_coords(self, coords):
        # TODO - clip the coordinates
        return coords

    def _round_displacement(self, displacement):
        return np.where(np.abs(displacement) * 1e2 > 1.5, displacement, 0)

    def _realign_frame(self, hand_frame):
        return self.frame_realignment_matrix @ hand_frame

    def _turn_frame_to_homo_mat(self, frame):
        t = frame[0]
        R = frame[1:]

        homo_mat = np.zeros((4, 4))
        homo_mat[:3, :3] = np.transpose(R)
        homo_mat[:3, 3] = t
        homo_mat[3, 3] = 1

        return homo_mat

    def _homo2cart(self, homo_mat):
        # Here we will use the resolution scale to set the translation resolution
        t = homo_mat[:3, 3]
        R = Rotation.from_matrix(
            homo_mat[:3, :3]).as_quat()

        cart = np.concatenate(
            [t, R], axis=0
        )

        return cart
    
    def cart2homo(self, cart):
        homo=np.zeros((4,4))
        t = cart[0:3]
        R = Rotation.from_quat(cart[3:]).as_matrix()

        homo[0:3,3] = t
        homo[:3,:3] = R
        homo[3,:] = np.array([0,0,0,1])
        return homo

    
    def _get_scaled_cart_pose(self, moving_robot_homo_mat):
        # Get the cart pose without the scaling
        unscaled_cart_pose = self._homo2cart(moving_robot_homo_mat)

        # Get the current cart pose
        current_homo_mat = copy(self.robot.get_pose()['position'])
        current_cart_pose = self._homo2cart(current_homo_mat)

        # Get the difference in translation between these two cart poses
        diff_in_translation = unscaled_cart_pose[:3] - current_cart_pose[:3]
        scaled_diff_in_translation = diff_in_translation * self.resolution_scale
        # print('SCALED_DIFF_IN_TRANSLATION: {}'.format(scaled_diff_in_translation))
        
        scaled_cart_pose = np.zeros(7)
        scaled_cart_pose[3:] = unscaled_cart_pose[3:] # Get the rotation directly
        scaled_cart_pose[:3] = current_cart_pose[:3] + scaled_diff_in_translation # Get the scaled translation only

        return scaled_cart_pose

    def return_real(self):
        return self.real
            

    def _reset_teleop(self):
        # Just updates the beginning position of the arm
        print('****** RESETTING TELEOP ****** ')
        self.robot_frame=self.end_eff_position_subscriber.recv_keypoints()
        self.robot_init_H=self.cart2homo(self.robot_frame)

        first_hand_frame = self._get_hand_frame()
        while first_hand_frame is None:
            first_hand_frame = self._get_hand_frame()
        self.hand_init_H = self._turn_frame_to_homo_mat(first_hand_frame)
        self.hand_init_t = copy(self.hand_init_H[:3, 3])

        self.is_first_frame = False

        return first_hand_frame

    def _apply_retargeted_angles(self, log=False):
        # Hand Teleoperation Code 
        cart=None
        hand_keypoints = self._get_finger_coords()
        desired_joint_angles=copy(self.joint_angle_subscriber.recv_keypoints())
        if not self.finger_configs['freeze_index'] and not self.finger_configs['no_index']:
            desired_joint_angles = self.finger_joint_solver.calculate_finger_angles(
                finger_type = 'index',
                finger_joint_coords = hand_keypoints['index'],
                curr_angles = desired_joint_angles,
                moving_avg_arr = self.moving_average_queues['index']
            )
        elif self.finger_configs['freeze_index']:
            self._generate_frozen_angles(desired_joint_angles, 'index')
        else:
            self._generate_frozen_angles(desired_joint_angles, 'index')
            print("No index")

            pass

         # Movement for the middle finger
        if not self.finger_configs['freeze_middle'] and not self.finger_configs['no_middle']:
            desired_joint_angles = self.finger_joint_solver.calculate_finger_angles(
                finger_type = 'middle',
                finger_joint_coords = hand_keypoints['middle'],
                curr_angles = desired_joint_angles,
                moving_avg_arr = self.moving_average_queues['middle']
            )
        elif self.finger_configs['freeze_middle']:
            self._generate_frozen_angles(desired_joint_angles, 'middle')

        else:
            print("No middle")
            pass

        # Movement for the ring finger
        # Calculating the translatory joint angles
        if not self.finger_configs['freeze_ring'] and not self.finger_configs['no_ring']:
            desired_joint_angles = self.finger_joint_solver.calculate_finger_angles(
                finger_type = 'ring',
                finger_joint_coords = hand_keypoints['ring'],
                curr_angles = desired_joint_angles,
                moving_avg_arr = self.moving_average_queues['ring']
            )
        elif self.finger_configs['freeze_ring']:
            self._generate_frozen_angles(desired_joint_angles, 'ring')
        else: 
            print("No ring")
            pass
       

        # Movement for the thumb finger - we disable 3D motion just for the thumb
        if not self.finger_configs['freeze_thumb'] and not self.finger_configs['no_thumb']:
            desired_joint_angles = self.thumb_angle_calculator(hand_keypoints['thumb'][-1], desired_joint_angles) # Passing just the tip coordinates
        elif self.finger_configs['freeze_thumb']:
            self._generate_frozen_angles(desired_joint_angles, 'thumb')
        else:
            self._generate_frozen_angles(desired_joint_angles, 'thumb')
            print("No thumb")
      
            
        self.joint_angle_publisher.pub_keypoints(desired_joint_angles,'desired_angles')    

        #Moving End Effector Teleoperation Code
        # See if there is a reset in the teleop
        new_arm_teleop_state = self._get_arm_teleop_state()
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

        if moving_hand_frame is None : 
            return # It means we are not on the arm mode yet instead of blocking it is directly returning
        
        self.hand_moving_H = self._turn_frame_to_homo_mat(moving_hand_frame)
        # Transformation code
        H_HI_HH = copy(self.hand_init_H) # Homo matrix that takes P_HI to P_HH - Point in Inital Hand Frame to Point in Home Hand Frame
        H_HT_HH = copy(self.hand_moving_H) # Homo matrix that takes P_HT to P_HH
        H_RI_RH = copy(self.robot_init_H) # Homo matrix that takes P_RI to P_RH
        
        H_R_V= np.array([[0 , 1, 0, 0], 
                        [-1 , 0, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0 ,0 , 1]])
        H_T_V = np.array([[0, 1 ,0, 0],
                         [-1 ,0, 0, 0],
                         [0, 0, 1, 0],
                        [0, 0, 0, 1]])

        H_HT_HI = np.linalg.pinv(H_HI_HH) @ H_HT_HH # Homo matrix that takes P_HT to P_HI
        H_HT_HI_r=(np.linalg.pinv(H_R_V)@H_HT_HI@H_R_V)[:3,:3]
        H_HT_HI_t=(np.linalg.pinv(H_T_V)@H_HT_HI@H_T_V)[:3,3]
        relative_affine = np.block(
        [[ H_HT_HI_r, H_HT_HI_t.reshape(3,1)], [0, 0, 0, 1]])
        H_RT_RH= H_RI_RH @ relative_affine
        self.robot_moving_H = copy(H_RT_RH)

        if log:
            print('** ROBOT MOVING H **\n{}\n** ROBOT INIT H **\n{}\n'.format(
                self.robot_moving_H, self.robot_init_H))
            print('** HAND MOVING H: **\n{}\n** HAND INIT H: **\n{} - HAND INIT T: {}'.format(
                self.hand_moving_H, self.hand_init_H, self.hand_init_t
            ))
            print('***** TRANSFORM MULT: ******\n{}'.format(
                H_HT_HI
            ))

            print('\n------------------------------------\n\n\n')

        # Use the resolution scale to get the final cart pose
        cart=self._homo2cart(H_RT_RH)
        print("Endeff in operator", cart)
        self.end_eff_position_publisher.pub_keypoints(cart,"endeff_coords")

        if self.use_filter:
            final_pose = self.comp_filter(final_pose)


    
