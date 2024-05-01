from copy import deepcopy as copy
#Holo-bot Components
from openteach.utils.network import ZMQKeypointSubscriber, ZMQKeypointPublisher
from .operator import Operator
from shapely.geometry import Point, Polygon 
from shapely.ops import nearest_points
from .calibrators.allegro import OculusThumbBoundCalibrator
# from openteach.robot.allegro.allegro import AllegroHand
from openteach.robot.allegro.allegro_retargeters import AllegroKDLControl, AllegroJointControl
from openteach.utils.files import *
from openteach.utils.vectorops import coord_in_bound
from openteach.utils.timer import FrequencyTimer
from openteach.constants import *
from openteach.components.recorders import *
from openteach.components.sensors import *
from openteach.utils.images import rotate_image, rescale_image
from collections import deque

#Isaac Gym components
from isaacgym import gymapi, gymutil
import gym
from isaacgym import gymtorch
from isaacgym.torch_utils import *



class AllegroHandSimOperator(Operator):
    def __init__(
            self,  
            host,
            transformed_keypoints_port, 
            finger_configs,
            stream_configs,stream_oculus,
            jointanglepublishport,
            jointanglesubscribeport):
        self.notify_component_start('allegro hand sim operator')
        self._host, self._port = host, transformed_keypoints_port
        self._hand_transformed_keypoint_subscriber = ZMQKeypointSubscriber(
            host = self._host,
            port = self._port,
            topic = 'transformed_hand_coords'
        )
           # Adding the thumb debugging components
        self._arm_transformed_keypoint_subscriber = ZMQKeypointSubscriber(
            host = self._host,
            port = self._port,
            topic = 'transformed_hand_frame'
        )
        # Adding the Joint Angle Publisher and Subscriber
        self.joint_angle_publisher = ZMQKeypointPublisher(
            host = host,
            port = jointanglepublishport
        )

        self.joint_angle_subscriber = ZMQKeypointSubscriber(
            host = host,
            port = jointanglesubscribeport,
            topic= 'current_angles'
        )
        # Initializing the solvers
        self.finger_configs = finger_configs
        self.fingertip_solver = AllegroKDLControl()
        self.finger_joint_solver = AllegroJointControl()


       
        # Initializing the queues
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

        # Using 3 dimensional thumb motion or two dimensional thumb motion
        if self.finger_configs['three_dim']:
            self.thumb_angle_calculator = self._get_3d_thumb_angles
        else:
            self.thumb_angle_calculator = self._get_2d_thumb_angles

        self._robot='Allegro_Sim'
    @property
    def timer(self):
        return self._timer

    @property
    def robot(self):
        return self._robot
    
    def return_real(self):
        return False

    
    @property
    def transformed_hand_keypoint_subscriber(self):
        return self._hand_transformed_keypoint_subscriber
    
    @property
    def transformed_arm_keypoint_subscriber(self):
        return self._arm_transformed_keypoint_subscriber

    # Calibrate the bounds for the thumb
    def _calibrate_bounds(self):
        self.notify_component_start('calibration')
        calibrator = OculusThumbBoundCalibrator(self._host, self._port)
        self.hand_thumb_bounds = calibrator.get_bounds() # Provides [thumb-index bounds, index-middle bounds, middle-ring-bounds]
        print(f'THUMB BOUNDS IN THE OPERATOR: {self.hand_thumb_bounds}')  

    # Get Finger Coordinates
    def _get_finger_coords(self):
        raw_keypoints = self.transformed_hand_keypoint_subscriber.recv_keypoints()
        return dict(
            index = np.vstack([raw_keypoints[0], raw_keypoints[OCULUS_JOINTS['index']]]),
            middle = np.vstack([raw_keypoints[0], raw_keypoints[OCULUS_JOINTS['middle']]]),
            ring = np.vstack([raw_keypoints[0], raw_keypoints[OCULUS_JOINTS['ring']]]),
            thumb =  np.vstack([raw_keypoints[0], raw_keypoints[OCULUS_JOINTS['thumb']]])
        )

    # Get Thumb 2D Angles
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

    # Get Thumb 3D Angles
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
    
    # Generate Frozen Angles
    def _generate_frozen_angles(self, joint_angles, finger_type):
        for idx in range(ALLEGRO_JOINTS_PER_FINGER):
            if idx > 0:
                joint_angles[idx + ALLEGRO_JOINT_OFFSETS[finger_type]] = 0
            else:
                joint_angles[idx + ALLEGRO_JOINT_OFFSETS[finger_type]] = 0

        return joint_angles

    # Apply Retargeted Angles
    def _apply_retargeted_angles(self):
        
        print("Applying retargeted angles") 
        hand_keypoints = self._get_finger_coords()
        desired_joint_angles=copy(self.joint_angle_subscriber.recv_keypoints())  
        print("Desired Joint Agnels", desired_joint_angles)     
        # Movement for the index finger
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
       
        
       
    

    
    