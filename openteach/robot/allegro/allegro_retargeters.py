import numpy as np
from abc import ABC
from copy import deepcopy as copy
from .allegro_kdl import AllegroKDL
from openteach.utils.network import ZMQKeypointPublisher, ZMQKeypointSubscriber
from openteach.utils.files import *
from openteach.utils.vectorops import *


class AllegroKinematicControl(ABC):
    def __init__(self, bounded_angles = True):
        np.set_printoptions(suppress = True)

        # Loading the Allegro Hand configs
        self.hand_configs = get_yaml_data(get_path_in_package("robot/allegro/configs/allegro_info.yaml"))
        self.finger_configs = get_yaml_data(get_path_in_package("robot/allegro/configs/allegro_link_info.yaml"))
        self.bound_info = get_yaml_data(get_path_in_package("robot/allegro/configs/allegro_bounds.yaml"))

        self.time_steps = self.bound_info['time_steps']

        self.bounded_angles = bounded_angles
        self.bounds = {}
        for finger in self.hand_configs['fingers'].keys():
            self.bounds[finger] = np.array(self.bound_info['jointwise_angle_bounds'][
                self.finger_configs['links_info'][finger]['offset'] : self.finger_configs['links_info'][finger]['offset'] + 4
            ])

    def _get_curr_finger_angles(self, curr_angles, finger_type):
        return np.array(curr_angles[
            self.finger_configs['links_info'][finger_type]['offset'] : self.finger_configs['links_info'][finger_type]['offset'] + 4
        ])


class AllegroJointControl(AllegroKinematicControl):
    def __init__(self, bounded_angles = True):
        super().__init__(bounded_angles)
        np.set_printoptions(suppress = True)

        self.linear_scaling_factors = self.bound_info['linear_scaling_factors']
        self.rotatory_scaling_factors = self.bound_info['rotatory_scaling_factors']

    def _get_filtered_angles(self, finger_type, calc_finger_angles, curr_angles, moving_avg_arr):
        curr_finger_angles = self._get_curr_finger_angles(curr_angles, finger_type)
        avg_finger_angles = moving_average(calc_finger_angles, moving_avg_arr, self.time_steps)       
        desired_angles = np.array(copy(curr_angles))

        # Applying angular bounds
        if self.bounded_angles is True:
            #print(avg_finger_angles)
            #print(curr_finger_angles)
            del_finger_angles = avg_finger_angles - curr_finger_angles
            clipped_del_finger_angles = np.clip(del_finger_angles, - self.bounds[finger_type], self.bounds[finger_type])

            for idx in range(self.hand_configs['joints_per_finger']):
                desired_angles[self.finger_configs['links_info'][finger_type]['offset'] + idx] += clipped_del_finger_angles[idx]
        else:
            for idx in range(self.hand_configs['joints_per_finger']):
                desired_angles[self.finger_configs['links_info'][finger_type]['offset'] + idx] = avg_finger_angles[idx]

        return desired_angles 

    def calculate_finger_angles(self, finger_type, finger_joint_coords, curr_angles, moving_avg_arr):
        translatory_angles = []
        for idx in range(self.hand_configs['joints_per_finger'] - 1): # Ignoring the rotatory joint
            angle = calculate_angle(
                finger_joint_coords[idx],
                finger_joint_coords[idx + 1],
                finger_joint_coords[idx + 2]
            )
            translatory_angles.append(angle * self.linear_scaling_factors[idx])

        rotatory_angle = [self.calculate_finger_rotation(finger_joint_coords) * self.rotatory_scaling_factors[finger_type]] 
        calc_finger_angles = rotatory_angle + translatory_angles
        filtered_angles = self._get_filtered_angles(finger_type, calc_finger_angles, curr_angles, moving_avg_arr)
        return filtered_angles


    def calculate_finger_rotation(self, finger_joint_coords):
        angle = calculate_angle(finger_joint_coords[0][:1], finger_joint_coords[1][:1], finger_joint_coords[-1][:1])
        
        # Checking if the finger tip is on the left side or the right side of the knuckle
        knuckle_vector = finger_joint_coords[1] - finger_joint_coords[0]
        tip_vector = finger_joint_coords[-1] - finger_joint_coords[0]
        knuckle_vector_slope = knuckle_vector[1] / knuckle_vector[0]
        tip_vector_slope = tip_vector[1] / tip_vector[0]

        if knuckle_vector_slope > tip_vector_slope:
            return angle
        else:
            return -1 * angle


class AllegroKDLControl(AllegroKinematicControl):
    def __init__(self,  bounded_angles = True):
        super().__init__(bounded_angles)
        self.solver = AllegroKDL()

    def calculate_desired_angles(
        self, 
        finger_type, 
        transformed_coords, 
        moving_avg_arr, 
        curr_angles
    ):
        curr_finger_angles = self._get_curr_finger_angles(curr_angles, finger_type)        
        avg_finger_coords = moving_average(transformed_coords, moving_avg_arr, self.time_steps)    
        calc_finger_angles = self.solver.finger_inverse_kinematics(finger_type, avg_finger_coords, curr_finger_angles)
        

        desired_angles = np.array(copy(curr_angles))

        # Applying angular bounds
        if self.bounded_angles is True:
            del_finger_angles = calc_finger_angles - curr_finger_angles
            clipped_del_finger_angles = np.clip(del_finger_angles, - self.bounds[finger_type], self.bounds[finger_type])
            for idx in range(self.hand_configs['joints_per_finger']):
                desired_angles[self.finger_configs['links_info'][finger_type]['offset'] + idx] += clipped_del_finger_angles[idx]
        else:
            for idx in range(self.hand_configs['joints_per_finger']):
                desired_angles[self.finger_configs['links_info'][finger_type]['offset'] + idx] = calc_finger_angles[idx]


        return desired_angles 

    def finger_1D_motion(
        self, 
        finger_type, 
        hand_y_val, 
        robot_x_val, 
        robot_y_val, 
        y_hand_bound, 
        z_robot_bound, 
        moving_avg_arr, 
        curr_angles
    ):
        '''
        For 1D control along the Z direction - used in index and middle fingers at a fixed depth and fixed y
        '''
        x_robot_coord = robot_x_val
        y_robot_coord = robot_y_val
        z_robot_coord = linear_transform(hand_y_val, y_hand_bound, z_robot_bound)
        transformed_coords = [x_robot_coord, y_robot_coord, z_robot_coord]

        desired_angles = self.calculate_desired_angles(finger_type, transformed_coords, moving_avg_arr, curr_angles)
        return desired_angles

    def finger_2D_motion(
        self, 
        finger_type, 
        hand_x_val, 
        hand_y_val, 
        robot_x_val, 
        x_hand_bound, 
        y_hand_bound, 
        y_robot_bound, 
        z_robot_bound, 
        moving_avg_arr, 
        curr_angles
    ):
        '''
        For 2D control in Y and Z directions - used in ring finger at a fixed depth
        '''
        x_robot_coord = robot_x_val
        y_robot_coord = linear_transform(hand_x_val, x_hand_bound, y_robot_bound)
        z_robot_coord = linear_transform(hand_y_val, y_hand_bound, z_robot_bound)
        transformed_coords = [x_robot_coord, y_robot_coord, z_robot_coord]

        desired_angles = self.calculate_desired_angles(finger_type, transformed_coords, moving_avg_arr, curr_angles)
        return desired_angles

    def finger_2D_depth_motion(
        self, 
        finger_type, 
        hand_y_val, 
        robot_y_val, 
        hand_z_val, 
        y_hand_bound, 
        z_hand_bound, 
        x_robot_bound, 
        z_robot_bound, 
        moving_avg_arr, 
        curr_angles
    ):
        '''
        For 2D control in X and Z directions - used in index and middle fingers at a varied depth
        '''
        x_robot_coord = linear_transform(hand_z_val, z_hand_bound, x_robot_bound)
        y_robot_coord = robot_y_val
        z_robot_coord = linear_transform(hand_y_val, y_hand_bound, z_robot_bound)
        transformed_coords = [x_robot_coord, y_robot_coord, z_robot_coord]

        desired_angles = self.calculate_desired_angles(finger_type, transformed_coords, moving_avg_arr, curr_angles)
        return desired_angles

    def finger_3D_motion(
        self, 
        finger_type, 
        hand_x_val, 
        hand_y_val, 
        hand_z_val, 
        x_hand_bound, 
        y_hand_bound, 
        z_hand_bound, 
        x_robot_bound, 
        y_robot_bound, 
        z_robot_bound, 
        moving_avg_arr, 
        curr_angles
    ):
        '''
        For 3D control in all directions - used in ring finger at a varied depth
        '''
        x_robot_coord = linear_transform(hand_z_val, z_hand_bound, x_robot_bound)
        y_robot_coord = linear_transform(hand_x_val, x_hand_bound, y_robot_bound)
        z_robot_coord = linear_transform(hand_y_val, y_hand_bound, z_robot_bound)
        transformed_coords = [x_robot_coord, y_robot_coord, z_robot_coord]

        desired_angles = self.calculate_desired_angles(finger_type, transformed_coords, moving_avg_arr, curr_angles)
        return desired_angles

    def thumb_motion_2D(
        self, 
        hand_coordinates, 
        xy_hand_bounds, 
        yz_robot_bounds, 
        robot_x_val, 
        moving_avg_arr, 
        curr_angles
    ):
        '''
        For 2D control in Y and Z directions - human bounds are mapped to robot bounds
        '''
        y_robot_coord, z_robot_coord = persperctive_transform(
            (hand_coordinates[0], hand_coordinates[1]), 
            xy_hand_bounds, 
            yz_robot_bounds
        )

        x_robot_coord = robot_x_val        
        transformed_coords = [x_robot_coord, y_robot_coord, z_robot_coord]
        
        desired_angles = self.calculate_desired_angles('thumb', transformed_coords, moving_avg_arr, curr_angles)
        return desired_angles

    def thumb_motion_3D(
        self, 
        hand_coordinates, 
        xy_hand_bounds, 
        yz_robot_bounds, 
        z_hand_bound, 
        x_robot_bound, 
        moving_avg_arr, 
        curr_angles
    ):
        '''
        For 3D control in all directions - human bounds are mapped to robot bounds with varied depth
        '''
        y_robot_coord, z_robot_coord = persperctive_transform(
            (hand_coordinates[0], hand_coordinates[1]), 
            xy_hand_bounds, 
            yz_robot_bounds
        )

        x_robot_coord = linear_transform(hand_coordinates[2], z_hand_bound, x_robot_bound)
        transformed_coords = [x_robot_coord, y_robot_coord, z_robot_coord]
        desired_angles = self.calculate_desired_angles('thumb', transformed_coords, moving_avg_arr, curr_angles)

        return desired_angles