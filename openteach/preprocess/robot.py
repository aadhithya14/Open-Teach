import os 
import cv2
import numpy as np
import h5py
import pickle 
import shutil

from tqdm import tqdm

import openteach.preprocess.utils.transform_utils as transform_utils # Taken from deoxys
from .prep_module import PreprocessorModule
# from openteach.robot.allegro.allegro_kdl import AllegroKDL

# from deoxys.utils import transform_utils
from openteach.preprocess.utils import MODALITY_TYPES

# preprocessor module for both robot arm and the hand

def get_closest_id(curr_id, desired_timestamp, all_timestamps):
    for i in range(curr_id, len(all_timestamps)):
        if all_timestamps[i] >= desired_timestamp:
            return i # Find the first timestamp that is after that
    
    return i

def get_arm_distance(current_cart, target_cart):
    # current_pos, current_quat = current_cart[:3], current_cart[3:]
    # target_pos, target_quat = target_cart[:3], target_cart[3:]
    current_pos, current_euler = current_cart[:3], current_cart[3:]
    target_pos, target_euler = target_cart[:3], target_cart[3:]
    current_quat = transform_utils.euler2quat(current_euler)
    target_quat = transform_utils.euler2quat(target_euler)

    pos_diff = np.linalg.norm(target_pos - current_pos)

    quat_diff = transform_utils.quat_multiply(current_quat, transform_utils.quat_inverse(target_quat))
    angle_diff = 180*np.linalg.norm(transform_utils.quat2axisangle(quat_diff))/np.pi

    return pos_diff, angle_diff

# def find_next_robot_state_timestamp(kdl_solver, allegro_positions, allegro_timestamps, allegro_id, kinova_positions, kinova_id, kinova_timestamps, threshold_step_size=0.015):
#     # Find the timestamp where allegro and kinova difference in total is larger than the given threshold
#     old_allegro_pos = allegro_positions[allegro_id]
#     old_allegro_fingertip_pos = kdl_solver.get_fingertip_coords(old_allegro_pos)
#     old_kinova_pos = kinova_positions[kinova_id]
#     for i in range(allegro_id, len(allegro_positions)):
#         curr_allegro_fingertip_pos = kdl_solver.get_fingertip_coords(allegro_positions[i])
#         curr_allegro_timestamp = allegro_timestamps[i]
#         kinova_id = get_closest_id(kinova_id, curr_allegro_timestamp, kinova_timestamps)
#         curr_kinova_pos = kinova_positions[kinova_id]
#         step_size = np.linalg.norm(old_allegro_fingertip_pos - curr_allegro_fingertip_pos) + np.linalg.norm(old_kinova_pos - curr_kinova_pos)
#         if step_size > threshold_step_size:
#             return curr_allegro_timestamp

#         if kinova_id >= len(kinova_positions):
#             return None
            
#     return None

class RobotPreprocessor(PreprocessorModule):
    def __init__(self,
                 subsample_separately,
                 robot_names, robot_thresholds, apply_subsample=True):
        super().__init__(
            subsample_separately=subsample_separately,
            robot_names=robot_names)

        # self.robot_thresholds = np.array(robot_thresholds)

        self.apply_subsample = apply_subsample
        self.robot_thresholds = robot_thresholds
        if not subsample_separately:
            robot_threshold_key = min(robot_thresholds)
            self.robot_threshold = robot_thresholds[robot_threshold_key]

        # self._get_solvers()

        # Get the file names
        self.robot_files = dict(
            load = dict(),
            dump = dict()
        )
        self.robot_types = []
        for robot_name in robot_names:
            robot_type = MODALITY_TYPES[robot_name]
            self.robot_types.append(robot_type)
            if robot_type == 'arm':
                self.robot_files['load']['arm'] = f'{robot_name}_cartesian_states.h5'
                self.robot_files['dump']['arm'] = f'{robot_name}_indices.pkl'
            # elif robot_type == 'hand':
            #     self.robot_files['load']['hand'] = [f'{robot_name}_joint_states.h5', f'{robot_name}_commanded_joint_states.h5'] 
            #     self.robot_files['dump']['hand'] = [f'{robot_name}_indices.pkl', f'{robot_name}_action_indices.pkl']

        # if 'hand' in self.robot_types:
        #     self.current_hand_id = 0
        #     self.current_hand_action_id = 0
        if 'arm' in self.robot_types:
            self.current_arm_id = 0

        self.indices = dict(
            arm = [],
            hand = [[], []]
        )

    def __repr__(self):
        return f'robot_preprocessor for {self.robot_names}'

    # def _get_solvers(self):
    #     if 'allegro' in self.robot_names:
    #         self.hand_solver = AllegroKDL()

    def update_root(self, demo_id, root):
        self.root = root
        self.demo_id = demo_id
        self.load_data()
        print('Updated the root of PreprocessorModule ({}) for demo_id: {} - root: {}'.format(
            self, demo_id, root
        ))
        self.reset_current_id()
        self.indices = dict(
            arm = [],
            hand = [[], []]
        )

    def update_next_id(self, desired_timestamp):
        for robot_type in self.robot_types:
            if robot_type == 'arm':
                next_arm_id = get_closest_id(
                    curr_id = self.current_arm_id,
                    desired_timestamp=desired_timestamp,
                    all_timestamps=self.data['arm']['timestamps']
                ) 
                self.current_arm_id = next_arm_id
            # if robot_type == 'hand':
            #     next_hand_id = get_closest_id(
            #         curr_id = self.current_hand_id,
            #         desired_timestamp=desired_timestamp,
            #         all_timestamps = self.data['hand']['timestamps']
            #     )
            #     self.current_hand_id = next_hand_id 
            #     next_hand_action_id = get_closest_id(
            #         curr_id = self.current_hand_action_id,
            #         desired_timestamp=desired_timestamp,
            #         all_timestamps=self.data['hand']['action_timestamps']
            #     )
            #     self.current_hand_action_id = next_hand_action_id

    @property
    def current_timestamp(self):
        ts_arr = []
        if 'arm' in self.robot_types:
            ts_arr.append(
                self.data['arm']['timestamps'][self.current_arm_id]
            )
        # if 'hand' in self.robot_types:
        #     ts_arr.append(
        #         self.data['hand']['timestamps'][self.current_hand_id]
        #     )
        #     ts_arr.append(
        #         self.data['hand']['action_timestamps'][self.current_hand_action_id]
        #     )
        
        return min(ts_arr)

    def load_data(self):
        self.data = dict()
        for robot_type in self.robot_types:
            self.data[robot_type] = dict()
            # if robot_type == 'hand':
            #     with h5py.File(
            #         os.path.join(self.root, self.robot_files['load'][robot_type][0]), 'r'
            #     ) as f:
            #         self.data[robot_type]['positions'] = f['positions'][()]
            #         self.data[robot_type]['timestamps'] = f['timestamps'][()]

            #     with h5py.File(
            #         os.path.join(self.root, self.robot_files['load'][robot_type][1]), 'r'
            #     ) as f:
            #         self.data[robot_type]['action_timestamps'] = f['timestamps'][()]
            
            if robot_type == 'arm':
                file_path = os.path.join(self.root, self.robot_files['load'][robot_type])
                with h5py.File(file_path, 'r') as f:
                    # self.data[robot_type]['positions'] = np.concatenate([f['positions'][()], f['orientations'][()]], axis=1)
                    self.data[robot_type]['positions'] = np.array(f['cartesian_positions'][()], dtype=np.float32)
                    self.data[robot_type]['timestamps'] = f['timestamps'][()]

            # else:
            
            #     file_path = os.path.join(self.root, self.robot_files['load'][robot_type])
            #     with h5py.File(file_path, 'r') as f:
            #         self.data[robot_type]['positions'] = f['positions'][()]
            #         self.data[robot_type]['timestamps'] = f['timestamps'][()]


    # def _find_next_hand_id(self, threshold_step_size, limit_timestamp=None):
    #     old_pos = self.data['hand']['positions'][self.current_hand_id]
    #     old_fingertip_pos = self.hand_solver.get_fingertip_coords(old_pos)
    #     for i in range(self.current_hand_id+1, len(self.data['hand']['positions'])):
    #         curr_fingertip_pos = self.hand_solver.get_fingertip_coords(self.data['hand']['positions'][i])
    #         step_size = np.linalg.norm(old_fingertip_pos - curr_fingertip_pos)
    #         if (not limit_timestamp is None and self.data['hand']['timestamps'][i] > limit_timestamp) or \
    #             step_size > threshold_step_size:
    #             return i
        
    #     return i

    def _find_next_arm_id(self, threshold_step_size, limit_timestamp=None):
        old_pos = self.data['arm']['positions'][self.current_arm_id]
        for i in range(self.current_arm_id+1, len(self.data['arm']['positions'])):
            curr_pos = self.data['arm']['positions'][i]
            curr_pos_dist, curr_angle_diff = get_arm_distance(old_pos, curr_pos)
            if (not limit_timestamp is None and self.data['arm']['timestamps'][i] > limit_timestamp) or \
                curr_pos_dist > threshold_step_size or (curr_angle_diff > 10 and curr_angle_diff < 350):
            # if curr_pos_dist > threshold_step_size or curr_angle_diff > 10: # If the angle is larger than 10 or the distance is more than the threshold step size
                return i 
            
        return i
    
    # If it has reached that timestamp
    def _find_next_robot_type_id(self, threshold_step_size, robot_type, limit_timestamp=None):
        if robot_type == 'arm':
            return self._find_next_arm_id(threshold_step_size, limit_timestamp)
        # elif robot_type == 'hand':
        #     return self._find_next_hand_id(threshold_step_size, limit_timestamp)
            
    
    # def _find_next_robot_state_timestamp(self, threshold_step_size):
    #     old_hand_pos = self.data['hand']['positions'][self.current_hand_id]
    #     old_fingertip_pos = self.hand_solver.get_fingertip_coords(old_hand_pos)
    #     old_arm_pos = self.data['arm']['positions'][self.current_arm_id]

    #     for i in range(self.current_hand_id, len(self.data['hand']['positions'])):
    #         # Get the current hand position
    #         curr_fingertip_pos = self.hand_solver.get_fingertip_coords(self.data['hand']['positions'][i])
    #         curr_hand_ts = self.data['hand']['timestamps'][i]

    #         # Get the corresponding arm position
    #         curr_arm_id = get_closest_id(
    #             curr_id = self.current_arm_id,
    #             desired_timestamp = curr_hand_ts,
    #             all_timestamps = self.data['arm']['timestamps']
    #         )
    #         if curr_arm_id >= len(self.data['arm']['positions']):
    #             return None
            
    #         # Calculate the arm distance
    #         curr_arm_pos = self.data['arm']['positions'][curr_arm_id]
    #         curr_pos_dist, curr_angle_diff = get_arm_distance(old_arm_pos, curr_arm_pos)

    #         # Calculate the hand distance taken
    #         curr_hand_dist = np.linalg.norm(old_fingertip_pos - curr_fingertip_pos)

    #         if curr_hand_dist + curr_pos_dist > threshold_step_size or curr_angle_diff > 10:
    #             return curr_hand_ts 
            
    #     return None
    
    
    def get_next_timestamp(self):
        # If subsample separately
        if self.subsample_separately:
            # Find the modality with the smallest threshold step first - this is for making the preprocessing faster
            sorted_robot_types = sorted(
                self.robot_thresholds.items(), key=lambda x:x[1])
            ts_arr = []
            for i,robot_type_tuple in enumerate(sorted_robot_types):
                robot_type = robot_type_tuple[0]
                # print('robot_type: {}'.format(robot_type))
                pos_robot_type_id = self._find_next_robot_type_id(
                    threshold_step_size = self.robot_thresholds[robot_type],
                    robot_type = robot_type,
                    limit_timestamp = None if i == 0 else ts_arr[i-1]
                )
                if pos_robot_type_id >= len(self.data[robot_type]['timestamps']): return -1
                ts_arr.append(self.data[robot_type]['timestamps'][pos_robot_type_id])

            # if 'arm' in self.robot_types:
            #     pos_arm_id = self._find_next_arm_id(
            #         threshold_step_size=self.robot_thresholds['arm']
            #     )
            #     if pos_arm_id >= len(self.data['arm']['timestamps']): return -1
            #     ts_arr.append(self.data['arm']['timestamps'][pos_arm_id])
            # if 'hand' in self.robot_types:
            #     pos_hand_id = self._find_next_hand_id(
            #         threshold_step_size=self.robot_thresholds['hand']
            #     )
            #     if pos_hand_id >= len(self.data['hand']['timestamps']): return -1
            #     ts_arr.append(self.data['hand']['timestamps'][pos_hand_id])

            return min(ts_arr)

        # else: 

        #     return self._find_next_robot_state_timestamp(
        #         threshold_step_size = self.robot_threshold
        #     )

    # def dump_fingertips(self):
    #     if self.fingertips and 'hand' in self.robot_types:
    #         print(f'Dumping fingertip positions in root {self.root}')
    #         fingertip_states = dict(
    #             positions = [],
    #             timestamps = []
    #         )
    #         for i in range(len(self.data['hand']['positions'])):
    #             joint_position = self.data['hand']['positions'][i]
    #             timestamp = self.data['hand']['timestamps'][i]
    #             # There are 3 (x,y,z) fingertip positions for each finger
    #             fingertip_position = self.hand_solver.get_fingertip_coords(joint_position)
    #             fingertip_states['positions'].append(fingertip_position)
    #             fingertip_states['timestamps'].append(timestamp)

    #         # Compress the data file
    #         robot_id = self.robot_types.index('hand')
    #         robot_name = self.robot_names[robot_id]
    #         fingertip_state_file = os.path.join(self.root, f'{robot_name}_fingertip_states.h5')
    #         with h5py.File(fingertip_state_file, 'w') as file:
    #             for key in fingertip_states.keys():
    #                 if key == 'timestamps':
    #                     fingertip_states[key] = np.array(fingertip_states[key], dtype=np.float64)
    #                 else:
    #                     fingertip_states[key] = np.array(fingertip_states[key], dtype=np.float32)

    #                 file.create_dataset(key, data = fingertip_states[key], compression='gzip', compression_opts=6)

    #         print(f'Saved fingertip positions in {fingertip_state_file}')

    def update_indices(self):
        self.indices['arm'].append([self.demo_id, self.current_arm_id])
        # self.indices['hand'][0].append([self.demo_id, self.current_hand_id])
        # self.indices['hand'][1].append([self.demo_id, self.current_hand_action_id])

    def dump_data_indices(self):
        if 'arm' in self.robot_types:
            file_path = os.path.join(self.root, self.robot_files['dump']['arm'])
            with open(file_path, 'wb') as f:
                pickle.dump(self.indices['arm'], f)
            print('dumped arm indices {} to: {}'.format(
                self.indices['arm'], self.robot_files['dump']['arm']
            ))
        # if 'hand' in self.robot_types:
        #     for i, robot_file in enumerate(self.robot_files['dump']['hand']):
        #         file_path = os.path.join(self.root, robot_file)
        #         with open(file_path, 'wb') as f: 
        #             pickle.dump(self.indices['hand'][i], f) 

        #         print('dumped hand indices to: {}'.format(
        #             self.robot_files['dump']['hand'][i]
        #         ))

    def is_finished(self):
        for robot_type in self.robot_types:
            if robot_type == 'arm' and self.current_arm_id >= len(self.data[robot_type]['timestamps'])-1:
                return True
            
            # if robot_type == 'hand' and self.current_hand_id >= len(self.data[robot_type]['timestamps'])-1:
            #     return True
            
            # if robot_type == 'hand' and self.current_hand_action_id >= len(self.data[robot_type]['action_timestamps'])-1:
            #     return True
            
        return False 
    
    def reset_current_id(self):
        self.current_arm_id = 0 
        # self.current_hand_id = 0
        # self.current_hand_action_id = 0