import os
import numpy as np
from tqdm import tqdm
from .sampler import Sampler
from openteach.robot.allegro.allegro_kdl import AllegroKDL
from openteach.utils.vectorops import get_distance
from openteach.constants import *
from openteach.robot.kinova import KinovaArm
import h5py

class AllegroSampler(Sampler):
    def __init__(self, data_path, cam_idxs, data_type, min_action_distance):
        super().__init__(data_path, cam_idxs, data_type, min_action_distance)
        self._robot = AllegroKDL()

    def _get_robot_data(self):
        print('Obtaining all the timestamps.')
        robot_state_file = os.path.join(self.data_path, 'allegro_joint_states.h5')
        self._allegro_timestamps = self._get_hdf5_timestamps(
            hdf5_file_path = robot_state_file
        )

        self._allegro_states = self._get_hdf5_data(
            hdf5_path = robot_state_file,
            required_data = 'joint_angles',
            dtype = np.float32
        )

        # self.kinova_state_file = os.path.join(self.data_path, 'kinova_cartesian_states.h5')

        # self._kinova_timestamps = self._get_hdf5_timestamps(
        #     hdf5_file_path = self.kinova_state_file
        # )

        # self.kinova_positions= self._get_hdf5_data(
        #         hdf5_path = self.kinova_state_file,
        #         required_data= 'joint_angles',
        #         dtype = np.float32
        #     )

        # self.kinova_orientations= self._get_hdf5_data(
        #         hdf5_path = self.kinova_state_file,
        #         required_data= 'orientations',
        #         dtype = np.float32
        #     )
        
        # self._kinova_states= np.concatenate([self.kinova_positions, self.kinova_orientations], axis=1)

    def sample_data(self):
        # Obtaining the starting idxs
        print('Obtaining all the starting indices.')
        self._get_starting_idxs()
        self.chosen_timestamps = []
        # Sampling the allegro indices based on the minimum action distance
        previous_idx = self._chosen_allegro_idxs[-1]
        previous_state = self._robot.get_fingertip_coords(self._allegro_states[previous_idx])

        print('Starting sampling process.')
        for current_idx in tqdm(range(
            self._chosen_allegro_idxs[-1], 
            self._allegro_states.shape[0], 
            ALLEGRO_SAMPLE_OFFSET
        )):
            # Check if next state is valid
            if current_idx < self._allegro_states.shape[0]:
                current_state = self._robot.get_fingertip_coords(self._allegro_states[current_idx])
            else:
                break
            
            # Check if action distance is greater than minimum action distance
            if get_distance(current_state, previous_state) > self._min_action_distance:
                # Update the index information
                self._chosen_allegro_idxs.append(current_idx)
                print("Choosen idxs",self._chosen_allegro_idxs)
                
                previous_idx = current_idx
                previous_state = current_state

                # Get the timestamp of the new state
                current_timestamp = self._allegro_timestamps[current_idx]
                # kinova_idx=self._get_matching_timestamp(self._kinova_timestamps, current_timestamp)
                # self._chosen_kinova_idxs.append(kinova_idx)
                #self.chosen_timestamps.append(current_timestamp)
                # Finding corresponding image idxs  
                success = self._sample_images(current_timestamp)

                if not success:
                    break

        print('Sampling finished.')
        print('Extracted number of states: {}'.format(len(self._chosen_allegro_idxs)))

    def kinova_data(self):
        
            self._data=self._get_hdf5_data(self.kinova_state_file,'joint_angles',np.float64)
            self.time_stamps=self._get_hdf5_timestamps(self.kinova_state_file)
            self.positions=[]
            for i in range(len(self.chosen_timestamps)):
                #print(self.chosen_timestamps)
                print(len(self.chosen_timestamps))
                print(len(self.time_stamps))
                idx=list(self.time_stamps).index(self.chosen_timestamps[i])
                print(idx)
                self.positions.append(self._data[idx])
            print(len(self.positions))
            #np.save(self.data_path / "processed_1" / "kinova_states_new.npy",np.array(self.positions))

    @property
    def sampled_allegro_states(self):
        return self._allegro_states[self._chosen_allegro_idxs]
    
    @property
    def sampled_kinova_states(self):
            return self._kinova_states[self._chosen_kinova_idxs]
    
    def sampled_robot_idxs(self):
        return self._chosen_allegro_idxs 



class KinovaSampler_old(Sampler):
    def __init__(self, data_path, cam_idxs, data_type, min_action_distance):
        super().__init__(data_path, cam_idxs, data_type, min_action_distance)
        #sself._robot = AllegroKDL()

    def _get_robot_data(self):
        print('Obtaining all the timestamps.')

        self.kinova_state_file = os.path.join(self.data_path, 'kinova_cartesian_states.h5')

        self._kinova_timestamps = self._get_hdf5_timestamps(
            hdf5_file_path = self.kinova_state_file
        )

        self.kinova_positions= self._get_hdf5_data(
                hdf5_path = self.kinova_state_file,
                required_data= 'joint_angles',
                dtype = np.float32
            )

        self.kinova_orientations= self._get_hdf5_data(
                hdf5_path = self.kinova_state_file,
                required_data= 'orientations',
                dtype = np.float32
            )
        
        self._kinova_states= np.concatenate([self.kinova_positions, self.kinova_orientations], axis=1)

    def kinova_data(self):
        
        self._data=self._get_hdf5_data(self.kinova_state_file,'joint_angles',np.float64)
        self.time_stamps=self._get_hdf5_timestamps(self.kinova_state_file)
        self.positions=[]
        for i in range(len(self.chosen_timestamps)):
            #print(self.chosen_timestamps)
            print(len(self.chosen_timestamps))
            print(len(self.time_stamps))
            idx=list(self.time_stamps).index(self.chosen_timestamps[i])
            print(idx)
            self.positions.append(self._data[idx])
        print(len(self.positions))
        np.save(self.data_path / "processed_1" / "kinova_states_new.npy",np.array(self.positions))
        

    @property
    def sampled_robot_states(self):
        return self._robot_states[self._chosen_robot_idxs]
    
    @property
    def sampled_kinova_states(self):
            return self._kinova_states[self._chosen_kinova_idxs]
   
    


class KinovaSampler(Sampler):
        def __init__(self, data_path, cam_idxs, data_type, min_action_distance):
            super().__init__(data_path, cam_idxs, data_type, min_action_distance)
            self._robot = KinovaArm()

        def _get_robot_data(self):
            print('Obtaining all the timestamps.')
            
            self.kinova_state_file = os.path.join(self.data_path, 'kinova_cartesian_states.h5')
            self._kinova_timestamps = self._get_hdf5_timestamps(
                hdf5_file_path = self.kinova_state_file
            )

            self.kinova_positions= self._get_hdf5_data(
                hdf5_path = self.kinova_state_file,
                required_data= 'joint_angles',
                dtype = np.float32
            )

            self.kinova_orientations= self._get_hdf5_data(
                hdf5_path = self.kinova_state_file,
                required_data= 'orientations',
                dtype = np.float32
            )
        
            self._kinova_states= np.concatenate([self.kinova_positions, self.kinova_orientations], axis=1)

            self.allegro_state_file = os.path.join(self.data_path, 'allegro_joint_states.h5')
            self._allegro_timestamps = self._get_hdf5_timestamps(
                hdf5_file_path = self.kinova_state_file
            )

            self._allegro_states = self._get_hdf5_data(
                    hdf5_path = self.allegro_state_file,
                    required_data = 'joint_angles',
                    dtype = np.float32
        )


        def sample_data(self):
            # Obtaining the starting idxs
            print('Obtaining all the starting indices.')
            self._get_starting_idxs()

            # Sampling the allegro indices based on the minimum action distance
            previous_idx = self._chosen_kinova_idxs[-1]
            previous_state = self._kinova_states[previous_idx]

            print('Starting sampling process.')
            for current_idx in tqdm(range(
                self._chosen_kinova_idxs[-1], 
                self._kinova_states.shape[0], 
                ALLEGRO_SAMPLE_OFFSET
            )):
                # Check if next state is valid
                if current_idx < self._kinova_states.shape[0]:
                    current_state = self._kinova_states[current_idx]
                else:
                    break
                
                # Check if action distance is greater than minimum action distance
                if get_distance(current_state, previous_state) > self._min_action_distance:
                    # Update the index information
                    self._chosen_kinova_idxs.append(current_idx)
                    previous_idx = current_idx
                    previous_state = current_state

                    # Get the timestamp of the new state
                    current_timestamp = self._kinova_timestamps[current_idx]
                    allegro_idx=self._get_matching_timestamp(self._allegro_timestamps, current_timestamp)
                    self._chosen_allegro_idxs.append(allegro_idx)

                    # Finding corresponding image idxs  
                    success = self._sample_images(current_timestamp)

                    if not success:
                        break

            print('Sampling finished.')
            print('Extracted number of states: {}'.format(len(self._chosen_kinova_idxs)))

        def kinova_data(self):
        
            self._data=self._get_hdf5_data(self.allegro_state_file,'joint_angles',np.float64)
            self.time_stamps=self._get_hdf5_timestamps(self.allegro_state_file)
            self.positions=[]
            for i in range(len(self.chosen_timestamps)):
                #print(self.chosen_timestamps)
                print(len(self.chosen_timestamps))
                print(len(self.time_stamps))
                idx=list(self.time_stamps).index(self.chosen_timestamps[i])
                print(idx)
                self.positions.append(self._data[idx])
            print(len(self.positions))
            #np.save(self.data_path / "processed_2" / "allegro_states.npy",np.array(self.positions))


        @property
        def sampled_kinova_states(self):
            return self._kinova_states[self._chosen_kinova_idxs]

        @property
        def sampled_allegro_states(self):
            return self._allegro_states[self._chosen_allegro_idxs]
