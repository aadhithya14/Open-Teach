import os
import cv2
import h5py
import numpy as np
from abc import ABC, abstractmethod
from openteach.utils.files import get_pickle_data
from openteach.constants import *

class Sampler(ABC):
    def __init__(self, data_path, cam_idxs, data_type, min_action_distance):
        self.data_path = data_path
        self.cam_idxs = cam_idxs
        self.data_type = data_type
        self._min_action_distance = min_action_distance

        # Obtaining all the timestamp arrays 
        self._get_image_frame_timestamps()
        self._get_robot_data()
        self._get_sensor_data()

    # Getting data
    def _get_sensor_data(self):
        pass

    @abstractmethod
    def _get_robot_data(self):
        pass

    # For obtaining the timestamps
    def _get_rgb_frame_idxs(self, metadata_path):
        return np.array(get_pickle_data(metadata_path)['timestamps'], dtype = np.float64)

    def _get_hdf5_data(self, hdf5_path, required_data, dtype):
        with h5py.File(hdf5_path, 'r') as depth_file:
            data = np.array(depth_file[required_data], dtype = dtype)
        return data

    def _get_hdf5_timestamps(self, hdf5_file_path):
        #with h5py.File(hdf5_file_path, '

        return self._get_hdf5_data(hdf5_file_path, 'timestamp', np.float64)

    def _get_image_frame_timestamps(self):
        self.image_frame_timestamps = dict()
        # Obtaining all the RGB frames
        if self.data_type == 'rgb' or self.data_type == 'all':
            print('Obtaining all the RGB image timestamps')
            rgb_frame_timestamps = []
            for idx in self.cam_idxs:
                rgb_frame_timestamps.append(self._get_rgb_frame_idxs(
                    metadata_path = os.path.join(self.data_path, 'cam_{}_rgb_video.metadata'.format(idx))
                ) / 1e3)
            self.image_frame_timestamps['rgb'] = rgb_frame_timestamps

        # Obtaining all the Depth frames
        if self.data_type == 'depth' or self.data_type == 'all':
            print('Obtaining all the Depth image timestamps')
            depth_frame_timestamps = []
            for idx in self.cam_idxs:
                depth_frame_timestamps.append(self._get_hdf5_timestamps(
                    hdf5_file_path = os.path.join(self.data_path, 'cam_{}_depth.h5'.format(idx))
                ) / 1e3)
            self.image_frame_timestamps['depth'] = depth_frame_timestamps


    # To pick the corresponding timestamps
    def _get_matching_timestamp(self, timestamp_array, reference_timestamp):
        differences = timestamp_array - reference_timestamp
        idx = np.argmax(differences >= 0)
        if idx >= 0 and differences[idx] >= 0:
            return idx
        else:
            return None

    def _get_latest_timestamp(self, timestamp_arrays, latest_timestamp = None):
        for frame_timestamps in timestamp_arrays:
            if latest_timestamp is None:
                latest_timestamp = frame_timestamps[0]
                continue

            if latest_timestamp < frame_timestamps[0]:
                latest_timestamp = frame_timestamps[0]

        return latest_timestamp

    def _get_image_starting_idxs(self, latest_timestamp):
        starting_idxs = dict()
        for data_type in self.image_frame_timestamps.keys():
            starting_idxs[data_type] = []
            for timestamp_array in self.image_frame_timestamps[data_type]:
                starting_idx = self._get_matching_timestamp(timestamp_array, latest_timestamp)
                starting_idxs[data_type].append([starting_idx])

        return starting_idxs

    def _get_starting_idxs(self):
        # Obtaining the latest timestamp between the image frames
        latest_timestamp = None
        for data_type in self.image_frame_timestamps.keys():
            latest_timestamp = self._get_latest_timestamp(
                timestamp_arrays = self.image_frame_timestamps[data_type], 
                latest_timestamp = latest_timestamp
            )

        # Obtaining the latest timestamp with the robot data
        # if latest_timestamp < self._kinova_timestamps[0]:
        #     latest_timestamp = self._kinova_timestamps[0]

        # Starting indices for the camera frames
        self._chosen_frame_idxs = self._get_image_starting_idxs(latest_timestamp)

        # Allegro starting index
        self._chosen_allegro_idxs = [self._get_matching_timestamp(self._allegro_timestamps, latest_timestamp)]
        # self._chosen_kinova_idxs = [self._get_matching_timestamp(self._kinova_timestamps, latest_timestamp)]
        

    # To sample frames from a fixed timestamp
    def _sample_images(self, instance_timestamp):
        new_image_frame_idxs = dict()
        for data_type in self.image_frame_timestamps.keys():
            new_image_frame_idxs[data_type] = []

            for cam_idx, timestamp_array in enumerate(self.image_frame_timestamps[data_type]):
                latest_used_idx = self._chosen_frame_idxs[data_type][cam_idx][-1]
                if latest_used_idx + 1 > len(timestamp_array): # If no more image frames left
                        return False

                clipped_image_timestamp_array = timestamp_array[latest_used_idx + 1:]
                if self._get_matching_timestamp(clipped_image_timestamp_array, instance_timestamp) is None:
                    return False
                else:
                    image_idx = self._get_matching_timestamp(clipped_image_timestamp_array, instance_timestamp) + latest_used_idx + 1
               
                
                if image_idx is None: # If no matches left
                    return False

                new_image_frame_idxs[data_type].append(image_idx)                        
        
        for data_type in self._chosen_frame_idxs.keys():
            for cam_idx in range(len(self._chosen_frame_idxs[data_type])):
                self._chosen_frame_idxs[data_type][cam_idx].append(new_image_frame_idxs[data_type][cam_idx])

        return True

    @property
    def sampled_rgb_frame_idxs(self):
        if self.data_type == 'rgb' or self.data_type == 'all':
            print("Entering the Correct Model")
            print(self._chosen_frame_idxs)
            return self._chosen_frame_idxs
        else:
            return None

    @property
    def sampled_depth_frame_idxs(self):
        if self.data_type == 'depth' or self.data_type == 'all':
            return self._chosen_frame_idxs['depth']
        else:
            return None

    def save_sampled_rgb_frames(self, cam_idx, store_path):
        if self.data_type == 'rgb' or self.data_type == 'all':
            capture = cv2.VideoCapture(
                os.path.join(self.data_path, 'cam_{}_rgb_video.avi'.format(cam_idx))
            )
            writer = cv2.VideoWriter(
                store_path, 
                cv2.VideoWriter_fourcc(*'XVID'), 
                SAMPLE_WRITER_FPS, 
                IMAGE_RECORD_RESOLUTION
            )

            counter, num_frames_recorded = 0, 0
            print('Writing the frames.')
            while capture.isOpened():
                ret, frame = capture.read()
                if ret == True: 
                    if counter in self._chosen_frame_idxs['rgb'][cam_idx]:
                        writer.write(frame)
                        num_frames_recorded += 1
                    counter += 1
                else: 
                    break
                
 
            # When everything done, release the video capture object
            print('Storing {} frames'.format(num_frames_recorded))
            print('Saving video in {}'.format(store_path))
            writer.release()
            capture.release()
        
    def get_sampled_depth_frames(self, cam_idx):
        if self.data_type == 'depth' or self.data_type == 'all':
            depth_data = self._get_hdf5_data(
                hdf5_path = os.path.join(self.data_path, 'cam_{}_depth.h5'.format(cam_idx)),
                required_data = 'depth_images',
                dtype = np.uint16
            )
            return depth_data[self._chosen_frame_idxs['depth'][cam_idx]]

    @abstractmethod
    def sample_data(self):
        pass