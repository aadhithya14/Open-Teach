import os
import time
import h5py
import hydra
import numpy as np
from .recorder import Recorder
from openteach.utils.timer import FrequencyTimer
from openteach.constants import *

# To record robot information
class RobotInformationRecord(Recorder):
    def __init__(
        self,
        robot_configs,
        recorder_function_key,
        storage_path
    ):

        # Data function and attributes
        self.robot = hydra.utils.instantiate(robot_configs, record_type=recorder_function_key)
        self.keypoint_function = self.robot.recorder_functions[recorder_function_key]

        # Timer
        self.timer = FrequencyTimer(self.robot.data_frequency)

        # Storage path for file
        self._filename = '{}_{}'.format(self.robot.name, recorder_function_key)
        self.notify_component_start('{}'.format(self._filename))
        self._recorder_file_name = os.path.join(storage_path, self._filename + '.h5')

        # Initializing data containers
        self.robot_information = dict()

    def stream(self):
        # Checking if the keypoint port is active
        print('Checking if the keypoint port is active...')
        while self.keypoint_function() is None:
            continue
        print('Starting to record keypoints to store in {}.'.format(self._recorder_file_name))

        self.num_datapoints = 0
        self.record_start_time = time.time()

        while True:
            self.timer.start_loop()
            try:
                datapoint = self.keypoint_function()
                #print(datapoint.keys())
                for attribute in datapoint.keys():
                    if attribute not in self.robot_information.keys():
                        self.robot_information[attribute] = [datapoint[attribute]]
                        continue
                    
                    self.robot_information[attribute].append(datapoint[attribute])

                self.num_datapoints += 1
                self.timer.end_loop()
            except KeyboardInterrupt:
                self.record_end_time = time.time()
                break

        # Displaying statistics
        self._display_statistics(self.num_datapoints)
        
        # Saving the metadata
        self._add_metadata(self.num_datapoints)

        # Writing to dataset
        print('Compressing keypoint data...')
        with h5py.File(self._recorder_file_name, "w") as file:
            # Main data
            for key in self.robot_information.keys():
                if key != 'timestamp':
                    self.robot_information[key] = np.array(self.robot_information[key], dtype = np.float32)
                    file.create_dataset(key +'s', data = self.robot_information[key], compression="gzip", compression_opts = 6)
                else:
                    self.robot_information['timestamp'] = np.array(self.robot_information['timestamp'], dtype = np.float64)
                    file.create_dataset('timestamps', data = self.robot_information['timestamp'], compression="gzip", compression_opts = 6)

            # Other metadata
            file.update(self.metadata)
        print('Saved keypoint data in {}.'.format(self._recorder_file_name))