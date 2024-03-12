import os
import time
import h5py
import hydra
import numpy as np
from .recorder import Recorder
from openteach.utils.timer import FrequencyTimer
from openteach.constants import *

class XelaSensorRecorder(Recorder):
    def __init__(
        self,
        controller_configs,
        storage_path
    ):

        # Initialize the sensor controllers
        self.sensor = hydra.utils.instantiate(controller_configs)

        # Timer 
        self.timer = FrequencyTimer(XELA_FPS)

        # Create the storage path for file
        self._filename = 'touch_sensor_values'
        self.notify_component_start('{}'.format(self._filename))
        self._recorder_file_name = os.path.join(storage_path, self._filename + '.h5')

        # Initializing the data containers 
        self.sensor_information = dict()

    def stream(self):
        print('Checking if XELA sensors are active...')
        while self.sensor.get_sensor_state() is None:
            continue

        print('Starting to record xela sensor values in {}'.format(self._recorder_file_name))

        self.num_datapoints = 0 
        self.record_start_time = time.time() 

        while True: 
            try:
                self.timer.start_loop() 
                sensor_state = self.sensor.get_sensor_state() # Has sensor_values and timestamps
                for attribute in sensor_state.keys():
                    if attribute not in self.sensor_information.keys():
                        self.sensor_information[attribute] = [sensor_state[attribute]]
                        continue 

                    self.sensor_information[attribute].append(sensor_state[attribute])

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
        print('Compressing XELA sensor data...')
        with h5py.File(self._recorder_file_name, 'w') as file:
            for key in self.sensor_information.keys():
                if key != 'timestamp':
                    self.sensor_information[key] = np.array(self.sensor_information[key], dtype=np.float32)
                    file.create_dataset(key, data = self.sensor_information[key], compression="gzip", compression_opts = 6)
                else:
                    self.sensor_information['timestamp'] = np.array(self.sensor_information['timestamp'], dtype = np.float64)
                    file.create_dataset('timestamps', data = self.sensor_information['timestamp'], compression="gzip", compression_opts = 6)

            # Other metadata 
            file.update(self.metadata)
        print('Saved XELA sensor data in {}'.format(self._recorder_file_name))
