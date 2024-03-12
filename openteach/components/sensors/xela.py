import numpy as np 

from copy import deepcopy as copy

from get_xela_values import XelaSensorControl , XelaCurvedSensorControl 
from openteach.components import Component 
from openteach.utils.timer import FrequencyTimer
from openteach.constants import *

class XelaSensors(Component):
    def __init__(self, init_duration):
        self._controller = XelaSensorControl()
        self._bias_values = np.zeros((XELA_NUM_SENSORS, XELA_NUM_TAXELS, 3))
        self._bias_duration = init_duration # Wait 3 seconds for finding the average
        self.timer = FrequencyTimer(XELA_FPS)
        self._set_bias()

    # The bias should be received for the first 2-3 seconds
    # to normalize the data accordingly
    def _set_bias(self):

        self.notify_component_start('XELA sensors - receiving bias')
        loop_index = 0
        while True: 
            try:
                self.timer.start_loop()
                
                xela_state = self._controller.get_sensor_state()
                if xela_state is None:
                    self.timer.end_loop()
                    continue
                
                bias_values = xela_state['sensor_values']
                # bias_values, _ = self.get_sensor_values()
                if bias_values is not None:
                    self._bias_values += bias_values
                    loop_index += 1
                self.timer.end_loop()

                if loop_index >= self._bias_duration * XELA_FPS:
                    self._bias_values /= loop_index
                    # self.debug_component('found bias_values: {}'.format(self._bias_values))
                    break

            except KeyboardInterrupt:
                break 

        # self._bias_values /= loop_index # Take the average

    def get_sensor_state(self):
        xela_state = self._controller.get_sensor_state()
        if xela_state is not None:
            normalized_state = dict(
                sensor_values = xela_state['sensor_values'] - self._bias_values,
                timestamp = xela_state['timestamp']
            )
            return normalized_state
        return xela_state # NOTE: This can be None as well

    def stream(self):
        # # Starting the xela stream
        self.notify_component_start('XELA sensors')
        print(f"Started the XELA sensor pipeline for the hand")

        while True: 
            try:
                self.timer.start_loop()
                
                xela_state = self.get_sensor_state()
                

                self.timer.end_loop()

            except KeyboardInterrupt:
                break 

class XelaCurvedSensors(Component):
    def __init__(self, init_duration):
        self._controller = XelaCurvedSensorControl()
        self._palm_bias_values = np.zeros((XELA_PALM_NUM_SENSORS, XELA_PALM_NUM_TAXELS, 3))
        self._fingertip_bias_values = np.zeros((XELA_FINGERTIP_NUM_SENSORS, XELA_FINGERTIP_NUM_TAXELS, 3))
        self._finger_bias_values = np.zeros((XELA_FINGER_NUM_SENSORS, XELA_FINGER_NUM_TAXELS,3))
        self._bias_duration = init_duration # Wait 3 seconds for finding the average
        self.timer = FrequencyTimer(XELA_FPS)
        self._set_bias()

    # The bias should be received for the first 2-3 seconds
    # to normalize the data accordingly
    def _set_bias(self):

        self.notify_component_start('XELA sensors - receiving bias')
        loop_index = 0
        while True: 
            try:
                self.timer.start_loop()
                
                sensor_state = self._controller.get_sensor_state()
                if not sensor_state is None:
                    curr_sensor_palm_values,curr_sensor_fingertip_values, curr_sensor_finger_values, timestamp = sensor_state
                    if curr_sensor_palm_values is None and curr_sensor_finger_values is None and curr_sensor_fingertip_values is None:
                        self.timer.end_loop()
                        continue
                
                    palm_bias_values= curr_sensor_palm_values
                    finger_bias_values= curr_sensor_finger_values
                    fingertip_bias_values = curr_sensor_fingertip_values
                    if palm_bias_values is not None and finger_bias_values is not None and fingertip_bias_values is not None:
                        self._palm_bias_values += palm_bias_values
                        self._finger_bias_values += finger_bias_values
                        self._fingertip_bias_values += fingertip_bias_values
                        loop_index += 1
                    self.timer.end_loop()

                    if loop_index >= self._bias_duration * XELA_FPS:
                        self._palm_bias_values /= loop_index
                        self._finger_bias_values /= loop_index
                        self._fingertip_bias_values /= loop_index
                        # self.debug_component('found bias_values: {}'.format(self._bias_values))
                        break

            except KeyboardInterrupt:
                break 

        # self._bias_values /= loop_index # Take the average

    def get_sensor_state(self):
        curr_sensor_palm_values,curr_sensor_fingertip_values, curr_sensor_finger_values, timestamp = self._controller.get_sensor_state()
        if curr_sensor_palm_values is not None and curr_sensor_finger_values is not None and curr_sensor_fingertip_values is not None:
            normalized_state = dict(
                palm_sensor_values = curr_sensor_palm_values - self._palm_bias_values,
                fingertip_sensor_values =  curr_sensor_fingertip_values - self._fingertip_bias_values,
                finger_sensor_values = curr_sensor_finger_values -self._finger_bias_values,
                timestamp = timestamp
            )
            return normalized_state
        return normalized_state # NOTE: This can be None as well

    def stream(self):
        # # Starting the xela stream
        self.notify_component_start('XELA sensors')
        print(f"Started the XELA sensor pipeline for the hand")

        while True: 
           # try:
                self.timer.start_loop()
                
                xela_state = self.get_sensor_state()
                
                self.timer.end_loop()

            #except KeyboardInterrupt:
                #break 