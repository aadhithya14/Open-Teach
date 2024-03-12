import hydra
import numpy as np 
from .plotters.xela_plotter import *
from get_xela_values import XelaCurvedSensorControl # Python path is set to that directory
from openteach.components import Component

class XelaVisualizer(Component):
    def __init__(
        self,
        sensor,
        display_plot
    ):
        self.notify_component_start('Xela Visualizer starting')

        # Initialize the plotter and the sensor controller
        self.plotter = XelaCurvedPlotter(display_plot)
        self.sensor = sensor
        # self.display_plot = display_plot

    def stream(self):
        while True:

            #xela_palm_sensor_values,xela_fingertip_sensor_values,xela_finger_sensor_values, timestamp  = self.sensor.get_sensor_state()
            sensor_state= self.sensor.get_sensor_state()
            palm_sensor_values=sensor_state['palm_sensor_values']
            fingertip_sensor_values=sensor_state['fingertip_sensor_values']
            finger_sensor_values=sensor_state['finger_sensor_values']
            #print(xela_finger_sensor_values)
            #print(palm_sensor_values)
            if  palm_sensor_values is not None or fingertip_sensor_values is not None or finger_sensor_values is not None:

                # Get the xela sensor values
                
                self.plotter.draw(palm_sensor_values, fingertip_sensor_values, finger_sensor_values)
            
        print('Stopping the XELA visualizer')

class XelaCurvedVisualizer(Component):
    def __init__(
        self,
        sensor,
        display_plot
    ):
        self.notify_component_start('Xela Visualizer starting')
        # Initialize the plotter and the sensor controller
        self.plotter = XelaCurvedPlotter(display_plot)
        self.sensor = sensor

    def stream(self):
        while True:
            try: 
                # Get the xela sensor values
                xela_state = self.sensor.get_sensor_state()

                if not xela_state is None:
                    palm_sensor_values = xela_state['palm_sensor_values']
                    fingertip_sensor_values = xela_state['fingertip_sensor_values']
                    finger_sensor_values = xela_state['finger_sensor_values']

                self.plotter.draw(palm_sensor_values, fingertip_sensor_values, finger_sensor_values)

            except:
                break 
            
        print('Stopping the XELA visualizer')