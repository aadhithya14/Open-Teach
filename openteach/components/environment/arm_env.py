from abc import ABC, abstractmethod
from openteach.components import Component
from openteach.utils.network import ZMQCameraPublisher, ZMQCompressedImageTransmitter
from openteach.constants import *
from openteach.utils.timer import FrequencyTimer
from openteach.utils.network import ZMQKeypointSubscriber, ZMQKeypointPublisher
from openteach.utils.images import rotate_image, rescale_image
import numpy as np

class Arm_Env(Component, ABC):
       
    @property
    @abstractmethod
    def timer(self):
        return self._timer

    
    def stream(self):
        self.notify_component_start('{} environment'.format(self.name))
        print("Start controlling the Simulation Arm using the Oculus Headset.\n")
        while True:
            try:
                self.timer.start_loop() 
                #Get RGB Images and Depth Images
                color_image,depth_image,timestamp=self.get_rgb_depth_images()
                #Publishes RGB images
                self.rgb_publisher.pub_rgb_image(color_image, timestamp)
                self.timestamp_publisher.pub_keypoints(timestamp,'timestamps')
                #Set this to True to view the sim rendering inside the Oculus.       
                if self._stream_oculus:
                        self.rgb_viz_publisher.send_image(rescale_image(color_image, 2)) # 640 * 360

                # Publishing the depth images
                self.depth_publisher.pub_depth_image(depth_image, timestamp)
               
                #Gets the endeffector position       
                position=self.get_endeff_position()
                #Publishes the endeffector position so that Operator can use.
                self.endeff_publisher.pub_keypoints(position,'endeff_coords')

                #Takes Action
                self.take_action()
                self.timer.end_loop()  

            except KeyboardInterrupt:
                break
        self.rgb_publisher.stop()
        self.depth_publisher.stop()
        self.endeff_publisher.stop()
        
       
        print('Stopping the environment!')