import numpy as np
import pyrealsense2 as rs
from openteach.components import Component
from openteach.utils.images import rotate_image, rescale_image
from openteach.utils.timer import FrequencyTimer
from openteach.utils.network import ZMQCameraPublisher, ZMQCompressedImageTransmitter
from openteach.constants import *
import subprocess as sp
import cv2
import time
import multiprocessing as mp

class FishEyeCamera(Component):
    def __init__(self,cam_index,stream_configs, stream_oculus = False):
        # Disabling scientific notations
        np.set_printoptions(suppress=True)
        self.cam_id = cam_index
        #self.output_file = output_file
        self._stream_configs = stream_configs
        self._stream_oculus = stream_oculus
       

        # Different publishers to avoid overload
        self.rgb_publisher = ZMQCameraPublisher(
            host = stream_configs['host'],
            port = stream_configs['port']#(0 if self.cam_id == 24 else self.cam_id)
        )

        
        if self._stream_oculus:
            self.rgb_viz_publisher = ZMQCompressedImageTransmitter(
                host = stream_configs['host'],
                port = stream_configs['set_port_offset'] + VIZ_PORT_OFFSET # Oculus only reads from a set port - this shouldn't change with the camera ID
                # port= 10005 + VIZ_PORT_OFFSET
            )
            print('STREAMING HERE IN FISH EYE CAM: {}'.format(cam_index))

        self.timer = FrequencyTimer(CAM_FPS) # 30 fps

        # Starting the Fisheye pipeline
        self._start_fisheye()

    def _start_fisheye(self):
        
        print("Cam Id is ", self.cam_id)
        self.cap = cv2.VideoCapture(self.cam_id)
        # self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
       
       
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 680)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
       
        print("Cap is ", self.cap.isOpened())
        # Check if the camera is opened successfully, wait until it is
        while not self.cap.isOpened():
            cap=self.cap.isOpened()


    def get_rgb_depth_images(self):
        frame = None
        while frame is None:
            ret, frame = self.cap.read()
        timestamp = time.time()
        return frame, timestamp
    
    def stream(self):
        # Starting the fisheye stream
        self.notify_component_start('FishEye')
        print(f"Started the pipeline for FishEye camera: {self.cam_id}!")
        print("Starting stream on {}:{}...\n".format(self._stream_configs['host'], self._stream_configs['port']))
        
        if self._stream_oculus:
            print('Starting oculus stream on port: {}\n'.format(self._stream_configs['port'] + VIZ_PORT_OFFSET))

        while True:
            try:
                self.timer.start_loop()
                color_image,timestamp = self.get_rgb_depth_images()

                # Publishing the rgb images
                self.rgb_publisher.pub_rgb_image(color_image, timestamp)
                if self._stream_oculus:
                    self.rgb_viz_publisher.send_image(rescale_image(color_image, 2)) # 640 * 360

                self.timer.end_loop()
                if cv2.waitKey(1) == ord('q'):
                    break
            except KeyboardInterrupt:
                break
        self.cap.release()
        print('Shutting down pipeline for camera {}.'.format(self.cam_id))
        self.rgb_publisher.stop()
        if self._stream_oculus:
            self.rgb_viz_publisher.stop()
        