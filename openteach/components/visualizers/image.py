import cv2
from openteach.components import Component
from openteach.constants import *
from openteach.utils.images import *
from openteach.utils.network import ZMQCameraSubscriber
from openteach.utils.timer import FrequencyTimer

class RobotImageVisualizer(Component):
    def __init__(self, host, cam_port_offset, cam_id):        
        self.camera_number = cam_id

        self.notify_component_start('camera {} rgb visualizer'.format(cam_id))
        self.subscriber = ZMQCameraSubscriber(host = host, port = cam_port_offset + cam_id - 1, topic_type = 'RGB')
        
        # Setting frequency
        self.timer = FrequencyTimer(CAM_FPS) 

    def stream(self):
        while True:
            try:
                self.timer.start_loop()

                image, _ = self.subscriber.recv_rgb_image()
                rescaled_image = rescale_image(image, VISUAL_RESCALE_FACTOR)
                cv2.imshow('Robot camera {} - RGB stream'.format(self.camera_number), rescaled_image)
                cv2.waitKey(1)

                self.timer.end_loop()
            except KeyboardInterrupt:
                break
                
        print('Exiting visualizer.')