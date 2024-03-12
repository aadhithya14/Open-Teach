import numpy as np
from .plotters.plotter_2d import *
from openteach.components import Component
from openteach.constants import OCULUS_NUM_KEYPOINTS
from openteach.utils.network import ZMQKeypointSubscriber

class Hand2DVisualizer(Component):
    def __init__(self, host, transformed_keypoint_port, oculus_feedback_port, display_plot):
        self.notify_component_start('hand 2D plotter')
        self.subscriber = ZMQKeypointSubscriber(
            host = host,
            port = transformed_keypoint_port,
            topic = 'transformed_hand_coords'
        )

        self.plotter2D = PlotHand2D(host, oculus_feedback_port, display_plot)

    def _get_keypoints(self):
        raw_keypoints = self.subscriber.recv_keypoints()
        return np.array(raw_keypoints).reshape(OCULUS_NUM_KEYPOINTS, 3)

    def stream(self):
        while True:
            try:
                keypoints = self._get_keypoints()
                self.plotter2D.draw(keypoints[:, 0], keypoints[:, 1])
            except:
                break

        self.subscriber.stop()
        print('Stopping the hand 2D visualizer process')