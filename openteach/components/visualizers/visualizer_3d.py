import numpy as np
from .plotters.plotter_3d import *
from openteach.components import Component
from openteach.constants import OCULUS_NUM_KEYPOINTS
from openteach.utils.network import ZMQKeypointSubscriber

class Hand3DVisualizer(Component):
    def __init__(self, host, port):
        self.notify_component_start('hand 3D plotter')
        self.subscriber = ZMQKeypointSubscriber(
            host = host,
            port = port,
            topic = 'transformed_hand_coords'
        )

        # Initializing the plotting object
        self.plotter3D = PlotHand3D()

    def _get_keypoints(self):
        raw_keypoints = self.subscriber.recv_keypoints()
        return np.array(raw_keypoints).reshape(OCULUS_NUM_KEYPOINTS, 3)

    def stream(self):
        while True:
            try:
                keypoints = self._get_keypoints()
                self.plotter3D.draw(keypoints[:, 0], keypoints[:, 1], keypoints[:, 2])
            except:
                break

        self.subscriber.stop()
        print('Stopping the hand 3D visualizer process.')


class OculusRightHandDirVisualizer(Component):
    def __init__(self, host, port, scaling_factor = 0.2):
        # Other parameters
        self.scaling_factor = scaling_factor

        # Initializing the Keypoint variable and subscriber
        self.subscriber = ZMQKeypointSubscriber(
            host = host,
            port = port,
            topic = 'transformed_hand_frame'
        )

        # Initializing the plotting object
        self.notify_component_start('hand direction plotter')
        self.dir_plotter = PlotHandDirection()

    def _get_directions(self):
        raw_directions = self.subscriber.recv_keypoints()
        return np.array(raw_directions).reshape(4, 3)

    def stream(self):
        while True:
            try:
                directions = self._get_directions()
                self.dir_plotter.draw(directions[:, 0], directions[:, 1], directions[:, 2])
            except:
                break

        self.subscriber.stop()
        print('Stopping the hand direction visualizer process.')