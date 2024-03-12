import os
import cv2
import matplotlib
import matplotlib.pyplot as plt
from .plotter import Plotter
from openteach.utils.network import ZMQCompressedImageTransmitter
from openteach.utils.files import *
from openteach.constants import *

def plot_line(X1, X2, Y1, Y2):
    plt.plot([X1, X2], [Y1, Y2])

class PlotHand2D(Plotter):
    def __init__(self, host, port, display_plot):
        # Display plot
        if not display_plot:
            matplotlib.use('Agg')

        # Thumb bound info
        self.display_plot = display_plot
        self.thumb_bounds = None
        self.thumb_bounds_path = VR_DISPLAY_THUMB_BOUNDS_PATH
        self.bound_update_counter = 0
        self._check_thumb_bounds()

        # Checking image storage path
        make_dir(os.path.join(CALIBRATION_FILES_PATH))

        # Figure settings
        self.fig = plt.figure(figsize=(6, 6), dpi=60)

        # Plot streamer settings
        self.socket = ZMQCompressedImageTransmitter(host = host, port = port)

    def _check_thumb_bounds(self):
        if check_file(self.thumb_bounds_path):
            self.thumb_bounds = get_npz_data(self.thumb_bounds_path)

    def _set_limits(self):
        plt.axis([-0.12, 0.12, -0.02, 0.2])

    def _draw_thumb_bounds(self):
        for idx in range(VR_THUMB_BOUND_VERTICES):
            plot_line(
                self.thumb_bounds[idx][0], 
                self.thumb_bounds[(idx + 1) % VR_THUMB_BOUND_VERTICES][0], 
                self.thumb_bounds[idx][1], 
                self.thumb_bounds[(idx + 1) % VR_THUMB_BOUND_VERTICES][1]
            )
        
    def draw_hand(self, X, Y):
        plt.plot(X, Y, 'ro')

        if self.thumb_bounds is not None:
            self._draw_thumb_bounds()

        # Drawing connections fromn the wrist - 0
        for idx in OCULUS_JOINTS['metacarpals']:
            plot_line(X[0], X[idx], Y[0], Y[idx])

        # Drawing knuckle to knuckle connections and knuckle to finger connections
        for key in ['knuckles', 'thumb', 'index', 'middle', 'ring', 'pinky']:
            for idx in range(len(OCULUS_JOINTS[key]) - 1):
                plot_line(
                    X[OCULUS_JOINTS[key][idx]], 
                    X[OCULUS_JOINTS[key][idx + 1]], 
                    Y[OCULUS_JOINTS[key][idx]], 
                    Y[OCULUS_JOINTS[key][idx + 1]]
                )

    def draw(self, X, Y):
        # Setting the plot limits
        self._set_limits()

        # Resetting the thumb bounds
        if self.bound_update_counter % 10 == 0:
            self._check_thumb_bounds()
            self.bound_update_counter = 0
        else:
            self.bound_update_counter += 1

        # Plotting the lines to visualize the hand
        self.draw_hand(X, Y)

        # Saving and obtaining the plot
        plt.savefig(VR_2D_PLOT_SAVE_PATH)
        plot = cv2.imread(VR_2D_PLOT_SAVE_PATH)
        self.socket.send_image(plot)

        # Resetting and pausing the 3D plot
        plt.pause(0.001) # This make the graph show up if matplotlib is in Tkinter mode
        plt.cla()