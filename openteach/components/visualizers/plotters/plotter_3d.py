from openteach.constants import *
from .plotter import Plotter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class PlotHand3D(Plotter):
    def __init__(self):
        # Initializing the figure
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')

        # Loading Joint information
        self.joint_information = OCULUS_JOINTS  
        self.view_limits = OCULUS_VIEW_LIMITS

        # Setting the visualizer limits
        self._set_limits()

    def _plot_line(self, X1, X2, Y1, Y2, Z1, Z2):
        self.ax.plot([X1, X2], [Y1, Y2], [Z1, Z2])

    def _set_limits(self):
        self.ax.set_xlim(self.view_limits['x_limits'])
        self.ax.set_ylim(self.view_limits['y_limits'])
        self.ax.set_zlim3d(self.view_limits['z_limits'][0], self.view_limits['z_limits'][1])

    def _draw_hand(self, X, Y, Z):
        self.plot3D = self.ax.scatter3D(X, Y, Z)

        # Drawing connections fromn the wrist - 0
        for idx in self.joint_information['metacarpals']:
            self._plot_line(X[0], X[idx], Y[0], Y[idx], Z[0], Z[idx])

        # Drawing knuckle to knuckle connections and knuckle to finger connections
        for key in ['knuckles', 'thumb', 'index', 'middle', 'ring', 'pinky']:
            for idx in range(len(self.joint_information[key]) - 1):
                self._plot_line(
                    X[self.joint_information[key][idx]], 
                    X[self.joint_information[key][idx + 1]], 
                    Y[self.joint_information[key][idx]], 
                    Y[self.joint_information[key][idx + 1]],
                    Z[self.joint_information[key][idx]], 
                    Z[self.joint_information[key][idx + 1]]
                )

    def draw(self, X, Y, Z):
        # Setting plotting limits
        self._set_limits()
        
        # Plotting the hand bones
        self._draw_hand(X, Y, Z)
        plt.draw()

        # Resetting and Pausing the 3D plot
        self.fig.canvas.flush_events()
        plt.pause(0.01)
        plt.cla()

        # Removing the drawing
        self.plot3D.remove()


class PlotHandDirection(Plotter):
    def __init__(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection = '3d')
        
        self._set_limits()

    def _set_limits(self):
        self.ax.set_xlim([-0.3, 0.3])
        self.ax.set_ylim([-0.3, 0.3])
        self.ax.set_zlim3d(-0.3, 0.3)

    def draw(self, X, Y, Z):
        self._set_limits()
        self.plot3D = self.ax.scatter(X, Y, Z)

        # Draw the axes
        self.ax.plot([X[0], X[1]], [Y[0], Y[1]], [Z[0], Z[1]], color="blue", label='hand_cross')
        self.ax.plot([X[0], X[2]], [Y[0], Y[2]], [Z[0], Z[2]], color="green", label='hand_normal')
        self.ax.plot([X[0], X[3]], [Y[0], Y[3]], [Z[0], Z[3]], color="red", label='hand_direction')

        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')

        plt.draw()

        # Resetting and Pausing the 3D plot
        self.fig.canvas.flush_events()
        plt.pause(0.01)
        plt.cla()

        # Removing the drawing
        self.plot3D.remove()