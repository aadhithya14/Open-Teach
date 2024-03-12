import cv2 
import numpy as np

import matplotlib 
import matplotlib.pyplot as plt 

from .plotter import Plotter 
from openteach.constants import *

class XelaPlotter(Plotter):
    def __init__(self, display_plot=True):
        if not display_plot:
            matplotlib.use('Agg')

        # Create the figures and self.axs
        self.fig, self.axs = plt.subplots(figsize=(15,15), nrows=4, ncols=4, dpi=60)
        
        # Blank images to be used since cv2 images is used
        # NOTE: Not sure if these are needed
        self.blank_image = np.zeros((240,240,3))
        self.imgs = []
        for i in range(4):
            self.imgs.append([self.axs[i,j].imshow(self.blank_image.copy()) for j in range(4)])
        
        for i in range(4):
            for j in range(4):
                self.axs[i,j].set_title("Finger: {} Sensor: {}".format(i,j))

        self._set_circle_coordinates()

    def _set_limits(self):
        pass

    def _set_circle_coordinates(self):
        self.circle_coords_in_one_circle = []
        for j in range(48, 192+1, 48): # Y
            for i in range(48, 192+1, 48): # X - It goes from top left to bottom right row first 
                self.circle_coords_in_one_circle.append([i,j]) 

    def _draw_one_sensor(self, ax, sensor_values, img=None):
        # sensor_values: (16,3) - 3 values for each taxel
        img_shape = (240,240,3)
        blank_image = np.ones(img_shape, np.uint8) * 255 
        if img is None:
            img = ax.imshow(blank_image.copy())

        # Plot the circles in one sensor
        for i in range(sensor_values.shape[0]):
            # Get the center coordinates of the circle
            center_coordinates = (
                self.circle_coords_in_one_circle[i][0] + int(sensor_values[i,0]/20),
                self.circle_coords_in_one_circle[i][1] + int(sensor_values[i,1]/20)
            )
            radius = max(10 + int(sensor_values[i,2]/10), 0)

            if i == 0: 
                frame_axis = cv2.circle(blank_image.copy(), center_coordinates, radius, color=(0,255,0), thickness=-1)
            else:
                frame_axis = cv2.circle(frame_axis.copy(), center_coordinates, radius, color=(0,255,0), thickness=-1)

        img.set_array(frame_axis)

        return img, frame_axis

    def draw(self, all_sensor_values):
        # Set the plot limits
        # self._set_limits() - TODO: Do we really not need this?

        # Plot the sensors for each values
        for row_id in range(4):
            for column_id in range(4):
                if row_id + column_id > 0: # The top left axis should stay empty
                    self.imgs[column_id][row_id], _ = self._draw_one_sensor(
                        self.axs[column_id, row_id],
                        sensor_values = all_sensor_values[row_id*4 + column_id - 1],
                        img = self.imgs[column_id][row_id]
                    )

        # TODO: Fix the issue with the last sensor value
        plt.savefig(XELA_PLOT_SAVE_PATH) # NOTE: this is for debugging purposes as well

        # Resetting and pausing the plot - NOTE: Make sure this works as well
        plt.pause(0.01)
        plt.cla()

class XelaCurvedPlotter(Plotter):
    def __init__(self, display_plot=True):
        if not display_plot:
            matplotlib.use('Agg')

        thumb = [['thumb_empty'],
          ['thumb_tip'],
          ['thumb_section2'],
          ['thumb_section3']]

        index = [['index_tip'],
                ['index_section1'],
                ['index_section2'],
                ['index_section3']]

        ring = [['ring_tip'],
                ['ring_section1'],
                ['ring_section2'],
                ['ring_section3']]

        middle = [['mid_tip'],
                ['mid_section1'],
                ['mid_section2'],
                ['mid_section3']]

        all_fingers = [thumb, index, middle, ring]

        hand = [[thumb, index, middle, ring],
                ['palm', 'palm', 'palm', 'palm']]
        
        fig, self.axs = plt.subplot_mosaic(hand, figsize=(10,20))

    def _set_limits(self):
        pass

    def plot_tactile_sensor(self,ax, sensor_values, use_img=False, img=None, title='Tip Position'):
    # sensor_values: (16, 3) - 3 values for each tactile - x and y represents the position, z represents the pressure on the tactile point
        img_shape = (240, 240, 3) # For one sensor
        blank_image = np.ones(img_shape, np.uint8) * 255
        if use_img == False: 
            img = ax.imshow(blank_image.copy())
        ax.set_title(title)

        # Set the coordinates for each circle
        tactile_coordinates = []
        for j in range(60, 180+1, 40): # Y
            for i in range(60, 180+1, 40): # X - It goes from top left to bottom right row first 
                tactile_coordinates.append([i,j])

        # Plot the circles 
        for i in range(sensor_values.shape[0]):
            center_coordinates = (
                tactile_coordinates[i][0] + int(sensor_values[i,0]/20), # NOTE: Change this
                tactile_coordinates[i][1] + int(sensor_values[i,1]/20)
            )
            radius = max(10 + int(sensor_values[i,2]/10), 2)
        
            if i == 0:
                frame_axis = cv2.circle(blank_image.copy(), center_coordinates, radius, color=(0,255,0), thickness=-1)
            else:
                frame_axis = cv2.circle(frame_axis.copy(), center_coordinates, radius, color=(0,255,0), thickness=-1)

        img.set_array(frame_axis)

        return img, frame_axis

    def plot_tactile_curved_tip(self,ax, sensor_values, use_img=False, img=None, title='Tip Position'):
        # sensor_values: (16, 3) - 3 values for each tactile - x and y represents the position, z represents the pressure on the tactile point
        img_shape = (240, 240, 3) # For one sensor
        blank_image = np.ones(img_shape, np.uint8) * 255
        if use_img == False: 
            img = ax.imshow(blank_image.copy())
        ax.set_title(title)

        # Set the coordinates for each circle
        tactile_coordinates = []
        for j in range(20, 240, 40): # y axis
            # x axis is somewhat hard coded
            for i in range(20, 240, 40):
                if j == 20 and (i == 100 or i == 140): # Only the middle two will be added
                    tactile_coordinates.append([i,j])
                elif (j > 20 and j < 100) and (i > 20 and i < 220):
                    tactile_coordinates.append([i,j])
                elif j >= 100: 
                    tactile_coordinates.append([i,j])
        
        # Plot the circles 
        for i in range(sensor_values.shape[0]):
            center_coordinates = (
                tactile_coordinates[i][0] + int(sensor_values[i,0]/20),
                tactile_coordinates[i][1] + int(sensor_values[i,1]/20)
            )
            radius = max(10 + int(sensor_values[i,2]/10), 2)
        
            if i == 0:
                frame_axis = cv2.circle(blank_image.copy(), center_coordinates, radius, color=(0,255,0), thickness=-1)
            else:
                frame_axis = cv2.circle(frame_axis.copy(), center_coordinates, radius, color=(0,255,0), thickness=-1)

        img.set_array(frame_axis)

        return img, frame_axis


    def plot_tactile_palm(self,ax, sensor_values, use_img=False, img=None, title='Tip Position'):
        # sensor_values: (16, 3) - 3 values for each tactile - x and y represents the position, z represents the pressure on the tactile point
        img_shape = (480, 960, 3) # For one sensor
        blank_image = np.ones(img_shape, np.uint8) * 255
        if use_img == False: 
            img = ax.imshow(blank_image.copy())
        ax.set_title(title)

        # Set the coordinates for each circle
        tactile_coordinates = []

        for j in range(70, 190+1, 40):
            for i in range(220, 420+1, 40):
                tactile_coordinates.append([i,j])

        for j in range(70, 190+1, 40):
            for i in range(540, 740+1, 40):
                tactile_coordinates.append([i,j])

        for j in range(270, 390+1, 40):
            for i in range(540, 740+1, 40):
                tactile_coordinates.append([i,j])



        # for j in range(70, 410+1, 40):
        #     for i in range(180, 780+1, 40):
        #         if (j < 230) and ((i >= 220 and i <= 420) or (i >= 540 and i <= 740)):
        #             tactile_coordinates.append([i,j])
        #         elif (j > 230) and (i >= 540 and i <= 740):
        #             tactile_coordinates.append([i,j])

        #print(len(tactile_coordinates))

        # Plot the circles 
        for i in range(sensor_values.shape[0]):
            #print(sensor_values[i,0])
            center_coordinates = (
                tactile_coordinates[i][0] + int(sensor_values[i,0]/20),
                tactile_coordinates[i][1] + int(sensor_values[i,1]/20)
            )
            radius = max(10 + int(sensor_values[i,2]/10), 2)
        
            if i == 0:
                frame_axis = cv2.circle(blank_image.copy(), center_coordinates, radius, color=(0,255,0), thickness=-1)
            else:
                frame_axis = cv2.circle(frame_axis.copy(), center_coordinates, radius, color=(0,255,0), thickness=-1)

        img.set_array(frame_axis)

        return img, frame_axis

    def draw(self,palm_sensor_values,fingertip_sensor_values,finger_sensor_values):
        cnt_fingertip=0
        cnt_finger=0
        for k in self.axs:
            #print('k: {}'.format(k))
            if 'tip' in k:
                #sensor_values = np.random.randn(30,3) * 20
                self.fingertip_sensor_values=fingertip_sensor_values
                #print(self.fingertip_sensor_values)
                self.plot_tactile_curved_tip(self.axs[k], sensor_values=self.fingertip_sensor_values[cnt_fingertip], title=k)
                cnt_fingertip+=1
            elif 'palm' in k:
                #sensor_values = np.random.randn(72,3) * 20
                palm_sensor_values = np.concatenate(palm_sensor_values, axis=0)
                assert palm_sensor_values.shape == (72,3), f'palm_sensor_values.shape: {palm_sensor_values.shape}'
                # self.palm_sensor_values= palm_sensor_values
                self.plot_tactile_palm(self.axs[k], sensor_values = palm_sensor_values, title=k)
                # cnt_palm+=1
            elif not 'empty' in k:
                self.finger_sensor_values= finger_sensor_values
                self.plot_tactile_sensor(self.axs[k], sensor_values=self.finger_sensor_values[cnt_finger], title=k)
                cnt_finger+=1
            self.axs[k].get_yaxis().set_ticks([])
            self.axs[k].get_xaxis().set_ticks([])

        plt.savefig(XELA_PLOT_SAVE_PATH) # NOTE: this is for debugging purposes as well

        # Resetting and pausing the plot - NOTE: Make sure this works as well
        plt.pause(0.01)
        plt.cla()
        

    