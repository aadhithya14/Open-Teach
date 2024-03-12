import sys
import numpy as np
from openteach.constants import *
from openteach.utils.files import *
from openteach.utils.network import ZMQKeypointSubscriber

class OculusThumbBoundCalibrator(object):
    def __init__(self, host, transformed_keypoints_port):
        # Initializing the keypoint subscriber
        self.transformed_keypoint_subscriber = ZMQKeypointSubscriber(
            host = host,
            port = transformed_keypoints_port,
            topic = 'transformed_hand_coords'
        )

        # Storage paths
        make_dir(CALIBRATION_FILES_PATH)

    def _stop(self):
        print('Stopping the calibrator.')
        self.transformed_keypoint_subscriber.stop()

    def _get_thumb_tip_coord(self):
        return self.transformed_keypoint_subscriber.recv_keypoints()[OCULUS_JOINTS['thumb'][-1]]

    def _get_xy_coords(self):
        return [self._get_thumb_tip_coord()[0], self._get_thumb_tip_coord()[1]]

    def _get_z_coord(self):
        return self._get_thumb_tip_coord()[-1]

    def _calibrate(self):
        _ = input("Place the thumb in the top right corner.")
        top_right_coord = self._get_xy_coords()

        _ = input("Place the thumb in the bottom right corner.")
        bottom_right_coord = self._get_xy_coords()

        _ = input("Place the thumb in the index bottom corner.")
        index_bottom_coord = self._get_xy_coords()

        _ = input("Place the thumb in the index top corner.")
        index_top_coord = self._get_xy_coords()

        _ = input("Stretch the thumb to get highest index bound z value.")
        index_high_z = self._get_z_coord()

        _ = input("Relax the thumb to get the lowest index bound z value.")
        index_low_z = self._get_z_coord()

        _ = input("Place the thumb in the middle bottom corner.")
        middle_bottom_coord = self._get_xy_coords()

        _ = input("Place the thumb in the middle top corner.")
        middle_top_coord = self._get_xy_coords()

        _ = input("Stretch the thumb to get highest middle bound z value.")
        middle_high_z = self._get_z_coord()

        _ = input("Relax the thumb to get the lowest middle bound z value.")
        middle_low_z = self._get_z_coord()

        _ = input("Place the thumb in the ring bottom corner.")
        ring_bottom_coord = self._get_xy_coords()

        _ = input("Place the thumb in the ring top corner.")
        ring_top_coord = self._get_xy_coords()
        
        _ = input("Stretch the thumb to get highest ring bound z value.")
        ring_high_z = self._get_z_coord()

        _ = input("Relax the thumb to get the lowest ring bound z value.")
        ring_low_z = self._get_z_coord()

        thumb_index_bounds = np.array([
            top_right_coord,
            bottom_right_coord,
            index_bottom_coord,
            index_top_coord,
            [index_low_z, index_high_z]
        ])

        thumb_middle_bounds = np.array([
            index_top_coord,
            index_bottom_coord,
            middle_bottom_coord,
            middle_top_coord,
            [middle_low_z, middle_high_z]
        ])

        thumb_ring_bounds = np.array([
            middle_top_coord,
            middle_bottom_coord,
            ring_bottom_coord,
            ring_top_coord,
            [ring_low_z, ring_high_z]
        ])

        thumb_bounds = np.vstack([thumb_index_bounds, thumb_middle_bounds, thumb_ring_bounds])

        handpose_coords = np.array([
            top_right_coord,
            bottom_right_coord,
            index_bottom_coord,
            middle_bottom_coord,
            ring_bottom_coord,
            ring_top_coord,
            middle_top_coord,
            index_top_coord
        ])

        np.save(VR_DISPLAY_THUMB_BOUNDS_PATH, handpose_coords)
        np.save(VR_THUMB_BOUNDS_PATH, thumb_bounds)

        return thumb_index_bounds, thumb_middle_bounds, thumb_ring_bounds

    def get_bounds(self):
        sys.stdin = open(0) # To take inputs while spawning multiple processes

        if check_file(VR_THUMB_BOUNDS_PATH):
            use_calibration_file = input("\nCalibration file already exists. Do you want to create a new one? Press y for Yes else press Enter")

            if use_calibration_file == "y":
                thumb_index_bounds, thumb_middle_bounds, thumb_ring_bounds = self._calibrate()
            else:
                calibrated_bounds = np.load(VR_THUMB_BOUNDS_PATH)
                thumb_index_bounds = calibrated_bounds[:5]
                thumb_middle_bounds = calibrated_bounds[5:10]
                thumb_ring_bounds = calibrated_bounds[10:]

        else:
            print("\nNo calibration file found. Need to calibrate hand poses.\n")
            thumb_index_bounds, thumb_middle_bounds, thumb_ring_bounds = self._calibrate()
        self._stop()
        #return thumb_index_bounds, thumb_middle_bounds, thumb_ring_bounds #NOTE: We're changing this to only return the ring top/botton and right top/bottom

        # Return one bound with the max of the zs as the z bound - NOTE: Here, we're getting the largest limits
        high_z = max(thumb_index_bounds[4][1], thumb_middle_bounds[4][1], thumb_ring_bounds[4][1])
        low_z = min(thumb_index_bounds[4][0], thumb_middle_bounds[4][0], thumb_ring_bounds[4][0])

        # Get the four bounds from index and the ring bounds
        thumb_bounds = [
            thumb_index_bounds[0],
            thumb_index_bounds[1],
            thumb_ring_bounds[2], 
            thumb_ring_bounds[3],
            [low_z, high_z]
        ]
        #print(thumb_bounds)
        return thumb_bounds
        
    
    """top_right_coord[1] = top_right_coord[1]+0.02
        bottom_right_coord[1]=bottom_right_coord[1]-0.02
        index_bottom_coord[1]=index_bottom_coord[1]-0.03
        middle_bottom_coord[1]=middle_bottom_coord[1]-0.03
        ring_bottom_coord[1]=ring_bottom_coord[1]-0.02
        ring_top_coord[1]=ring_top_coord[1]+0.02
        middle_top_coord[1]=middle_top_coord[1]+0.02
        index_top_coord[1]=index_top_coord[1]+0.02
        index_high_z=index_high_z-0.01
        index_low_z=index_low_z-0.01  
    """