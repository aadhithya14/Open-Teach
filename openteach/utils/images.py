import cv2
import numpy as np

def rescale_image(image, rescale_factor):
    width, height = int(image.shape[1] / rescale_factor), int(image.shape[0] / rescale_factor)
    return cv2.resize(image, (width, height), interpolation = cv2.INTER_AREA)

def stack_images(image_array):
    return np.hstack(image_array)

def rotate_image(image, angle):
        if angle == 90:
            image = cv2.rotate(image, cv2.ROTATE_90)
        elif angle == 180:
            image = cv2.rotate(image, cv2.ROTATE_180)
        elif angle == 270:
            image = cv2.rotate(image, cv2.ROTATE_270)
        
        return image