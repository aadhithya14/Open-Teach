from abc import ABC, abstractmethod
from openteach.components import Component
import numpy as np

class Operator(Component, ABC):
    @property
    @abstractmethod
    def timer(self):
        return self._timer

    # This function is used to create the robot
    @property
    @abstractmethod
    def robot(self):
        return self._robot

    # This function is the subscriber for the hand keypoints
    @property
    @abstractmethod
    def transformed_hand_keypoint_subscriber(self):
        return self._transformed_hand_keypoint_subscriber
    
    #This function is the subscriber for the arm keypoints
    @property
    @abstractmethod
    def transformed_arm_keypoint_subscriber(self):
        return self._transformed_arm_keypoint_subscriber

    #This function has the majority of retargeting code happening
    @abstractmethod
    def _apply_retargeted_angles(self):
        pass

    #This function applies the retargeted angles to the robot
    def stream(self):
        self.notify_component_start('{} control'.format(self.robot))
        print("Start controlling the robot hand using the Oculus Headset.\n")

        while True:
            # try:
                if self.return_real() is True:
                    if self.robot.get_joint_position() is not None:
                        #print("######")
                        self.timer.start_loop()
                        
                        # Retargeting function
                        self._apply_retargeted_angles()

                        self.timer.end_loop()
                else:
                    self.timer.start_loop()
                    
                    # Retargeting function
                    self._apply_retargeted_angles()

                    self.timer.end_loop()

            # except KeyboardInterrupt:
            #     break
        
        self.transformed_arm_keypoint_subscriber.stop()
        self.transformed_hand_keypoint_subscriber.stop()
        print('Stopping the teleoperator!')