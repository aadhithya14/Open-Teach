#import rospy
import numpy as np
import time
from copy import deepcopy as copy
from enum import Enum
import math

from openteach.constants import SCALE_FACTOR
from scipy.spatial.transform import Rotation as R
from openteach.constants import *
import time
import stretch_body.robot

class DexArmControl():
    def __init__(self,ip,  record_type=None):
        self.robot=stretch_body.robot.Robot()
    # Controller initializers
    def _init_control(self):
        self.robot.startup()
        self.robot.lift.move_to(0.4)
        self.robot.arm.move_by(0.1)
        self.robot.push_command()
        time.sleep(2.0)
    
    def get_arm_pose(self):
       pass 

    def get_gripper_state(self):
        gripper_position=self.robot.get_gripper_position()
        gripper_pose= dict(
            position = np.array(gripper_position[1], dtype=np.float32).flatten(),
            timestamp = time.time()
        )
        return gripper_pose

    def home_arm(self, cartesian_pose):
        self.a.motor.disable_sync_mode()
        if not self.a.startup():
            exit() # failed to start arm!
        self.a.home()
    
    def arm_control(self, arm_pose):
        self.robot.arm.move_by(arm_pose)
        self.robot.arm.push_command()

    def base_control(self, base_pose):
        self.robot.base.translate_by(base_pose)
        self.robot.base.push_command()
        
    def lift_control(self, lift_pose):
        current_pos=self.robot.lift.status['pos']
        
        self.robot.lift.move_to(lift_pose)
        self.robot.lift.push_command()
            
    def get_arm_cartesian_state(self):
        arm_cartesian_state = self.robot.arm.status['pos']
        cartesian_state = dict(
            position = np.array(arm_cartesian_state, dtype=np.float32).flatten(),
            timestamp = time.time()
        )
        return arm_cartesian_state

    def get_lift_state(self):
        lift_position=self.robot.lift.status['pos']
        lift_pose= dict(
            position = np.array(lift_position, dtype=np.float32).flatten(),
            timestamp = time.time()
        )
        return lift_pose

    def get_base_state(self):
        base_position=self.robot.base.status['pos']
        base_pose= dict(
            position = np.array(base_position, dtype=np.float32).flatten(),
            timestamp = time.time()
        )
        return base_pose
    
    def get_end_of_the_arm_state(self):
        end_of_arm_yaw= self.robot.end_of_arm.status["wrist_yaw"]["pos"]
        end_of_arm_pitch= self.robot.end_of_arm.status["wrist_pitch"]["pos"]
        end_of_arm_roll= self.robot.end_of_arm.status["wrist_roll"]["pos"]
        r1= R.from_euler('zyx', [end_of_arm_roll, end_of_arm_pitch, end_of_arm_yaw], degrees=True)
        return r1

    # Full robot commands
    def move_robot(self,arm_angles):
        self.a.move_to(arm_angles)
        self.a.push_command()
        self.a.motor.wait_until_at_setpoint()

    def home_lift(self):
        self.l.move_to(0.6)
        self.l.push_command()
        self.l.motor.wait_until_at_setpoint()
        
    def home_robot(self):
        self.home_arm() # For now we're using cartesian values

    def set_gripper_status(self, position):
        self.robot.end_of_arm.move_to('stretch_gripper',position)
        
    def get_gripper_position(self):
        gripper_position=self.robot.end_of_arm.get_position('stretch_gripper')