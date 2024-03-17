import numpy as np
import time
import rospy
from copy import deepcopy as copy


class DexArmControl():
    def __init__(self, record_type=None, robot_type='both'):
    # Initialize Controller Specific Information
        try:
                rospy.init_node("dex_arm", disable_signals = True, anonymous = True)
        except:
                pass
    # Controller initializers
    def _init_robot_control(self):
        # Have the ROS subscribers and publishers here
        pass

    # Rostopic callback functions
    def _callback_robot_joint_state(self, joint_state):
        self.robot_joint_state = joint_state

    #Commanded joint state is basically the joint state being sent as an input to the controller
    def _callback_robot_commanded_joint_state(self, joint_state):
        self.robot_commanded_joint_state = joint_state

    # State information function
    def get_robot_state(self):
        # Get the robot joint state 
        raw_joint_state =self.robot_joint_state
        joint_state = dict(
            position = np.array(raw_joint_state.position, dtype = np.float32),
            velocity = np.array(raw_joint_state.velocity, dtype = np.float32),
            effort = np.array(raw_joint_state.effort, dtype = np.float32),
            timestamp = raw_joint_state.header.stamp.secs + (raw_joint_state.header.stamp.nsecs * 1e-9)
        )
        return joint_state

    # Commanded joint state is the joint state being sent as an input to the controller
    def get_commanded_robot_state(self):
        raw_joint_state = copy(self.robot_commanded_joint_state)

        joint_state = dict(
            position = np.array(raw_joint_state.position, dtype = np.float32),
            velocity = np.array(raw_joint_state.velocity, dtype = np.float32),
            effort = np.array(raw_joint_state.effort, dtype = np.float32),
            timestamp = raw_joint_state.header.stamp.secs + (raw_joint_state.header.stamp.nsecs * 1e-9)
        )
        return joint_state
        
    # Get the robot joint/cartesian position
    def get_robot_position(self):
       #Get Robot Position
        pass

    # Get the robot joint velocity
    def get_robot_velocity(self):
        #Get Robot Velocity
        pass

    # Get the robot joint torque
    def get_robot_torque(self):
        # Get torque applied by the robot.
        pass

    # Get the commanded robot joint position
    def get_commanded_robot_joint_position(self):
        pass

    # Movement functions
    def move_robot(self, joint_angles):
        pass

    # Home Robot
    def home_robot(self):
        pass

    # Reset the Robot
    def reset_robot(self):
        pass

    # Full robot commands
    def move_robot(self, joint_angles, arm_angles):
        pass

    def arm_control(self, arm_pose):
        pass

    #Home the Robot
    def home_robot(self):
        pass
        # For now we're using cartesian values
