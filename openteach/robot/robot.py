from abc import ABC, abstractmethod

class RobotWrapper(ABC):
    @property
    @abstractmethod
    def name(self):
        pass

    @property
    @abstractmethod
    def recorder_functions(self):
        pass

    @property
    @abstractmethod
    def data_frequency(self):
        pass

    @abstractmethod
    def get_joint_state(self):
        pass

    @abstractmethod
    def get_joint_position(self):
        pass

    # @abstractmethod
    def get_cartesian_position(self):
        pass

    # @abstractmethod
    def get_joint_velocity(self):
        pass

    # @abstractmethod
    def get_joint_torque(self):
        pass

    @abstractmethod
    def home(self):
        pass

    @abstractmethod
    def move(self, input_angles):
        pass

    @abstractmethod
    def move_coords(self, input_coords):
        pass

    # @abstractmethod
    # def reset(self):
    #     pass

    # @abstractmethod
    # def arm_control(self):
    #     pass

    # @abstractmethod
    # def set_gripper_state(self):
    #     pass