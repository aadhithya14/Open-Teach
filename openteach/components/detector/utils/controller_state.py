import time
import numpy as np
import pandas as pd
from typing import Tuple
from dataclasses import dataclass
from openteach.components.detector.utils.state import State
from scipy.spatial.transform import Rotation as R


@dataclass
class ControllerState(State):
    left_x: bool
    left_y: bool
    left_menu: bool
    left_thumbstick: bool
    left_index_trigger: float
    left_hand_trigger: float
    left_thumbstick_axes: np.ndarray[Tuple[float, float]]
    left_local_position: np.ndarray[Tuple[float, float, float]]
    left_local_rotation: np.ndarray[Tuple[float, float, float, float]]

    right_a: bool
    right_b: bool
    right_menu: bool
    right_thumbstick: bool
    right_index_trigger: float
    right_hand_trigger: float
    right_thumbstick_axes: np.ndarray[Tuple[float, float]]
    right_local_position: np.ndarray[Tuple[float, float, float]]
    right_local_rotation: np.ndarray[Tuple[float, float, float, float]]

    created_timestamp: float

    @property
    def right_position(self) -> np.ndarray:
        return self.right_affine[:3, 3]

    @property
    def left_position(self) -> np.ndarray:
        return self.left_affine[:3, 3]

    @property
    def right_rotation_matrix(self) -> np.ndarray:
        return self.right_affine[:3, :3]

    @property
    def left_rotation_matrix(self) -> np.ndarray:
        return self.left_affine[:3, :3]

    @property
    def left_affine(self) -> np.ndarray:
        return self.get_affine(self.left_local_position, self.left_local_rotation)

    @property
    def right_affine(self) -> np.ndarray:
        return self.get_affine(self.right_local_position, self.right_local_rotation)

    def get_affine(self, controller_position: np.ndarray, controller_rotation: np.ndarray):
        """ Returns a 4x4 affine matrix from the controller's position and rotation.
        Args:
            controller_position: 3D position of the controller.
            controller_rotation: 4D quaternion of the controller's rotation.

            All in headset space.
        """

        return np.block([[R.as_matrix(R.from_quat(controller_rotation)), controller_position[:, np.newaxis]],
                         [np.zeros((1, 3)), 1.]])


def parse_controller_state(controller_state_string: str) -> ControllerState:

    left_data, right_data = controller_state_string.split('|')

    left_data = left_data.split(';')[1:-1]
    right_data = right_data.split(';')[1:-1]

    def parse_bool(val: str) -> bool:
        return val.split(':')[1].lower().strip() == "true"

    def parse_float(val: str) -> float:
        return float(val.split(':')[1])

    def parse_list_float(val: str) -> np.ndarray:
        return np.array(list(map(float, val.split(':')[1].split(','))))

    def parse_section(data: list) -> Tuple:
        return (
            # Buttons
            parse_bool(data[0]),
            parse_bool(data[1]),
            parse_bool(data[2]),
            parse_bool(data[3]),
            # Triggers
            parse_float(data[4]),
            parse_float(data[5]),
            # Thumbstick
            parse_list_float(data[6]),
            # Pose
            parse_list_float(data[7]),
            parse_list_float(data[8])
        )

    left_parsed = parse_section(left_data)
    right_parsed = parse_section(right_data)

    return ControllerState(time.time(), *left_parsed, *right_parsed)


# TODO: Move this to a test file
if __name__ == "__main__":
    # Sample message
    test_msg = "Left Controller:;  Left X: True;  Left Y: False;  Left Menu: False;  Left Thumbstick: False;  Left Index Trigger: 0;  Left Hand Trigger: 0;  Left Thumbstick Axes: 0,0;  Left Local Position: -0.6630062,0.7440274,0.08777055;  Left Local Rotation: 0.1541033,-0.04510121,0.5017885,0.8499568;|Right Controller:;  Right A: False;  Right B: False;  Right Menu: False;  Right Thumbstick: False;  Right Index Trigger: 0;  Right Hand Trigger: 0;  Right Thumbstick Axes: 0,0;  Right Local Position: -0.5966942,0.749879,0.1490001;  Right Local Rotation: 0.1249516,0.1079503,0.3456937,0.9237044;"
    controller_state = parse_controller_state(test_msg)
    controller_state_df = controller_state.to_df()

    # from bimanual.utils.loggers import write_to_csv
    # from bimanual.utils.debug_utils import DebugTimer

    # for _ in range(5):
    #     with DebugTimer("write_to_csv"):  # looks like 0.001s/ 0.002s per call
    #         write_to_csv("test22.csv", controller_state_df)

    # from IPython import embed

    # embed()