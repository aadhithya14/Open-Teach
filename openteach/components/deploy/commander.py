import pickle
from openteach.constants import DEPTH_PORT_OFFSET
from openteach.utils.network import ZMQCameraSubscriber, create_request_socket

class DeployAPI(object):
    def __init__(self, configs, required_data):
        """
        Data structure for required_data: {
            'rgb_idxs': []
            'depth_idxs': []
        }
        """
        self.configs = configs
        self.required_data = required_data

        # Connect to the server
        self._connect_to_robot_server()

        # Initializing the required camera classes
        self._init_camera_subscribers()

    def _connect_to_robot_server(self):
        self.robot_socket = create_request_socket(
            host = self.configs.robot.deployment.host,
            port = self.configs.robot.deployment.port
        )

    def _init_camera_subscribers(self):
        self._rgb_streams, self._depth_streams = [], []

        for idx in range(len(self.configs.robot_cam_serial_numbers)):
            if idx + 1 in self.required_data['rgb_idxs']:
                self._rgb_streams.append(ZMQCameraSubscriber(
                    host = self.configs.host_address,
                    port = self.configs.cam_port_offset + idx,
                    topic_type = 'RGB'
                ))

                self._depth_streams.append(ZMQCameraSubscriber(
                    host = self.configs.host_address,
                    port = self.configs.cam_port_offset + idx + DEPTH_PORT_OFFSET,
                    topic_type = 'Depth'
                ))

    def get_robot_state(self):
        self.robot_socket.send(pickle.dumps('get_state', protocol = -1))
        robot_states = pickle.loads(self.robot_socket.recv())
        return robot_states

    def get_rgb_images(self):
        images = [stream.recv_rgb_image() for stream in self._rgb_streams]
        return images

    def get_depth_images(self):
        images = [stream.recv_depth_image() for stream in self._depth_streams]
        return images

    def send_robot_action(self, action_dict):
        self.robot_socket.send(pickle.dumps(action_dict, protocol = -1))
        self.robot_socket.recv()