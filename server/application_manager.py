import zmq
import base64
import numpy as np
import pickle

class VideoStreamer(object):
    def __init__(self, host, cam_port):
        self._init_socket(host, cam_port)

    def _init_socket(self, host, port):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt(zmq.CONFLATE, 1)
        self.socket.connect('tcp://{}:{}'.format(host, port))
        self.socket.setsockopt(zmq.SUBSCRIBE, b"rgb_image")

    def _get_image(self):
        raw_data = self.socket.recv()
        data = raw_data.lstrip(b"rgb_image ")
        data = pickle.loads(data)
        encoded_data = np.fromstring(base64.b64decode(data['rgb_image']), np.uint8)
        return encoded_data.tobytes()

    def yield_frames(self):
        while True:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + self._get_image() + b'\r\n')  # concat frame one by one and show result


class MonitoringApplication(object):
    def __init__(self, configs):
        # Loading the network configurations
        self.host_address = configs.host_address
        self.keypoint_port = configs.keypoint_port
        self.port_offset = configs.cam_port_offset
        self.num_cams = len(configs.robot_cam_serial_numbers)

        # Initializing the streamers        
        self._init_cam_streamers()
        self._init_graph_streamer()

        # Initializing frequency checkers
        self._init_frequency_checkers()
        
    def _init_graph_streamer(self):
        # TODO
        pass

    def _init_frequency_checkers(self):
        # TODO - Raw keypoint frequency
        # TODO - Transformed keypoint frequency
        pass

    def _init_cam_streamers(self):
        self.cam_streamers = []
        for idx in range(self.num_cams):
            self.cam_streamers.append(
                VideoStreamer(
                    host = self.host_address,
                    cam_port = self.port_offset + idx
                )
            )

    def get_cam_streamer(self, id):
        return self.cam_streamers[id - 1]