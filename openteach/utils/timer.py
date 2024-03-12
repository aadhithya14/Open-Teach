import cv2
import time
import zmq
import pickle
import numpy as np
import base64

class FrequencyTimer(object):
    def __init__(self, frequency_rate):
        self.time_available = 1e9 / frequency_rate

    def start_loop(self):
        self.start_time = time.time_ns()

    def end_loop(self):
        wait_time = self.time_available + self.start_time
        
        while time.time_ns() < wait_time:
            continue

class SocketChecker(object):
    def __init__(self, host, port, topic_name, print_data, data_type = None):
        self.data = None
        self.previous_data = None
        self.print_data = print_data
        self.data_type = data_type
        self.topic_name = topic_name
        self._init_connection(host, port)

    def _init_connection(self, host, port):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt(zmq.CONFLATE, 1)
        self.socket.setsockopt(zmq.SUBSCRIBE, bytes(self.topic_name, 'utf-8'))
        self.socket.connect('tcp://{}:{}'.format(host, port))

    def _reinit_counter(self):
        self.counter = 0
        self.start_time = time.time()

    def _calculate_frequency(self):
        return self.counter / (time.time() - self.start_time)

    def _decode_array(self):
        processed_data = self.data.lstrip(
            bytes("{} ".format(self.topic_name), 'utf-8')
        )
        print(pickle.loads(processed_data))

    def _decode_rgb_image(self):
        frame = self.data.lstrip(b"rgb_image ")
        encoded_data = np.fromstring(base64.b64decode(frame), np.uint8)
        image = cv2.imdecode(encoded_data, 1)
        cv2.imshow(image)
        cv2.waitKey(1)

    def check_connection(self):
        self._reinit_counter()
        while True:
            self.data = self.socket.recv()
            if self.data is not None and self.data is not self.previous_data:
                # To see the data - usually reduces the actual frequency. Use it to just see the stream
                if self.print_data:
                    if self.data_type == 'FloatArray':
                        self._decode_array()
                    else:
                        self._decode_rgb_image()

                self.counter += 1
                print('Frequency: {}'.format(self._calculate_frequency()))

                if self.counter > 10:
                    self._reinit_counter()
            else:
                self.start_time = time.time()