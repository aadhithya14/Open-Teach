import zmq
import cv2
import base64
import numpy as np
import pickle
import blosc as bl
import threading
from queue import Queue, Empty
import subprocess
import os
import time
import socket
import struct
from threading import Thread

# ZMQ Sockets
def create_push_socket(host, port):
    context = zmq.Context()
    socket = context.socket(zmq.PUSH)
    socket.bind('tcp://{}:{}'.format(host, port))
    return socket

def create_pull_socket(host, port):
    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.setsockopt(zmq.CONFLATE, 1)
    socket.bind('tcp://{}:{}'.format(host, port))
    return socket

def create_response_socket(host, port):
    content = zmq.Context()
    socket = content.socket(zmq.REP)
    socket.bind('tcp://{}:{}'.format(host, port))
    return socket

def create_request_socket(host, port):
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect('tcp://{}:{}'.format(host, port))
    return socket

# Pub/Sub classes for Keypoints
class ZMQKeypointPublisher(object):
    def __init__(self, host, port):
        self._host, self._port = host, port
        self._init_publisher()

    def _init_publisher(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind('tcp://{}:{}'.format(self._host, self._port))

    def pub_keypoints(self, keypoint_array, topic_name):
        """
        Process the keypoints into a byte stream and input them in this function
        """
        buffer = pickle.dumps(keypoint_array, protocol = -1)
        self.socket.send(bytes('{} '.format(topic_name), 'utf-8') + buffer)

    def stop(self):
        print('Closing the publisher socket in {}:{}.'.format(self._host, self._port))
        self.socket.close()
        self.context.term()

class ZMQKeypointSubscriber(threading.Thread):
    def __init__(self, host, port, topic):
        self._host, self._port, self._topic = host, port, topic
        self._init_subscriber()

        # Topic chars to remove
        self.strip_value = bytes("{} ".format(self._topic), 'utf-8')

    def _init_subscriber(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt(zmq.CONFLATE, 1)
        self.socket.connect('tcp://{}:{}'.format(self._host, self._port))
        self.socket.setsockopt(zmq.SUBSCRIBE, bytes(self._topic, 'utf-8'))

    def recv_keypoints(self, flags=None):
        if flags is None:
            raw_data = self.socket.recv()
            raw_array = raw_data.lstrip(self.strip_value)
            return pickle.loads(raw_array)
        else: # For possible usage of no blocking zmq subscriber
            try:
                raw_data = self.socket.recv(flags)
                raw_array = raw_data.lstrip(self.strip_value)
                return pickle.loads(raw_array)
            except zmq.Again:
                # print('zmq again error')
                return None
    def stop(self):
        print('Closing the subscriber socket in {}:{}.'.format(self._host, self._port))
        self.socket.close()
        self.context.term()

# Pub/Sub classes for storing data from Realsense Cameras
class ZMQCameraPublisher(object):
    def __init__(self, host, port):
        self._host, self._port = host, port
        self._init_publisher()

    def _init_publisher(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        print('tcp://{}:{}'.format(self._host, self._port))
        self.socket.bind('tcp://{}:{}'.format(self._host, self._port))


    def pub_intrinsics(self, array):
        self.socket.send(b"intrinsics " + pickle.dumps(array, protocol = -1))

    def pub_rgb_image(self, rgb_image, timestamp):
        _, buffer = cv2.imencode('.jpg', rgb_image, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        data = dict(
            timestamp = timestamp,
            rgb_image = base64.b64encode(buffer)
        )
        self.socket.send(b"rgb_image " + pickle.dumps(data, protocol = -1))

    def pub_depth_image(self, depth_image, timestamp):
        compressed_depth = bl.pack_array(depth_image, cname = 'zstd', clevel = 1, shuffle = bl.NOSHUFFLE)
        data = dict(
            timestamp = timestamp,
            depth_image = compressed_depth
        )
        self.socket.send(b"depth_image " + pickle.dumps(data, protocol = -1))

    def stop(self):
        print('Closing the publisher socket in {}:{}.'.format(self._host, self._port))
        self.socket.close()
        self.context.term()

class ZMQCameraSubscriber(threading.Thread):
    def __init__(self, host, port, topic_type):
        self._host, self._port, self._topic_type = host, port, topic_type
        self._init_subscriber()

    def _init_subscriber(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt(zmq.CONFLATE, 1)
        print('tcp://{}:{}'.format(self._host, self._port))
        self.socket.connect('tcp://{}:{}'.format(self._host, self._port))

        if self._topic_type == 'Intrinsics':
            self.socket.setsockopt(zmq.SUBSCRIBE, b"intrinsics")
        elif self._topic_type == 'RGB':
            self.socket.setsockopt(zmq.SUBSCRIBE, b"rgb_image")
        elif self._topic_type == 'Depth':
            self.socket.setsockopt(zmq.SUBSCRIBE, b"depth_image")

    def recv_intrinsics(self):
        raw_data = self.socket.recv()
        raw_array = raw_data.lstrip(b"intrinsics ")
        return pickle.loads(raw_array)

    def recv_rgb_image(self):
        raw_data = self.socket.recv()
        data = raw_data.lstrip(b"rgb_image ")
        data = pickle.loads(data)
        encoded_data = np.fromstring(base64.b64decode(data['rgb_image']), np.uint8)
        return cv2.imdecode(encoded_data, 1), data['timestamp']
        
    def recv_depth_image(self):
        raw_data = self.socket.recv()
        striped_data = raw_data.lstrip(b"depth_image ")
        data = pickle.loads(striped_data)
        depth_image = bl.unpack_array(data['depth_image'])
        return np.array(depth_image, dtype = np.int16), data['timestamp']
        
    def stop(self):
        print('Closing the subscriber socket in {}:{}.'.format(self._host, self._port))
        self.socket.close()
        self.context.term()

# Publisher for image visualizers
class ZMQCompressedImageTransmitter(object):
    def __init__(self, host, port):
        self._host, self._port = host, port
        # self._init_push_socket()
        self._init_publisher()

    def _init_publisher(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind('tcp://{}:{}'.format(self._host, self._port))

    def _init_push_socket(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUSH)
        self.socket.bind('tcp://{}:{}'.format(self._host, self._port))

    def send_image(self, rgb_image):
        _, buffer = cv2.imencode('.jpg', rgb_image, [int(cv2.IMWRITE_WEBP_QUALITY), 10])
        self.socket.send(np.array(buffer).tobytes())

    def stop(self):
        print('Closing the publisher in {}:{}.'.format(self._host, self._port))
        self.socket.close()
        self.context.term()

class ZMQCompressedImageReciever(threading.Thread):
    def __init__(self, host, port):
        self._host, self._port = host, port
        # self._init_pull_socket()
        self._init_subscriber()

    def _init_subscriber(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt(zmq.CONFLATE, 1)
        self.socket.connect('tcp://{}:{}'.format(self._host, self._port))
        self.socket.subscribe("")

    def _init_pull_socket(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PULL)
        self.socket.setsockopt(zmq.CONFLATE, 1)
        self.socket.connect('tcp://{}:{}'.format(self._host, self._port))

    def recv_image(self):
        raw_data = self.socket.recv()
        encoded_data = np.fromstring(raw_data, np.uint8)
        decoded_frame = cv2.imdecode(encoded_data, 1)
        return decoded_frame
        
    def stop(self):
        print('Closing the subscriber socket in {}:{}.'.format(self._host, self._port))
        self.socket.close()
        self.context.term()

class ZMQButtonFeedbackSubscriber(threading.Thread):
    def __init__(self, host, port):
        self._host, self._port = host, port
        # self._init_pull_socket()
        self._init_subscriber()

    def _init_subscriber(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt(zmq.CONFLATE, 1)
        self.socket.connect('tcp://{}:{}'.format(self._host, self._port))
        self.socket.subscribe("")

    def _init_pull_socket(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PULL)
        self.socket.setsockopt(zmq.CONFLATE, 1)
        self.socket.connect('tcp://{}:{}'.format(self._host, self._port))


    def recv_keypoints(self):
        raw_data = self.socket.recv()
        return pickle.loads(raw_data)
    
    def stop(self):
        print('Closing the subscriber socket in {}:{}.'.format(self._host, self._port))
        self.socket.close()
        self.context.term()

class SCRCPY_client():
    def __init__(self, SCRCPY_dir, FFMPEG_bin, ADB_bin, host, port, SCRCPY_ver, oculus_address):
        self.bytes_sent = 0
        self.bytes_rcvd = 0
        self.images_rcvd = 0
        self.bytes_to_read = 0
        self.FFmpeg_info = []
        self.ACTIVE = True
        self.LANDSCAPE = True
        self.FFMPEGREADY = False
        self.ffoutqueue = Queue()
        self.SVR_sendFrameMeta = True
        self.HEADER_SIZE = 12
        self.RECVSIZE = 0x10000
        self.SCRCPY_ver=SCRCPY_ver
        self.host = host
        self.port = port
        self.SCRCPY_dir = SCRCPY_dir
        self.FFMPEG_bin = FFMPEG_bin
        self.ADB_bin = ADB_bin
        self.oculus_address = oculus_address
        
        if self._init_publisher():
            if self._init_subscriber():
                self.start_processing()
            else:
                print("Oculus not initialized successfully!")
        else: print("Oculus not initialized successfully!")

    def stdout_thread(self):
        while self.ACTIVE:
            rd = self.ffm.stdout.read(self.bytes_to_read)
            if rd:
                self.bytes_rcvd += len(rd)
                self.images_rcvd += 1
                self.ffoutqueue.put(rd)

    def stderr_thread(self):
        while self.ACTIVE:
            rd = self.ffm.stderr.readline()
            if rd:
                self.FFmpeg_info.append(rd.decode("utf-8"))

    def stdin_thread(self):
        while self.ACTIVE:
            if self.SVR_sendFrameMeta:
                header = self.sock.recv(self.HEADER_SIZE)
                #fd.write(header)
                pts = int.from_bytes(header[:8],
                    byteorder='big', signed=False)
                frm_len = int.from_bytes(header[8:],
                    byteorder='big', signed=False)
                
                data = self.sock.recv(frm_len)
                #fd.write(data)
                self.bytes_sent += len(data)
                self.ffm.stdin.write(data)
            else:
                data = self.sock.recv(self.RECVSIZE)
                self.bytes_sent += len(data)
                self.ffm.stdin.write(data)

    def _init_publisher(self):
        try:
            adb_push = subprocess.Popen(
                [self.ADB_bin,'-s',self.oculus_address,'push',
                os.path.join(self.SCRCPY_dir,'scrcpy-server-v1.25'),
                #  os.path.join(SCRCPY_dir,'Server.java'),
                '/data/local/tmp/scrcpy-server.jar'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self.SCRCPY_dir)
            adb_push_comm = ''.join([x.decode("utf-8") for x in adb_push.communicate() if x is not None])

            if "error" in adb_push_comm:
                print("Is your device/emulator visible to ADB?")
                raise Exception(adb_push_comm)
            '''
            ADB Shell is Blocking, don't wait up for it 
            Args for the server are as follows:
            maxSize         (integer, multiple of 8) 0
            bitRate         (integer)
            tunnelForward   (optional, bool) use "adb forward" instead of "adb tunnel"
            crop            (optional, string) "width:height:x:y"
            sendFrameMeta   (optional, bool) 

            '''

            subprocess.Popen(
                [self.ADB_bin,'-s',self.oculus_address,'forward',
                'tcp:'+str(self.port),'localabstract:scrcpy'],
                cwd=self.SCRCPY_dir).wait()
            # time.sleep(1)

            subprocess.Popen(
                [self.ADB_bin,'-s',self.oculus_address,'shell',
                'CLASSPATH=/data/local/tmp/scrcpy-server.jar',
                #  'CLASSPATH=/data/local/tmp/Server.java',
                'app_process','/','com.genymobile.scrcpy.Server',
                # str(SVR_maxSize),str(SVR_bitRate),
                # SVR_tunnelForward, SVR_crop, SVR_sendFrameMeta],
                str(self.SCRCPY_ver),
                #   "crop=2560:1440:2050:600 ", 
                "crop=2100:2700:2050:0 ", 
                  "rotation=22","tunnel_forward=true", "control=false", "cleanup=false"
                # , "max_size=1660"
                , "max_size=1280"
                ])
                # "rotation-offset=22", "scale=159", "position-x-offset=-520", "position-y-offset=-490"])
                # cwd=SCRCPY_dir)
            time.sleep(1)
        

        except FileNotFoundError:
            raise FileNotFoundError("Couldn't find ADB at path ADB_bin: "+
                    str(self.ADB_bin))
        return True

    def _init_subscriber(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.host, self.port))

        DUMMYBYTE = self.sock.recv(1)
        #fd.write(DUMMYBYTE)
        if not len(DUMMYBYTE):
            raise ConnectionError("Did not recieve Dummy Byte!")
        else:
            print("Oculus Connected!")

        # Receive device specs
        devname = self.sock.recv(64)
        # fd.write(devname)
        self.deviceName = devname.decode("utf-8")

        if not len(self.deviceName):
            raise ConnectionError("Did not recieve Device Name!")
        print("Device Name: "+self.deviceName)
        
        res = self.sock.recv(4)
        #fd.write(res)
        self.WIDTH, self.HEIGHT = struct.unpack(">HH", res)
        print("Oculus WxH: "+str(self.WIDTH)+"x"+str(self.HEIGHT))

        self.bytes_to_read = self.WIDTH * self.HEIGHT * 3
        
        return True

    def start_processing(self, connect_attempts=200):
        # Set up FFmpeg 
        ffmpegCmd = [self.FFMPEG_bin, '-y',
                     '-r', '20', '-i', 'pipe:0',
                     '-vcodec', 'rawvideo',
                     '-pix_fmt', 'rgb24',
                     '-f', 'image2pipe',
                     'pipe:1']
        try:
            self.ffm = subprocess.Popen(ffmpegCmd,
                                        stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE)
        except FileNotFoundError:
            raise FileNotFoundError("Couldn't find FFmpeg at path FFMPEG_bin: "+
                            str(self.FFMPEG_bin))
        self.ffoutthrd = Thread(target=self.stdout_thread,
                                args=())
        self.fferrthrd = Thread(target=self.stderr_thread,
                                args=())
        self.ffinthrd = Thread(target=self.stdin_thread,
                               args=())
        self.ffoutthrd.daemon = True
        self.fferrthrd.daemon = True
        self.ffinthrd.daemon = True

        self.fferrthrd.start()
        time.sleep(0.25)
        self.ffinthrd.start()
        time.sleep(0.25)
        self.ffoutthrd.start()

        print("Waiting on FFmpeg to detect source")
        for i in range(connect_attempts):
            if any(["Output #0, image2pipe" in x for x in self.FFmpeg_info]):
                self.FFMPEGREADY = True
                break
            time.sleep(1)
            print('still waiting on FFmpeg...')
        else:
            print("FFmpeg error?")
            print(''.join(self.FFmpeg_info))
            raise Exception("FFmpeg could not open stream")
        return True


    def recv_image(self, most_recent=True):
        if self.ffoutqueue.empty():
            return None, None
        
        if most_recent:
            frames_skipped = -1
            while not self.ffoutqueue.empty():
                timestamp = time.time()
                frm = self.ffoutqueue.get()
                frames_skipped +=1
        else:
            frm = self.ffoutqueue.get()
        
        frm = np.frombuffer(frm, dtype=np.ubyte)
        frm = frm.reshape((self.HEIGHT, self.WIDTH, 3))
        return frm, timestamp
    
    def stop(self):
        print('Closing the subscriber Oculus socket')
        self.ffm.terminate()
        self.ffm.kill()  
        self.ACTIVE = False
        
        self.fferrthrd.join()
        self.ffinthrd.join()
        self.ffoutthrd.join()