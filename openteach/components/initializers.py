import os
import hydra
from abc import ABC
from .recorders.image import RGBImageRecorder, DepthImageRecorder, FishEyeImageRecorder
from .recorders.robot_state import RobotInformationRecord
from .recorders.sim_state import SimInformationRecord
from .recorders.sensors import XelaSensorRecorder
from .sensors import *
from multiprocessing import Process
from openteach.constants import *



class ProcessInstantiator(ABC):
    def __init__(self, configs):
        self.configs = configs
        self.processes = []

    def _start_component(self,configs):
        raise NotImplementedError('Function not implemented!')

    def get_processes(self):
        return self.processes


class RealsenseCameras(ProcessInstantiator):
    """
    Returns all the camera processes. Start the list of processes to start
    the camera stream.
    """
    def __init__(self, configs):
        super().__init__(configs)
        # Creating all the camera processes
        self._init_camera_processes()

    def _start_component(self, cam_idx, cam_serial_num):
        component = RealsenseCamera(
            stream_configs = dict(
                host = self.configs.host_address,
                port = self.configs.cam_port_offset + cam_idx
            ),
            cam_serial_num = cam_serial_num, #self.configs.robot_cam_serial_numbers[cam_idx],
            cam_id = cam_idx + 1,
            cam_configs = self.configs.cam_configs,
            stream_oculus = True if self.configs.oculus_cam == cam_idx else False
        )
        component.stream()

    def _init_camera_processes(self):
        # for cam_idx in range(len(self.configs.robot_cam_serial_numbers)):
        for camera in self.configs.robot_cam_serial_numbers:
            cam_idx = list(camera.keys())[0]
            cam_serial_num = camera[cam_idx]
            self.processes.append(Process(
                target = self._start_component,
                args = (cam_idx, cam_serial_num )
            ))

class FishEyeCameras(ProcessInstantiator):
    """
    Returns all the fish eye camera processes. Start the list of processes to start
    the camera stream.
    """
    def __init__(self, configs):
        super().__init__(configs)
        # Creating all the camera processes
        self._init_camera_processes()

    def _start_component(self, cam_idx, cam_serial_num):
        print('cam_idx: {}, stream_oculus: {}'.format(cam_idx, True if self.configs.oculus_cam == cam_idx else False))
        component = FishEyeCamera(
            cam_index=cam_serial_num, #self.configs.fisheye_cam_numbers[cam_idx],
            stream_configs = dict(
                host = self.configs.host_address,
                port = self.configs.fish_eye_cam_port_offset+ cam_idx,
                set_port_offset = self.configs.fish_eye_cam_port_offset 
            ),
            
            stream_oculus = True if self.configs.stream_oculus and self.configs.oculus_cam == cam_idx else False,
            
        )
        component.stream()

    def _init_camera_processes(self):
        # for cam_idx in range(len(self.configs.fisheye_cam_numbers)):
        for camera in self.configs.fisheye_cam_numbers:
            cam_idx = list(camera.keys())[0]
            cam_serial_num = camera[cam_idx]
            self.processes.append(Process(
                target = self._start_component,
                args = (cam_idx, cam_serial_num, )
            ))


class TeleOperator(ProcessInstantiator):
    """
    Returns all the teleoperation processes. Start the list of processes 
    to run the teleop.
    """
    def __init__(self, configs):
        super().__init__(configs)
      
        # For Simulation environment start the environment as well
        if configs.sim_env:
            self._init_sim_environment()
        # Start the Hand Detector
        self._init_detector()
        # Start the keypoint transform
        # self._init_keypoint_transform()
        # self._init_visualizers()


        if configs.operate: 
            self._init_operator()
        
    #Function to start the components
    def _start_component(self, configs):    
        component = hydra.utils.instantiate(configs)
        component.stream()

    #Function to start the detector component
    def _init_detector(self):
        self.processes.append(Process(
            target = self._start_component,
            args = (self.configs.robot.detector, )
        ))
        print("Detector init")

    #Function to start the sim environment
    def _init_sim_environment(self):
         for env_config in self.configs.robot.environment:
            self.processes.append(Process(
                target = self._start_component,
                args = (env_config, )
            ))

    #Function to start the keypoint transform
    def _init_keypoint_transform(self):
        for transform_config in self.configs.robot.transforms:
            self.processes.append(Process(
                target = self._start_component,
                args = (transform_config, )
            ))

    #Function to start the visualizers
    def _init_visualizers(self):
       
        for visualizer_config in self.configs.robot.visualizers:
            self.processes.append(Process(
                target = self._start_component,
                args = (visualizer_config, )
            ))
        # XELA visualizer
        if self.configs.run_xela:
            for visualizer_config in self.configs.xela_visualizers:
                self.processes.append(Process(
                    target = self._start_component,
                    args = (visualizer_config, )
                ))

    #Function to start the operator
    def _init_operator(self):
        for operator_config in self.configs.robot.operators:
            
            self.processes.append(Process(
                target = self._start_component,
                args = (operator_config, )

            ))

    
# Data Collector Class
class Collector(ProcessInstantiator):
    """
    Returns all the recorder processes. Start the list of processes 
    to run the record data.
    """
    def __init__(self, configs, demo_num, depth=False):
        super().__init__(configs)
        self.demo_num = demo_num
        self.depth = depth
        self._storage_path = os.path.join(
            self.configs.storage_path, 
            'demonstration_{}'.format(self.demo_num)
        )
       
        self._create_storage_dir()
        self._init_camera_recorders()
        # Initializing the recorders
        if self.configs.sim_env is True:
            self._init_sim_recorders()
        else:
            print("Initialising robot recorders")
            self._init_robot_recorders()
        
        
        if self.configs.is_xela is True:
            self._init_sensor_recorders()

    def _create_storage_dir(self):
        if os.path.exists(self._storage_path):
            return 
        else:
            os.makedirs(self._storage_path)

    #Function to start the components
    def _start_component(self, component):
        component.stream()

    # Record the rgb components
    def _start_rgb_component(self, cam_idx=0):
        # This part has been isolated and made different for the sim and real robot
        # If using simulation and real robot on the same network, only one of them will stream into the VR. Close the real robot realsense camera stream before launching simulation.
        if self.configs.sim_env is False:
            print("RGB function")
            component = RGBImageRecorder(
                host = self.configs.host_address,
                image_stream_port = self.configs.cam_port_offset + cam_idx,
                storage_path = self._storage_path,
                filename = 'cam_{}_rgb_video'.format(cam_idx)
            )
        else:
            print("Reaching correct function")
            component = RGBImageRecorder(
            host = self.configs.host_address,
            image_stream_port = self.configs.sim_image_port+ cam_idx,
            storage_path = self._storage_path,
            filename = 'cam_{}_rgb_video'.format(cam_idx),
            sim = True
        )
        component.stream()

    # Record the depth components
    def _start_depth_component(self, cam_idx):
        if self.configs.sim_env is not True:
            component = DepthImageRecorder(
                host = self.configs.host_address,
                image_stream_port = self.configs.cam_port_offset + cam_idx + DEPTH_PORT_OFFSET,
                storage_path = self._storage_path,
                filename = 'cam_{}_depth'.format(cam_idx)
            )
        else:
            component = DepthImageRecorder(
                host = self.configs.host_address,
                image_stream_port = self.configs.sim_image_port + cam_idx + DEPTH_PORT_OFFSET,
                storage_path = self._storage_path,
                filename = 'cam_{}_depth'.format(cam_idx)
            )
        component.stream()

    #Function to start the camera recorders
    def _init_camera_recorders(self):
        if self.configs.sim_env is not True:
            print("Camera recorder starting")
            cam_idx = 0
            # for cam_idx in range(len(self.configs.robot_cam_serial_numbers)):
            for camera in self.configs.robot_cam_serial_numbers:
                cam_idx = list(camera.keys())[0]
                self.processes.append(Process(
                    target = self._start_rgb_component,
                    args = (cam_idx, )
                ))

                if self.depth:
                    self.processes.append(Process(
                        target = self._start_depth_component,
                        args = (cam_idx, )
                    ))

            import yaml
            with open('configs/fisheyecamera.yaml') as file:
                fish_eye_cam_configs = yaml.load(file, Loader=yaml.FullLoader)
            # for fisheye_cam_idx in range(len(fish_eye_cam_configs['fisheye_cam_numbers'])):
            for camera in fish_eye_cam_configs['fisheye_cam_numbers']:
                fisheye_cam_idx = list(camera.keys())[0]
                self.processes.append(Process(
                    target = self._start_fish_eye_component,
                    # args = (cam_idx + 1 + fisheye_cam_idx, )
                    args = (fisheye_cam_idx, )
                ))
        else:
          
            for cam_idx in range(self.configs.num_cams):
                self.processes.append(Process(
                    target = self._start_rgb_component,
                    args = (cam_idx, )
                ))

                if self.depth:
                    self.processes.append(Process(
                        target = self._start_depth_component,
                        args = (cam_idx, )
                    ))

    #Function to start the sim recorders
    def _init_sim_recorders(self):
        port_configs = self.configs.robot.port_configs
        for key in self.configs.robot.recorded_data[0]:
            self.processes.append(Process(
                        target = self._start_sim_component,
                        args = (port_configs[0],key)))

    #Function to start the xela sensor recorders
    def _start_xela_component(self,
        controller_config
    ):
        component = XelaSensorRecorder(
            controller_configs=controller_config,
            storage_path=self._storage_path
        )
        component.stream()

    #Function to start the sensor recorders
    def _init_sensor_recorders(self):
        """
        For the XELA sensors or any other sensors
        """
        for controller_config in self.configs.robot.xela_controllers:
            self.processes.append(Process(
                target = self._start_xela_component,
                args = (controller_config, )
            ))

    #Function to start the fish eye recorders
    def _start_fish_eye_component(self, cam_idx):
        component = FishEyeImageRecorder(
            host = self.configs.host_address,
            image_stream_port = self.configs.fish_eye_cam_port_offset + cam_idx,
            storage_path = self._storage_path,
            filename = 'cam_{}_rgb_video'.format(cam_idx)
            # filename = 'cam_{}_rgb_video'.format(cam_idx + 1 + fish_eye_cam_idx)
        )
        component.stream()

    #Function to start the robot recorders
    def _start_robot_component(
        self, 
        robot_configs, 
        recorder_function_key):
        component = RobotInformationRecord(
            robot_configs = robot_configs,
            recorder_function_key = recorder_function_key,
            storage_path = self._storage_path
        )

        component.stream()

    #Function to start the sim recorders
    def _start_sim_component(self,port_configs, recorder_function_key):
        component = SimInformationRecord(
                   port_configs = port_configs,
                   recorder_function_key= recorder_function_key,
                   storage_path=self._storage_path
        )
        component.stream()

    #Function to start the robot recorders
    def _init_robot_recorders(self):
        # Instantiating the robot classes
        for idx, robot_controller_configs in enumerate(self.configs.robot.controllers):
            for key in self.configs.robot.recorded_data[idx]:
                self.processes.append(Process(
                    target = self._start_robot_component,
                    args = (robot_controller_configs, key, )
                ))


    

   