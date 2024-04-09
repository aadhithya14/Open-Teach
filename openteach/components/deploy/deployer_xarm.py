import hydra
import numpy as np
import pickle

from multiprocessing import Process

from openteach.components import Component
from openteach.utils.network import create_response_socket
from openteach.utils.timer import FrequencyTimer
from openteach.constants import DEPLOY_FREQ #, VR_FREQ

class DeployServer(Component):
    def __init__(self, configs):
        self.configs = configs
        
        # Initializing the camera subscribers
        self._init_robot_subscribers()

        # Initializing the sensor subscribers 
        if configs.use_sensor:
            self._init_sensor_subscribers()

        self.deployment_socket = create_response_socket(
            host = self.configs.host_address,
            port = self.configs.deployment_port
        )

        self.timer = FrequencyTimer(DEPLOY_FREQ)

    def _init_robot_subscribers(self):
        robot_controllers = hydra.utils.instantiate(self.configs.robot.controllers)
        self._robots = dict()
        for robot in robot_controllers:
            self._robots[robot.name] = robot

    def _init_sensor_subscribers(self):
        xela_controllers = hydra.utils.instantiate(self.configs.robot.xela_controllers)
        self._sensors = dict()
        self._sensors['xela'] = xela_controllers[0] # There is only 1 controller
    
    def _reset(self):
        for robot in self._robots.keys():
            self._robots[robot].reset()
        self.deployment_socket.send(pickle.dumps(True, protocol = -1))

    def _perform_robot_action(self, robot_action_dict):
        try:

            # Kinova should be applied earlier than allegro
            robot_order = ['xarm'] #['franka', 'allegro']

            for robot in robot_order:
                if robot in robot_action_dict.keys():
                    if robot not in self._robots.keys():
                        print('Robot: {} is an illegal argument.'.format(robot))
                        return False
                    
                    if robot == 'xarm':
                        gripper_action = robot_action_dict[robot]['gripper']
                        cartesian_coords = robot_action_dict[robot]['cartesian']

                        # gripper
                        if gripper_action > 0.25: #0.4: #400: # 0.5:
                            self._robots[robot].set_gripper_state(800)
                        else:
                            self._robots[robot].set_gripper_state(0)
                        
                        # cartesian
                        self._robots[robot].arm_control(cartesian_coords)

                    concat_action = np.concatenate([robot_action_dict[robot]['cartesian'], robot_action_dict[robot]['gripper']])       
                    print('Applying action {} on robot: {}'.format(concat_action, robot))

            # for robot in robot_order:
            #     if robot in robot_action_dict.keys():
            #         if robot not in self._robots.keys():
            #             print('Robot: {} is an illegal argument.'.format(robot))
            #             return False
            #         if robot == 'kinova' or robot == 'franka':
            #             # We use cartesian coords with kinova and not the joint angles
            #             print('Moving the arm in cartesian coords! to: {}'.format(robot_action_dict[robot]))
            #             if robot == 'franka': # Move the arm with a given duration
            #                 # self._robots[robot].move_coords(robot_action_dict[robot], duration=1/DEPLOY_FREQ)
            #                 self._robots[robot].arm_control(robot_action_dict[robot])
            #             else:
            #                 self._robots[robot].move_coords(robot_action_dict[robot])

            #         else: 
            #             print('Moving allegro to given angles')
            #             self._robots[robot].move(robot_action_dict[robot])
                    # print('Applying action {} on robot: {}'.format(robot_action_dict[robot], robot))
            return True
        except:
            print(f'robot: {robot} failed executing in perform_robot_action')
            return False

    def _get_robot_states(self):
        data = dict()
        for robot_name in self._robots.keys():
            if robot_name == 'xarm':
                cartesian_state = self._robots[robot_name].get_cartesian_state()
                gripper_state = self._robots[robot_name].get_gripper_state()
                robot_state = np.concatenate([
                    cartesian_state['position'],
                    cartesian_state['orientation'],
                    gripper_state['position']
                ])
                data[robot_name] = robot_state
            # # Get the cartesian state for kinova
            # elif robot_name == 'kinova' or robot_name == 'franka':
            #     data[robot_name] = self._robots[robot_name].get_cartesian_position() # Get the position only
            #     # print(f'data[{robot_name}].shape: {data[robot_name].shape}')
            # else: # allegro
            #     data[robot_name] = self._robots[robot_name].get_joint_state()
            #     # print(f'data[{robot_name}].shape: {data[robot_name].shape}')
        return data

    def _get_sensor_states(self):
        data = dict() 
        for sensor_name in self._sensors.keys():
            data[sensor_name] = self._sensors[sensor_name].get_sensor_state() # For xela this will be dict {sensor_values: [...], timestamp: [...]}

        return data

    def _send_robot_state(self):
        robot_states = self._get_robot_states()
        print('robot_states: {}'.format(robot_states))
        self.deployment_socket.send(pickle.dumps(robot_states, protocol = -1))

    def _send_sensor_state(self):
        sensor_states = self._get_sensor_states()
        self.deployment_socket.send(pickle.dumps(sensor_states, protocol = -1))

    def _send_both_state(self):
        combined = dict()
        robot_states = self._get_robot_states()
        combined['robot_state'] = robot_states 
        if self.configs.use_sensor:
            sensor_states = self._get_sensor_states()
            combined['sensor_state'] = sensor_states
        self.deployment_socket.send(pickle.dumps(combined, protocol = -1))

    def stream(self):
        self.notify_component_start('robot deployer')
        # self.visualizer_process.start()
        while True:
            # try:
                print('\nListening')
                self.timer.start_loop()
                # robot_action = pickle.loads(self.deployment_socket.recv())
                robot_action = self.deployment_socket.recv()

                if robot_action == b'get_state':
                    print('Requested for robot state information.')
                    self._send_robot_state()
                    continue

                if robot_action == b'get_sensor_state':
                    print('Requested for sensor information.')
                    self._send_sensor_state()
                    continue

                if robot_action == b'reset':
                    print('Resetting the robot.')
                    self._reset()
                    continue

                robot_action = pickle.loads(robot_action)
                print("Received robot action: {}".format(robot_action))
                success = self._perform_robot_action(robot_action)
                print('success: {}'.format(success))
                # More accurate sleep
                
                self.timer.end_loop()

                if success:
                    print('Before sending the states')
                    # self._send_both_state()
                    self._send_robot_state()
                    print('Applied robot action.')
                else:
                    self.deployment_socket.send("Command failed!")
            # except:
            #     print('Illegal values passed. Terminating session.')
            #     break

        # self.visualizer_process.join()
        print('Closing robot deployer component.')
        self.deployment_socket.close()