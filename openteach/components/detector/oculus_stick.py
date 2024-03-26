from openteach.constants import VR_FREQ,  ARM_LOW_RESOLUTION, ARM_HIGH_RESOLUTION, ARM_TELEOP_CONT, ARM_TELEOP_STOP , GRIPPER_CLOSE, GRIPPER_OPEN, VR_TCP_HOST, VR_TCP_PORT, VR_CONTROLLER_TOPIC
from openteach.components import Component
from openteach.utils.timer import FrequencyTimer
from openteach.utils.network import create_pull_socket, create_subscriber_socket, ZMQKeypointPublisher
from openteach.components.detector.utils.controller_state import parse_controller_state

# This class is used to detect the hand keypoints from the VR and publish them.            
class OculusVRStickDetector(Component):
    def __init__(self, 
                 host, 
                #  oculus_right_port,
                #  oculus_left_port,
                #  keypoint_pub_port, 
                #  button_port, 
                #  button_publish_port,
                controller_state_pub_port
    ):
        self.notify_component_start('vr detector')
        # # Initializing the network socket for getting the raw keypoints
        # self.raw_keypoint_right_socket = create_pull_socket(host, oculus_right_port)
        # # Initializing the network socket for resolution button feedback
        # self.button_keypoint_socket = create_pull_socket(host, button_port)
        # #self.teleop_reset_socket = create_pull_socket(host, teleop_reset_port)
        # self.raw_keypoint_left_socket= create_pull_socket(host, oculus_left_port)

        # Create a subscriber socket
        self.stick_socket = create_subscriber_socket(VR_TCP_HOST, VR_TCP_PORT, bytes(VR_CONTROLLER_TOPIC, 'utf-8'))

        # ZMQ Keypoint publisher
        # self.hand_keypoint_publisher = ZMQKeypointPublisher(
        #     host = host,
        #     port = keypoint_pub_port
        # )

        # # Publisher socket for button feedback
        # self.button_socket_publisher = ZMQKeypointPublisher(
        #     host = host,
        #     port = button_publish_port
        # ) 
        self.controller_state_publisher = ZMQKeypointPublisher(
            host = host,
            port = controller_state_pub_port
        )
        self.timer = FrequencyTimer(VR_FREQ)


    # # Function to process the data token received from the VR
    # def _process_data_token(self, data_token):
    #     return data_token.decode().strip()

    # # Function to Extract the Keypoints from the String Token sent by the VR
    # def _extract_data_from_token(self, token):        
    #     data = self._process_data_token(token)
    #     information = dict()
    #     keypoint_vals = [0] if data.startswith('absolute') else [1]

    #     # Data is in the format <hand>:x,y,z|x,y,z|x,y,z
    #     vector_strings = data.split(':')[1].strip().split('|')
    #     for vector_str in vector_strings:
    #         vector_vals = vector_str.split(',')
    #         for float_str in vector_vals[:3]:
    #             keypoint_vals.append(float(float_str))
            
    #     information['keypoints'] = keypoint_vals
    #     return information

    # # Function to Publish the right hand transformed Keypoints
    # def _publish_right_data(self, keypoint_dict):
    #     self.hand_keypoint_publisher.pub_keypoints(
    #         keypoint_array = keypoint_dict['keypoints'], 
    #         topic_name = 'right'
    #     )

    # # Function to Publish the left hand transformed Keypoints
    # def _publish_left_data(self, keypoint_dict):
    #     self.hand_keypoint_publisher.pub_keypoints(
    #         keypoint_array= keypoint_dict['keypoints'],
    #         topic_name= 'left'
    #     )

    # # Function to Publish the Resolution Button Feedback
    # def _publish_button_data(self,button_feedback):
    #     self.button_socket_publisher.pub_keypoints(
    #         keypoint_array = button_feedback, 
    #         topic_name = 'button'
    #     )
    
    # Function to Publish the message
    def _publish_controller_state(self, controller_state):
        self.controller_state_publisher.pub_keypoints(
            keypoint_array = controller_state,
            topic_name = 'controller_state'
        )

    # Function to publish the left/right hand keypoints and button Feedback 
    def stream(self):

        while True:
            try:
                self.timer.start_loop()

                # Getting the raw keypoints
                # raw_right_keypoints = self.raw_keypoint_right_socket.recv()
                # raw_left_keypoints = self.raw_keypoint_left_socket.recv()
                button_feedback = self.button_keypoint_socket.recv()
                message = self.stick_socket.recv_string()
                if message == VR_CONTROLLER_TOPIC:
                    continue

                controller_state = parse_controller_state(message)

                # Publish message
                self._publish_controller_state(controller_state)

                # if button_feedback==b'Low':
                #     button_feedback_num = ARM_LOW_RESOLUTION
                # else:
                #     button_feedback_num = ARM_HIGH_RESOLUTION

                # # Processing the keypoints and publishing them
                # keypoint_right_dict = self._extract_data_from_token(raw_right_keypoints)
                # keypoint_left_dict = self._extract_data_from_token(raw_left_keypoints)
                # self._publish_right_data(keypoint_right_dict)
                # self._publish_left_data(keypoint_left_dict)
                # self._publish_button_data(button_feedback_num)
                self.timer.end_loop()

            except KeyboardInterrupt:
                break

        self.controller_state_publisher.stop()
        # self.raw_keypoint_right_socket.close()
        # self.raw_keypoint_left_socket.close()
        # self.hand_keypoint_publisher.stop()
        print('Stopping the oculus keypoint extraction process.')