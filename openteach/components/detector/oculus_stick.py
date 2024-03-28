from openteach.constants import BIMANUAL_VR_FREQ, VR_TCP_HOST, VR_TCP_PORT, VR_CONTROLLER_TOPIC
from openteach.components import Component
from openteach.utils.timer import FrequencyTimer
from openteach.utils.network import create_subscriber_socket, ZMQKeypointPublisher
from openteach.components.detector.utils.controller_state import parse_controller_state

# This class is used to detect the hand keypoints from the VR and publish them.            
class OculusVRStickDetector(Component):
    def __init__(self, 
                 host, 
                controller_state_pub_port
    ):
        self.notify_component_start('vr detector')

        # Create a subscriber socket
        self.stick_socket = create_subscriber_socket(VR_TCP_HOST, VR_TCP_PORT, VR_CONTROLLER_TOPIC) #bytes(VR_CONTROLLER_TOPIC, 'utf-8'))

        # Create a publisher for the controller state
        self.controller_state_publisher = ZMQKeypointPublisher(
            host = host,
            port = controller_state_pub_port
        )
        self.timer = FrequencyTimer(BIMANUAL_VR_FREQ)
    
    # Function to Publish the message
    def _publish_controller_state(self, controller_state):
        self.controller_state_publisher.pub_keypoints(
            keypoint_array = controller_state,
            topic_name = 'controller_state'
        )

    # Function to publish the left/right hand keypoints and button Feedback 
    def stream(self):

        print("oculus stick stream")
        while True:
            try:
                self.timer.start_loop()

                message = self.stick_socket.recv_string()
                if message == "oculus_controller":
                    continue

                controller_state = parse_controller_state(message)

                # Publish message
                self._publish_controller_state(controller_state)

                self.timer.end_loop()

            except KeyboardInterrupt:
                break

        self.controller_state_publisher.stop()
        print('Stopping the oculus keypoint extraction process.')