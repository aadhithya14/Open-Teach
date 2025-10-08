#!/usr/bin/env python3
"""
ROS 2 bridge for bimanual Meta Quest hand poses from OpenTeach.
Subscribes to OpenTeach's ZMQ stream and publishes to ROS 2 topics.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Pose, Point, Quaternion
from std_msgs.msg import Header
from openteach.utils.network import ZMQKeypointSubscriber
from openteach.constants import OCULUS_NUM_KEYPOINTS, BIMANUAL_VR_FREQ
import numpy as np


class QuestBimanualBridge(Node):
    """
    ROS 2 node that bridges Meta Quest hand tracking to ROS 2 topics.
    """

    def __init__(self):
        super().__init__('quest_bimanual_bridge')

        # Declare parameters
        self.declare_parameter('zmq_host', '127.0.0.1')
        self.declare_parameter('zmq_port', 8088)
        self.declare_parameter('publish_rate', 90.0)  # Match BIMANUAL_VR_FREQ
        self.declare_parameter('frame_id', 'vr_world')

        # Get parameters
        zmq_host = self.get_parameter('zmq_host').value
        zmq_port = self.get_parameter('zmq_port').value
        publish_rate = self.get_parameter('publish_rate').value
        self.frame_id = self.get_parameter('frame_id').value

        # Create ROS 2 publishers
        self.pub_right = self.create_publisher(
            PoseArray,
            '/vr/right_hand_poses',
            10
        )
        self.pub_left = self.create_publisher(
            PoseArray,
            '/vr/left_hand_poses',
            10
        )

        # Optional: Publish just wrist poses for simpler IK
        self.pub_right_wrist = self.create_publisher(
            Pose,
            '/vr/right_wrist_pose',
            10
        )
        self.pub_left_wrist = self.create_publisher(
            Pose,
            '/vr/left_wrist_pose',
            10
        )

        # Initialize ZMQ subscribers to OpenTeach
        self.get_logger().info(f'Connecting to OpenTeach at {zmq_host}:{zmq_port}')
        try:
            self.right_sub = ZMQKeypointSubscriber(
                host=zmq_host,
                port=zmq_port,
                topic='right'
            )
            self.left_sub = ZMQKeypointSubscriber(
                host=zmq_host,
                port=zmq_port,
                topic='left'
            )
            self.get_logger().info('✓ Connected to OpenTeach ZMQ stream')
        except Exception as e:
            self.get_logger().error(f'Failed to connect to OpenTeach: {e}')
            raise

        # Create timer for publishing at specified rate
        timer_period = 1.0 / publish_rate  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.get_logger().info(f'Bridge running at {publish_rate} Hz')
        self.get_logger().info('Publishing to:')
        self.get_logger().info('  - /vr/right_hand_poses (all 24 bones)')
        self.get_logger().info('  - /vr/left_hand_poses (all 24 bones)')
        self.get_logger().info('  - /vr/right_wrist_pose (wrist only)')
        self.get_logger().info('  - /vr/left_wrist_pose (wrist only)')

    def keypoints_to_pose_array(self, keypoints):
        """
        Convert OpenTeach keypoints to ROS 2 PoseArray.

        Args:
            keypoints: Array [pose_type, x1,y1,z1, ..., x24,y24,z24]

        Returns:
            PoseArray with 24 poses (one per bone)
        """
        pose_array = PoseArray()
        pose_array.header = Header()
        pose_array.header.stamp = self.get_clock().now().to_msg()
        pose_array.header.frame_id = self.frame_id

        # Skip first element (pose type), reshape to (24, 3)
        positions = np.array(keypoints[1:]).reshape(OCULUS_NUM_KEYPOINTS, 3)

        for pos in positions:
            pose = Pose()
            pose.position = Point(x=float(pos[0]), y=float(pos[1]), z=float(pos[2]))
            # No orientation info in raw keypoints - set identity quaternion
            # You could compute orientation from bone directions if needed
            pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
            pose_array.poses.append(pose)

        return pose_array

    def extract_wrist_pose(self, keypoints):
        """
        Extract just the wrist pose (bone 0) from keypoints.

        Args:
            keypoints: Array [pose_type, x1,y1,z1, ..., x24,y24,z24]

        Returns:
            Pose message for wrist
        """
        pose = Pose()
        # Wrist is the first bone (indices 1,2,3 after skipping pose_type)
        pose.position = Point(
            x=float(keypoints[1]),
            y=float(keypoints[2]),
            z=float(keypoints[3])
        )
        pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        return pose

    def timer_callback(self):
        """
        Timer callback to receive and publish hand data.
        """
        try:
            # Receive from OpenTeach (non-blocking)
            # Note: recv_keypoints() is blocking by default, which matches our timer rate
            right_kp = self.right_sub.recv_keypoints()
            left_kp = self.left_sub.recv_keypoints()

            # Publish right hand
            if right_kp is not None:
                # Full hand skeleton
                right_poses = self.keypoints_to_pose_array(right_kp)
                self.pub_right.publish(right_poses)

                # Just wrist
                right_wrist = self.extract_wrist_pose(right_kp)
                self.pub_right_wrist.publish(right_wrist)

            # Publish left hand
            if left_kp is not None:
                # Full hand skeleton
                left_poses = self.keypoints_to_pose_array(left_kp)
                self.pub_left.publish(left_poses)

                # Just wrist
                left_wrist = self.extract_wrist_pose(left_kp)
                self.pub_left_wrist.publish(left_wrist)

        except Exception as e:
            self.get_logger().error(f'Error in timer callback: {e}')

    def destroy_node(self):
        """Clean up ZMQ connections."""
        self.get_logger().info('Shutting down bridge...')
        try:
            self.right_sub.stop()
            self.left_sub.stop()
        except:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    try:
        bridge = QuestBimanualBridge()
        rclpy.spin(bridge)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'Error: {e}')
    finally:
        if 'bridge' in locals():
            bridge.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
