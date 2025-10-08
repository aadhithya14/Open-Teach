#!/usr/bin/env python3
"""
Simple script to receive and print bimanual hand poses from Meta Quest via OpenTeach.
This demonstrates the data format you'll receive for building a ROS bridge.
"""

import numpy as np
from openteach.utils.network import ZMQKeypointSubscriber
from openteach.constants import OCULUS_NUM_KEYPOINTS, BIMANUAL_VR_FREQ

def print_hand_data(hand_name, keypoints):
    """
    Print hand keypoint data in a readable format.

    Args:
        hand_name: 'left' or 'right'
        keypoints: Array with format [pose_type, x1,y1,z1, x2,y2,z2, ..., x24,y24,z24]
    """
    pose_type = "absolute" if keypoints[0] == 0 else "relative"

    print(f"\n{'='*60}")
    print(f"{hand_name.upper()} HAND - {pose_type} pose")
    print(f"{'='*60}")

    # Extract the 24 3D keypoints (skip first element which is pose type)
    bone_positions = np.array(keypoints[1:]).reshape(OCULUS_NUM_KEYPOINTS, 3)

    # Print wrist position (first bone)
    print(f"Wrist: [{bone_positions[0][0]:.4f}, {bone_positions[0][1]:.4f}, {bone_positions[0][2]:.4f}]")

    # Print a few key finger joints
    finger_names = ['Thumb tip', 'Index tip', 'Middle tip', 'Ring tip', 'Pinky tip']
    finger_indices = [5, 8, 11, 14, 18]  # Based on OCULUS_JOINTS from constants.py

    for name, idx in zip(finger_names, finger_indices):
        pos = bone_positions[idx]
        print(f"{name:12s}: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")

    print(f"Total keypoints: {len(bone_positions)}")


def main():
    """
    Main loop to subscribe to bimanual hand data and print it.
    """
    # Network configuration (from configs/network.yaml)
    HOST = '0.0.0.0'  # Listen on all interfaces
    KEYPOINT_PORT = 8088  # From network.yaml: keypoint_port
    LEFT_KEYPOINT_PORT = 8099  # From network.yaml: transformed_position_left_keypoint_port

    print("Initializing bimanual hand pose subscriber...")
    print(f"Listening for hand data on port {KEYPOINT_PORT} (right) and {LEFT_KEYPOINT_PORT} (left)")
    print(f"Expected data rate: {BIMANUAL_VR_FREQ} Hz (Meta Quest bimanual mode)")
    print("Make sure OpenTeach VR detector is running!")
    print("\nPress Ctrl+C to stop.\n")

    # Create subscribers for both hands
    # Note: The actual port setup depends on how OpenTeach is configured
    # For bimanual, both hands are published on the same port with different topics
    try:
        right_hand_sub = ZMQKeypointSubscriber(
            host='127.0.0.1',  # Connect to localhost (where OpenTeach server publishes)
            port=KEYPOINT_PORT,
            topic='right'
        )

        left_hand_sub = ZMQKeypointSubscriber(
            host='127.0.0.1',
            port=KEYPOINT_PORT,  # Same port, different topic for bimanual
            topic='left'
        )

        print("✓ Subscribers initialized successfully!\n")

    except Exception as e:
        print(f"✗ Failed to initialize subscribers: {e}")
        print("\nMake sure:")
        print("1. OpenTeach is installed: pip install -e .")
        print("2. VR detector is running (from Unity app + Python server)")
        print("3. Network ports in configs/network.yaml match this script")
        return

    # Main loop
    frame_count = 0
    try:
        while True:
            # Receive keypoints from both hands
            # Note: This is blocking - will wait until data arrives
            right_keypoints = right_hand_sub.recv_keypoints()
            left_keypoints = left_hand_sub.recv_keypoints()

            frame_count += 1

            # Print every 45 frames to avoid flooding console (~2Hz at 90Hz VR rate)
            if frame_count % 45 == 0:
                print(f"\n{'#'*60}")
                print(f"Frame {frame_count}")
                print(f"{'#'*60}")

                if right_keypoints is not None:
                    print_hand_data('right', right_keypoints)
                else:
                    print("No right hand data received")

                if left_keypoints is not None:
                    print_hand_data('left', left_keypoints)
                else:
                    print("No left hand data received")

    except KeyboardInterrupt:
        print("\n\nStopping hand pose subscriber...")
        right_hand_sub.stop()
        left_hand_sub.stop()
        print("✓ Stopped successfully!")

    except Exception as e:
        print(f"\n✗ Error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
