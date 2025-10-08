# Bimanual Hand Pose Bridge Guide

This guide explains how to extract bimanual hand poses from OpenTeach's Meta Quest integration and prepare them for your IK system.

## Quick Start

### 1. Understanding the Data Flow

```
Meta Quest (Unity)
    ↓ (ZMQ PUSH, ports 8087 & 8110)
OculusVRTwoHandDetector (Python)
    ↓ (ZMQ PUB/SUB, port 8088)
Your Bridge Script
    ↓ (ROS messages)
Your IK System
```

### 2. Run the Demo Script

First, make sure OpenTeach is installed:
```bash
pip install -e .
```

Then run the hand pose printer:
```bash
python print_hand_poses.py
```

**Important**: This script expects OpenTeach's VR detector to be running. See "Running the Full Pipeline" below.

## Data Format

### Hand Keypoints Array Structure

Each hand sends an array of **73 floats**:
- **Index 0**: Pose type (0 = absolute world position, 1 = relative to wrist)
- **Index 1-72**: 24 bones × 3 coordinates (x, y, z)

### Bone Layout (24 keypoints per hand)

From `openteach/constants.py`:
```python
OCULUS_JOINTS = {
    'metacarpals': [2, 6, 9, 12, 15],
    'knuckles': [6, 9, 12, 16],
    'thumb': [2, 3, 4, 5, 19],      # [metacarpal, proximal, intermediate, distal, tip]
    'index': [6, 7, 8, 20],          # [metacarpal, proximal, intermediate, tip]
    'middle': [9, 10, 11, 21],       # [metacarpal, proximal, intermediate, tip]
    'ring': [12, 13, 14, 22],        # [metacarpal, proximal, intermediate, tip]
    'pinky': [15, 16, 17, 18, 23]    # [metacarpal, proximal, intermediate, distal, tip]
}
```

**Bone 0** is the wrist position.

## Running the Full Pipeline

### Option 1: Use OpenTeach's Built-in Teleoperation

If you want to see the full OpenTeach system:

1. **Configure network settings** in `configs/network.yaml`:
   ```yaml
   host_address: '172.24.71.206'  # Your server IP
   ```

2. **Start the teleoperation server** with bimanual config:
   ```bash
   python teleop.py robot=bimanual operate=false
   ```
   (Set `operate=false` to skip robot control, just receive VR data)

3. **Connect Meta Quest** to the Unity app and set server IP

4. **Run your bridge script**:
   ```bash
   python print_hand_poses.py
   ```

### Option 2: Minimal Setup (Just VR Data)

Create a minimal script that only runs the VR detector:

```python
from openteach.components.detector.oculusbimanual import OculusVRTwoHandDetector
from multiprocessing import Process

def run_detector():
    detector = OculusVRTwoHandDetector(
        host='0.0.0.0',
        oculus_right_port=8087,   # Unity sends right hand here
        oculus_left_port=8110,    # Unity sends left hand here
        keypoint_pub_port=8088,   # Your bridge subscribes here
        button_port=8095,
        button_publish_port=8093
    )
    detector.stream()

if __name__ == '__main__':
    p = Process(target=run_detector)
    p.start()
    p.join()
```

Then run your `print_hand_poses.py` in another terminal.

## Building Your ROS Bridge

### Minimal ROS Bridge Example

```python
#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import PoseArray, Pose, Point, Quaternion
from openteach.utils.network import ZMQKeypointSubscriber
from openteach.constants import OCULUS_NUM_KEYPOINTS
import numpy as np

def keypoints_to_pose_array(keypoints):
    """Convert OpenTeach keypoints to ROS PoseArray."""
    pose_array = PoseArray()
    pose_array.header.stamp = rospy.Time.now()
    pose_array.header.frame_id = "vr_world"

    # Skip first element (pose type), reshape to (24, 3)
    positions = np.array(keypoints[1:]).reshape(OCULUS_NUM_KEYPOINTS, 3)

    for pos in positions:
        pose = Pose()
        pose.position = Point(x=pos[0], y=pos[1], z=pos[2])
        # Add orientation if needed (from bone directions)
        pose.orientation = Quaternion(x=0, y=0, z=0, w=1)
        pose_array.poses.append(pose)

    return pose_array

def main():
    rospy.init_node('quest_bimanual_bridge')

    # ROS publishers
    pub_right = rospy.Publisher('/vr/right_hand', PoseArray, queue_size=10)
    pub_left = rospy.Publisher('/vr/left_hand', PoseArray, queue_size=10)

    # ZMQ subscribers to OpenTeach
    right_sub = ZMQKeypointSubscriber(
        host='127.0.0.1',
        port=8088,
        topic='right'
    )
    left_sub = ZMQKeypointSubscriber(
        host='127.0.0.1',
        port=8088,
        topic='left'
    )

    rate = rospy.Rate(90)  # 90Hz to match bimanual VR mode

    while not rospy.is_shutdown():
        # Receive from OpenTeach
        right_kp = right_sub.recv_keypoints()
        left_kp = left_sub.recv_keypoints()

        # Convert and publish to ROS
        if right_kp is not None:
            pub_right.publish(keypoints_to_pose_array(right_kp))
        if left_kp is not None:
            pub_left.publish(keypoints_to_pose_array(left_kp))

        rate.sleep()

if __name__ == '__main__':
    main()
```

## Network Port Reference

From `configs/network.yaml`:

| Port  | Purpose                                    |
|-------|--------------------------------------------|
| 8087  | Unity → Python (right hand raw keypoints) |
| 8110  | Unity → Python (left hand raw keypoints)  |
| 8088  | Python detector → Your bridge (both hands)|
| 8095  | Unity → Python (resolution button)        |

## Key Files Reference

- **`openteach/components/detector/oculusbimanual.py`**: VR data receiver and parser
- **`openteach/utils/network.py`**: ZMQ publisher/subscriber classes
- **`openteach/constants.py`**: Bone indices and configuration constants
- **`configs/robot/bimanual.yaml`**: Full bimanual teleoperation config
- **`VR/Bimanual-Robot-Unity/Assets/Scripts/Gesture Detection/GestureDetector.cs`**: Unity hand tracking code

## Tips for Your IK System

1. **Coordinate Systems**: OpenTeach uses Unity's left-handed coordinate system. You may need to transform:
   - Unity: +X right, +Y up, +Z forward
   - ROS (REP-103): +X forward, +Y left, +Z up

2. **Key Bones for IK**: You probably only need:
   - Wrist (bone 0) for end-effector position
   - Fingertips (bones 5, 8, 11, 14, 18) for gripper state
   - Or just wrist + palm orientation if using a simple gripper

3. **Pose Type**: The first element tells you if it's:
   - `0`: Absolute world position (when hands are tracking)
   - `1`: Relative to wrist (less useful for teleoperation)

4. **Frequency**: Data comes at **90Hz** (`BIMANUAL_VR_FREQ` in constants.py) for bimanual mode, matching Quest 2/3's 90Hz display mode. You can downsample if needed.

## Troubleshooting

**"No data received"**
- Check that OpenTeach detector is running
- Verify port numbers match in Unity config and Python
- Check firewall settings if Quest is on different network

**"Connection refused"**
- Make sure `host_address` in configs/network.yaml is correct
- Ensure ZMQ ports aren't blocked

**"Data seems wrong"**
- Check coordinate system transformations
- Verify Unity app is sending to correct IP

## Next Steps

1. Run `print_hand_poses.py` to verify data reception
2. Modify the minimal ROS bridge for your message types
3. Test with your IK system
4. Add filtering/smoothing if needed (OpenTeach has `TransformHandPositionCoords` for this)
