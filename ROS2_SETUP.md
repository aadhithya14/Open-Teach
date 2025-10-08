# ROS 2 Bridge Setup Guide

This guide shows how to set up the ROS 2 bridge for bimanual Meta Quest hand tracking.

## Prerequisites

- ROS 2 (Humble, Iron, or Jazzy recommended)
- OpenTeach installed: `pip install -e .`
- Python 3.8+

## Installation

### 1. Install ROS 2 Python dependencies

```bash
pip install rclpy
```

Or if you have a full ROS 2 workspace:
```bash
source /opt/ros/humble/setup.bash  # or your ROS 2 distro
```

### 2. Make the bridge executable

```bash
chmod +x ros2_bridge_bimanual.py
```

## Running the Bridge

### Step 1: Start OpenTeach VR Detector

You need the OpenTeach detector to receive data from the Meta Quest:

**Option A: Minimal detector only**
```python
# minimal_detector.py
from openteach.components.detector.oculusbimanual import OculusVRTwoHandDetector

detector = OculusVRTwoHandDetector(
    host='0.0.0.0',
    oculus_right_port=8087,
    oculus_left_port=8110,
    keypoint_pub_port=8088,
    button_port=8095,
    button_publish_port=8093
)
detector.stream()
```

Run it:
```bash
python minimal_detector.py
```

**Option B: Full OpenTeach teleoperation (without robot control)**
```bash
python teleop.py robot=bimanual operate=false
```

### Step 2: Connect Meta Quest

1. Install the OpenTeach Unity app on your Quest (APK in `VR/` folder)
2. Launch the app on Quest
3. Enter your server IP address in the VR menu
4. Start streaming

### Step 3: Start ROS 2 Bridge

In a new terminal:

```bash
# Source ROS 2
source /opt/ros/humble/setup.bash

# Run the bridge
python3 ros2_bridge_bimanual.py
```

Or with custom parameters:
```bash
python3 ros2_bridge_bimanual.py \
    --ros-args \
    -p zmq_host:=127.0.0.1 \
    -p zmq_port:=8088 \
    -p publish_rate:=90.0 \
    -p frame_id:=vr_world
```

## ROS 2 Topics Published

The bridge publishes to these topics:

| Topic                    | Type                     | Description                           |
|--------------------------|--------------------------|---------------------------------------|
| `/vr/right_hand_poses`   | `geometry_msgs/PoseArray`| All 24 bones of right hand            |
| `/vr/left_hand_poses`    | `geometry_msgs/PoseArray`| All 24 bones of left hand             |
| `/vr/right_wrist_pose`   | `geometry_msgs/Pose`     | Right wrist position only             |
| `/vr/left_wrist_pose`    | `geometry_msgs/Pose`     | Left wrist position only              |

### Viewing the data

```bash
# List topics
ros2 topic list

# Echo right hand data
ros2 topic echo /vr/right_hand_poses

# Echo just wrist positions
ros2 topic echo /vr/right_wrist_pose

# Check publishing frequency
ros2 topic hz /vr/right_hand_poses
```

### Visualize in RViz2

```bash
# Launch RViz2
rviz2

# Add displays:
# 1. Set Fixed Frame to "vr_world"
# 2. Add → By topic → /vr/right_hand_poses → PoseArray
# 3. Add → By topic → /vr/left_hand_poses → PoseArray
```

## Integration with Your IK System

### Option 1: Subscribe to Wrist Poses Only

If you just need end-effector positions:

```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose

class IKController(Node):
    def __init__(self):
        super().__init__('ik_controller')

        self.create_subscription(
            Pose,
            '/vr/right_wrist_pose',
            self.right_wrist_callback,
            10
        )
        self.create_subscription(
            Pose,
            '/vr/left_wrist_pose',
            self.left_wrist_callback,
            10
        )

    def right_wrist_callback(self, msg):
        # Send to your IK solver
        target_pos = [msg.position.x, msg.position.y, msg.position.z]
        # your_ik.solve_right_arm(target_pos)
        pass

    def left_wrist_callback(self, msg):
        # Send to your IK solver
        target_pos = [msg.position.x, msg.position.y, msg.position.z]
        # your_ik.solve_left_arm(target_pos)
        pass
```

### Option 2: Subscribe to Full Hand Skeleton

If you need finger/gripper state:

```python
from geometry_msgs.msg import PoseArray

class IKController(Node):
    def __init__(self):
        super().__init__('ik_controller')

        self.create_subscription(
            PoseArray,
            '/vr/right_hand_poses',
            self.right_hand_callback,
            10
        )

    def right_hand_callback(self, msg):
        # msg.poses[0] = wrist
        # msg.poses[5] = thumb tip
        # msg.poses[8] = index tip
        # msg.poses[11] = middle tip
        # msg.poses[14] = ring tip
        # msg.poses[18] = pinky tip

        wrist = msg.poses[0].position
        thumb_tip = msg.poses[5].position

        # Calculate gripper state from thumb-index distance
        index_tip = msg.poses[8].position
        gripper_width = self.distance(thumb_tip, index_tip)

        # your_ik.solve_right_arm(wrist, gripper_width)

    def distance(self, p1, p2):
        import math
        return math.sqrt(
            (p1.x - p2.x)**2 +
            (p1.y - p2.y)**2 +
            (p1.z - p2.z)**2
        )
```

## Coordinate Frame Transformations

### Unity (Meta Quest) Frame
- +X: Right
- +Y: Up
- +Z: Forward

### ROS 2 REP-103 Standard Frame
- +X: Forward
- +Y: Left
- +Z: Up

If you need to transform coordinates:

```python
def unity_to_ros(unity_pos):
    """Convert Unity coordinates to ROS."""
    return [
        unity_pos[2],   # Unity Z → ROS X (forward)
        -unity_pos[0],  # Unity -X → ROS Y (left)
        unity_pos[1]    # Unity Y → ROS Z (up)
    ]
```

You can add this transformation in the bridge's `keypoints_to_pose_array()` method if needed.

## Troubleshooting

**Bridge fails to connect to ZMQ**
```bash
# Check if OpenTeach detector is running
ps aux | grep python | grep detector

# Verify port is listening
netstat -an | grep 8088
```

**No data on ROS topics**
```bash
# Check bridge is publishing
ros2 topic list

# Check for errors
ros2 run ros2_bridge_bimanual --ros-args --log-level DEBUG
```

**Low frame rate**
```bash
# Check actual publishing rate
ros2 topic hz /vr/right_hand_poses

# Should see ~90 Hz. If lower, check:
# 1. Meta Quest is connected
# 2. OpenTeach detector is running
# 3. Network latency
```

**Coordinate frame issues**
- Add the coordinate transform in the bridge code
- Or use ROS 2 `tf2` to publish transforms
- Make sure your IK system expects the same coordinate frame

## Performance Tips

1. **Reduce publishing rate** if 90Hz is too much:
   ```bash
   python3 ros2_bridge_bimanual.py --ros-args -p publish_rate:=30.0
   ```

2. **Subscribe only to topics you need** (e.g., just wrist poses)

3. **Use QoS profiles** for real-time:
   ```python
   from rclpy.qos import QoSProfile, ReliabilityPolicy

   qos = QoSProfile(
       depth=1,
       reliability=ReliabilityPolicy.BEST_EFFORT
   )

   self.create_publisher(PoseArray, '/vr/right_hand_poses', qos)
   ```

## Next Steps

1. Run `python3 print_hand_poses.py` first to verify data reception
2. Start the ROS 2 bridge
3. Verify topics with `ros2 topic list` and `ros2 topic echo`
4. Integrate with your IK system
5. Add coordinate transforms if needed
6. Tune publishing rate for your application
