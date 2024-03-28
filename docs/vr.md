# VR Application Installation.

This can be done in two ways.

1. You can install the application directly into your Quest using the APK file we provide using SideQuest.
2. Inorder to have more flexibility and modify the VR APK we have also released the source code corresponding to it. The APK is made in Unity 2021.3.5f1. To ensure clean builds with no errors using this version is recommended.
3. For building the apk from source , add the zipped code files to Unity and within the settings in File menu run Build and Run. The targeted device in Build settings should be the Quest you are connecting to. Also users might need Meta account with Meta Quest Developer Hub and will need to enable Developer mode for the Meta Quest Application. 
4. To setup the VR Application in Oculus Headset, enter the IP Address of the robot server. The robot server and the Oculus Headset should be in the same network. 

## User Interface

## Single Robot Arm + Robot Hand: 

​	Since the Robot Hand is a Right Hand, the user controls to switch modes are in Left Hand to have seemless keypoint stream happening from the Right Hand. 

| Pinch ( Left Hand) | Mode                 | Stream Border Color |
| ------------------ | -------------------- | ------------------- |
| Index Pinch        | Only Hand Mode       | Green               |
| Middle Pinch       | Arm + Hand Mode      | Blue                |
| Ring Pinch         | Pause                | Red                 |
| Pinky Pinch        | Resolution Selection | Black               |

**Note: Here the teleoperation is just like mimicking the human hand and arm actions**

## Multi Robot Arm (Bimanual):

Since both the hands are being used here for teleoperation and gripper mode selection we use pinches in both the hands. Due to the noise in hand pose detection while moving the hands , for better detection of pinches we use keypoint distance threshold based approach between two fingers. For our setup we use Xarms as bimanual robots.

| Pinch (On Both Hands) | Mode                                                         | Stream Border Color |
| ----------------------- | ------------------------------------------------------------ | ------------------- |
| Index Pinch             | Start the Teleop ( Only used at the start of the teleoperation ) | Green               |
| Middle Pinch            | Pause/Resume the Robot                                       | Red                 |
| Ring Pinch              | Pause/Resume the Robot                                       | Red                 |
| Pinky Pinch             | Gripper Open/Close                                           | Yellow              |


**Note: Here the teleoperation is not mimicking the arm actions. Like other bimanual teleoperation methods we imagine we are holding the end effector of the arm and rotating and translating accordingly**


The VR APK files are available [here](/VR/APK/).

After you install the APK file. You will be prompted with a blank screen with red border with a Menu button on it. Click the Menu button (Ensure you have Hand tracking enabled in the Oculus.), you will see IP: Not Defined. Just Click on Change IP and enter the IP using the dropdown (The VR and the Robot should be under the same network provider). Once the IP is enter go back to the screen where you clicked Change IP and Click Stream. The screen border will become green and  your App is ready to stream the keypoints.

#### Note: Remember to enter your same IP on the server host address variable [config](/configs/network.yaml)

Once finished setting up the APK proceed to [teleop](/docs/teleop_data_collect.md).

If Teleoperation server is not started, the APK will work for sometime and stop as there are ports to send the information to. 