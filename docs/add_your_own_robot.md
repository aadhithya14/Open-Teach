# Add your own robot

To add any robot manipulator/simulation to Open-Teach, all you need to write are few wrappers.

1. For Robot Manipulator

- [x] ROS-LINK WRAPPER: You can any robot to Open-Teach by adding three wrappers. First add a python file to ros_links directory which will have DexArmControl class. For an example of any manipulator you can look into one of the codefiles within this directory. This will establish ros_link between the controllers and openteach. Check the template file [here](https://github.com/aadhithya14/Open-Teach/blob/main/openteach/ros_links/ros_link.py). 

- [x] ROBOT WRAPPER: You have to write a wrapper for any robot for getting the basic information of the robot and to send information to the robot. For an example on how to create this robot wrapper check [here](https://github.com/aadhithya14/Open-Teach/blob/main/openteach/robot/robot.py).

- [x] OPERATOR: This is the wrapper which does helps in the teleoperation . For creating a new wrapper for this ,most parts in this file can be reused from any of the manipulator codefiles in [here](https://github.com/aadhithya14/Open-Teach/tree/main/openteach/components/operators/template.py). The transformations might have to be tuned accordingly as per your requirements. 
    - [x] Velocity Control - Kinova file in the operator directory uses velocity control and shows how you can integrate a robot with velocity control into openteach.

    - [x] Position Control - Franka and Xarm in the operator directory uses position control and shows how you can integrate any robot with position control with openteach.


2. For any manipulator/hand simulation

- [x] Hand Simulation- You can inherit [hand_env](https://github.com/aadhithya14/Open-Teach/blob/main/openteach/components/environment/hand_env.py) from environment directory while adding any new hand. Similarly you can create your operator like allegro_sim operator using most of the components of it. Currently the simulation supports allegro hand right and curved xela hand right. We are planning to expand this to multiple robots with a unified framework. For adding a new robot hand sim , you have to add urdfs  

- [x] Arm simulation suite- You can inherit [arm_env](https://github.com/aadhithya14/Open-Teach/blob/main/openteach/components/environment/arm_env.py) from environment directory if you want to add any new environment.
Similarly you can create your own operator like libero_sim operator using most components of it. Libero sim uses relative pose as actions, but you can choose to use position control in the new wrapper which you want to add. The comments on how to use any simulation arm with position control is mentioned in the comments of libero_sim.

3. Configs

- You need to add a robot/sim config to use a new robot/simulation. Follow the template [config](https://github.com/aadhithya14/Open-Teach/blob/main/configs/robot/template_robot.yaml) for a single arm. Follow [config](https://github.com/aadhithya14/Open-Teach/blob/main/configs/robot/bimanual.yaml) for bimanual arm.