# Sensors

This codebase also shows promise by having multiple sensors integrated with the codebase showcasing the fact that we can easily integrate whatever sensors we want to add with minimalistic code. 

We have Tactile-uSkin, Realsense Cameras and USB FishEye Cameras already integrated. 

We start the camera sensors using

- [x] Realsense Camera-  You can easily start a realsense camera with our codebase and stream the images inside the Quest 3 to have a different view angle while teleoperating. This gives user a 3D perception of the environment. This camera stream can also be easily recorded and can be used to foundational models for robotics.

1. Just modify the camera numbers to your realsense numbers [here](/configs/camera.yaml/)
2. To start the camera run
   `bash launch_server.sh`
3. You can check the logs to see if all your cameras have been started properly. 
4. If you get this error 
`[ERROR] "RuntimeError: Frame didn't arrive within 5000"`
Just plug out the cable and plug it in back quickly. Follow this [link](https://github.com/IntelRealSense/librealsense/issues/6628) for more instructions.
5. To stream the oculus camera inside the VR set oculus_cam=cam_idx(list index of the cameras connected and added) in  [configs](/configs/camera.yaml)


- [x] FishEye Camera- We have already integrated a USB fisheye camera within our codebase.You can easily start streaming the fisheye by running
To start the camera run
`bash launch_server_fisheye.sh`



