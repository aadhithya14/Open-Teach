from abc import ABC, abstractmethod

import numpy as np
#from utils import clamp, AssetDesc
import math
import hydra
from copy import copy
import gym
from gym.spaces import Box
#import torch
from openteach.components import Component
from openteach.utils.timer import FrequencyTimer
from openteach.utils.network import ZMQCameraPublisher, ZMQCompressedImageTransmitter,ZMQKeypointPublisher,ZMQKeypointSubscriber
from openteach.components.environment.hand_env import Hand_Env
from openteach.constants import *


import cv2 
import os
# This is added to avoid any errors in case of multiple GPUs. This part specifically assigns the first GPU and uses it as compute device and graphics device.
os.environ['MESA_VK_DEVICE_SELECT'] = '10de:24b0'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from isaacgym import gymapi, gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import *
import time
import torch



class AllegroHandEnv(Hand_Env):
        def __init__(self,
                     host,
                     camport,
                     jointanglepublishport,
                     jointanglesubscribeport,
                     timestamppublisherport,
                     endeff_publish_port,
                     endeffpossubscribeport,
                     actualanglepublishport,
                     stream_oculus,
                     num_per_row = 1,
                     spacing = 2.5,
                     show_axis=0,
                     env_suite='cube_flipping',
                     control_mode= 'Position_Velocity',
                     object= 'block',
                     asset = 'allegro_hand'):
               

            # Define timer, network IP and ports
            self._timer=FrequencyTimer(CAM_FPS_SIM)
            self.host=host
            self.camport=camport
            self.jointanglepublishport=jointanglepublishport
            self.jointanglesubscribeport=jointanglesubscribeport
            self._stream_oculus = stream_oculus

            #Define ZMQ pub/sub
            #Port for publishing rgb images.
            self.rgb_publisher = ZMQCameraPublisher(
                    host = host,
                    port = camport
            )
            
            #Publishing the stream into the oculus.
            if self._stream_oculus:
                    self.rgb_viz_publisher = ZMQCompressedImageTransmitter(
                            host = host,
                            port = camport + VIZ_PORT_OFFSET
                    )

            #Publisher for Depth data
            self.depth_publisher = ZMQCameraPublisher(
                    host = host,
                    port = camport + DEPTH_PORT_OFFSET 
            )


            #Publisher for Joint Angle 
            self.joint_angle_publisher = ZMQKeypointPublisher(
                    host = host,
                    port = jointanglepublishport
            )

            #Publisher for Actual Current Joint Angles
            self.actualanglepublisher = ZMQKeypointPublisher(
                    host = host,
                    port = actualanglepublishport
            )

            #Publisher for calculated angles from teleoperator.
            self.joint_angle_subscriber = ZMQKeypointSubscriber(
                    host=host,
                    port= jointanglesubscribeport,
                    topic='desired_angles'
            )

            #Publisher for endeffector Positions 
            self.endeff_publisher = ZMQKeypointPublisher(
                    host = host,
                    port = endeff_publish_port
            )

            #Publisher for endeffector Velocities
            self.endeff_pos_subscriber = ZMQKeypointSubscriber(
                    host = host,
                    port = endeffpossubscribeport,
                    topic='endeff_coords'
            )

            #Publisher for timestamps
            self.timestamp_publisher = ZMQKeypointPublisher(
                    host=host,
                    port=timestamppublisherport
            )

            
            self.physics_engine=gymapi.SIM_PHYSX
            self.gym=gymapi.acquire_gym()
            self.num_per_row=num_per_row
            self.spacing=spacing
            self.show_axis=show_axis
            self.name="Allegro_Sim"
            
            #Env specific parameters
            self.env_lower = gymapi.Vec3(-self.spacing, 0.0, -self.spacing)
            self.env_upper = gymapi.Vec3(self.spacing, self.spacing, self.spacing)
            self.actor_handles=[]
            self.object_indices=[]
            self.actor_indices=[]
            self.table_handles=[]
            
            self.env_suite=env_suite 
            self.control_mode=control_mode
            self.asset_name=asset
        
            # set common parameters
            sim_params = gymapi.SimParams()
            sim_params.dt = self.dt= 1/60
            sim_params.substeps = 2
            sim_params.up_axis = gymapi.UP_AXIS_Z
            sim_params.gravity = gymapi.Vec3(0.0, -9.8, 0)
            
            # set PhysX-specific parameters
            if self.physics_engine==gymapi.SIM_PHYSX:
                sim_params.physx.use_gpu = True
                sim_params.physx.solver_type = 1
                sim_params.physx.num_position_iterations = 6
                sim_params.physx.num_velocity_iterations = 1
                sim_params.physx.contact_offset = 0.01
                sim_params.physx.rest_offset = 0.0
                self.compute_device_id=0
                self.graphics_device_id=0
                self.asset_id=1

            # set Flex-specific parameters
            elif self.physics_engine==gymapi.SIM_FLEX:
                sim_params.flex.solver_type = 5
                sim_params.flex.num_outer_iterations = 4
                sim_params.flex.num_inner_iterations = 20
                sim_params.flex.relaxation = 0.8
                sim_params.flex.warm_start = 0.5
                self.compute_device_id=0
                self.graphics_device_id=0
                    
            # create sim with these parameters
            print("Trying to make simulation")
            self.sim = self.gym.create_sim(self.compute_device_id,self.graphics_device_id, self.physics_engine, sim_params)
            print("Simulation created")
            # Add ground
            plane_params = gymapi.PlaneParams()
            self.gym.add_ground(self.sim, plane_params)

            # create viewer (Can be used if you want to visualise the simulation)
            #self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())

            #Get the camera pose and place the camera there

            # if self.viewer is None:
            #         print("*** Failed to create viewer")
            #         quit()

            # set asset options
            asset_options = gymapi.AssetOptions()
            asset_options.fix_base_link = True
            asset_options.flip_visual_attachments =  False 
            asset_options.use_mesh_materials = True
            asset_options.disable_gravity = True
            
            table_asset_options = gymapi.AssetOptions()
            table_asset_options.fix_base_link = True
            table_asset_options.flip_visual_attachments = False
            table_asset_options.collapse_fixed_joints = True
            table_asset_options.disable_gravity = False

            # get asset file
            asset_root = os.path.join(os.path.dirname(__file__), "assets/urdf/")
            
            asset_file_dict={
                        "allegro_hand": "allegro_hand_description/urdf/model_only_hand.urdf",
                        "allegro_hand_curved": "allegro_hand_description/urdf/allegro_hand_curved.urdf",
            }
            asset_file = asset_file_dict[self.asset_name]
            table_asset_file= "allegro_hand_description/urdf/table.urdf"
            asset_files_dict = {
                    "block": "objects/cube_multicolor.urdf",
                    "egg": "mjcf/open_ai_assets/hand/egg.xml",
                    "pen": "mjcf/open_ai_assets/hand/pen.xml",
                    "wrench": "wrench/foam_wrench.urdf",
                    "rod": "objects/rod.urdf",
                    "can":"ycb/010_potted_meat_can/010_potted_meat_can.urdf",
                    "eraser":"allegro_hand_description/urdf/eraser.urdf",
                    "banana":"ycb/011_banana/011_banana.urdf",
                    "mug":"ycb/025_mug/025_mug.urdf",
                    "brick":"ycb/061_foam_brick/061_foam_brick.urdf"
            }
            object_asset_file = asset_files_dict[object]
            print("Loading asset '%s' from '%s'" % (asset_file, asset_root)) 
            # Load the assets 
            self.asset = self.gym.load_urdf(self.sim, asset_root, asset_file, asset_options)
            self.table_asset = self.gym.load_urdf(self.sim, asset_root, table_asset_file, table_asset_options)            
            # Loads the urdf according to the object
            object_asset_options = gymapi.AssetOptions()
            self.object_asset= self.gym.load_urdf(self.sim, asset_root, object_asset_file,object_asset_options)
            
            self.num_dofs=self.get_dof_count()
                      
            object_position, object_rotation=self.create_env()
            self.actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
            
            self.root_state_tensor = gymtorch.wrap_tensor(self.actor_root_state_tensor)

            #Set Root State Tensors
            self.root_state_tensor = self.root_state_tensor.view(-1, 13)
            self.object_indices=to_torch(self.object_indices, dtype=torch.int32,device='cpu')
            state=self.reset(object_position=object_position,object_rotation=object_rotation)

        #Function creates the environment
        def create_env(self):
            print("Loading Assets")
            self.camera_handles = []
            self.object_handles=[]
            camera_position_dict = {
                    "cube_flipping": gymapi.Vec3(1.06,1.6 , -0.02),
                    "cube_rotating": gymapi.Vec3(1.06,1.6 , -0.02),
                    "can_picking": gymapi.Vec3(0.8,1, 0.01),
                    "sponge_flipping": gymapi.Vec3(0.8,1, 0.01),
                    "eraser_turning": gymapi.Vec3(0.8,1, 0.01),
                    "banana": gymapi.Vec3(1.3,1.5 , 0.01),
                    "pinch_grasping": gymapi.Vec3(0.8,1, 0.01)
            }

            camera_target_dict = {
                    "cube_flipping": gymapi.Vec3(1.03,1.3 , -0.02),
                    "cube_rotating": gymapi.Vec3(1.03,1.3 , -0.02),
                    "can_picking": gymapi.Vec3(1,0.9, 0.01),
                    "sponge_flipping": gymapi.Vec3(1,0.9, 0.01),
                    "eraser_turning": gymapi.Vec3(1,0.9, 0.01),
                    "banana": gymapi.Vec3(1,1.3 , 0.01),
                    "pinch_grasping": gymapi.Vec3(1,0.9, 0.01)
            }

            actor_position = {
                    "cube_flipping": gymapi.Vec3(1,1.2,0),
                    "cube_rotating": gymapi.Vec3(1,1.2,0),
                    "can_picking": gymapi.Vec3(1,0.95,0),
                    "banana": gymapi.Vec3(1,1.2,0),
                    "sponge_flipping": gymapi.Vec3(1,0.88,0),
                    "eraser_turning": gymapi.Vec3(1,0.9,0),
                    "pinch_grasping": gymapi.Vec3(1,0.93,0),
            }
            actor_rotation = {
                    "cube_flipping": gymapi.Quat(-0.707,-0.707, 0,0),
                    "cube_rotating": gymapi.Quat(-0.707,-0.707, 0,0),
                    "can_picking": gymapi.Quat(-0.707,0.707, 0,0),
                    "banana": gymapi.Quat(-1.54,-0.707, 0,0),
                    "sponge_flipping": gymapi.Quat(-0.707,0.707, 0,0),
                    "eraser_turning": gymapi.Quat(-0.707,0.707, 0,0),
                    "pinch_grasping": gymapi.Quat(-0.707,0.707, 0,0)
            }

            object_pose= gymapi.Transform()

            object_pos = {
                    "cube_flipping": gymapi.Vec3(1, 1.3, 0.06),
                    "cube_rotating": gymapi.Vec3(1.1, 1.3, 0.03),
                    "can_picking": gymapi.Vec3(0.9, 0.9, 0),
                    "sponge_flipping": gymapi.Vec3(0.98, 0.82, 0),
                    "eraser_turning": gymapi.Vec3(0.99, 0.82, 0),
                    "banana": gymapi.Vec3(1., 1.3, 0.01),
                    "pinch_grasping": gymapi.Vec3(0.94, 0.85, 0)
            }

            object_rot = {
                    "cube_flipping": gymapi.Quat(-1.3, -0.707, 0, 0),
                    "cube_rotating": gymapi.Quat(-1.54, -0.707, 0, 0),
                    "can_picking": gymapi.Quat(-0.707, 0.707, 0, 0),
                    "sponge_flipping": gymapi.Quat(-0.707, 0.707, 0.707, 0.3),
                    "eraser_turning": gymapi.Quat(-0.707, 0.707, 0.707, 0.3),
                    "banana": gymapi.Quat(-1.54, -0.707, 0, 0),
                    "pinch_grasping": gymapi.Quat(-0.707, -0.707, 0, 0)
            }

    
            self.env= self.gym.create_env(self.sim, self.env_lower, self.env_upper, self.num_per_row)
            camera_props = gymapi.CameraProperties()
            camera_props.horizontal_fov = 100
            camera_props.width = 480
            camera_props.height = 480
            camera_props.enable_tensors = True
            self.camera_handle = self.gym.create_camera_sensor(self.env, camera_props)
            camera_pos = camera_position_dict[self.env_suite]
            camera_target = camera_target_dict[self.env_suite]
            self.gym.set_camera_location(self.camera_handle, self.env, camera_pos, camera_target)
            self.camera_handles.append(self.camera_handle)
            self.gym.start_access_image_tensors(self.sim)       
            actor_pose= gymapi.Transform()  
            actor_pose.p=actor_position[self.env_suite]
            actor_pose.r=actor_rotation[self.env_suite] 
            object_position= object_pos[self.env_suite]
            object_rotation= object_rot[self.env_suite]
            object_pose.p=object_position
            object_pose.r=object_rotation      
            table_pose = gymapi.Transform()
            table_pose.p = gymapi.Vec3(0.7, 0.0, 0.3)
            table_pose.r = gymapi.Quat(-0.707107, 0, 0.0, 0.707)
            
            self.actor_handle = self.gym.create_actor(self.env, self.asset,actor_pose, "actor", 0, 1)
            self.table_handle = self.gym.create_actor(self.env, self.table_asset, table_pose, "table", 0, 1)
            object_handle = self.gym.create_actor(self.env, self.object_asset,object_pose, "object",0, 0, 0)
            self.object_handles.append(object_handle)
            if self.env_suite !='None':
                object_idx = self.gym.get_actor_index(self.env, object_handle, gymapi.DOMAIN_SIM)
                print("Env suite is not None")
                self.object_indices.append(object_idx)
                
            actor_idx = self.gym.get_actor_index(self.env,self.actor_handle, gymapi.DOMAIN_SIM)
            self.actor_indices.append(actor_idx)
            #self.actor_handles.append(actor_handle)
            props = self.gym.get_actor_dof_properties(self.env, self.actor_handle)
            
            if self.asset_name=='allegro_hand':
                    self.color_hand()
            else:
                    self.color_curved_hand()
            props["stiffness"] =[3]*16
            props["damping"] = [0.18]*16
            props["friction"] = [0.01]*16
            props["armature"] = [0.001]*16
            props["velocity"] = [2.0]*16
            self.set_control_mode(props,self.control_mode)
            self.gym.set_actor_dof_properties(self.env, self.actor_handle, props) 
            return object_position, object_rotation

        # Color the robot hand      
        def color_hand(self):
            for j in range(self.num_dofs+13):   
                if j!=20 and j!=15 and j!=10 and j!=5 : 
                    self.gym.set_rigid_body_color(self.env, self.actor_handle,j, gymapi.MESH_VISUAL, gymapi.Vec3(0.15, 0.15, 0.15)) 
        
        # Color the curved robot hand
        def color_curved_hand(self):
            for j in range(self.num_dofs+13):   
                self.gym.set_rigid_body_color(self.env, self.actor_handle,j, gymapi.MESH_VISUAL, gymapi.Vec3(0.15, 0.15, 0.15))

        # Get the DOF names 
        def get_dof_names(self):
            dof_names = self.gym.get_asset_dof_names(self.asset)
            return dof_names

        # Get the DOF count
        def get_dof_count(self):
            num_dofs = self.gym.get_asset_dof_count(self.asset)
            return num_dofs

        # Create Viewer
        def create_viewer(self):
            viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            if viewer is None:
                print("*** Failed to create viewer")
                quit()
            return viewer
       
        # Reset the environment
        def reset(self,object_position,object_rotation):
            home_position=torch.zeros((1,self.num_dofs),dtype=torch.float32, device='cpu')        
             
            home_position=torch.tensor([-0.00137183, -0.22922094,  0.7265581 ,  0.79128325,0.9890924 ,  0.37431374,  0.36866143,
                                        0.77558154,0.00662423, -0.23064502,  0.73253167,  0.7449019 ,  0.08261403, -0.15844858,
                                        0.82595366,  0.7666822 ])
            self.set_position(home_position)
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.step_graphics(self.sim)
            self.gym.render_all_camera_sensors(self.sim)
            state=self.compute_observation(observation = 'position')
            self.root_state_tensor[self.object_indices[0],0:3]=to_torch(np.array([object_position.x,object_position.y,object_position.z]),dtype=torch.float,device='cpu')
            self.root_state_tensor[self.object_indices[0],3:7]=to_torch(np.array([object_rotation.x,object_rotation.y,object_rotation.z,object_rotation.w]),dtype=torch.float,device='cpu')      
            self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                    gymtorch.unwrap_tensor(self.root_state_tensor),
                                                    gymtorch.unwrap_tensor(self.object_indices), len(self.object_indices))
            return state

        # Compute the observation
        def compute_observation(self, observation):
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_actor_root_state_tensor(self.sim)
            self.gym.refresh_rigid_body_state_tensor(self.sim) 
            if observation=='image':
                state = self.gym.get_camera_image(self.sim,self.env, self.camera_handle, gymapi.IMAGE_COLOR)  
            elif observation=='position':     
                state = np.zeros(self.num_dofs)
                for i in range(self.num_dofs):
                    state[i]=self.gym.get_dof_position(self.env,i)  
            elif observation=='velocity':
                state=np.zeros(self.num_dofs)
                for i in range(self.num_dofs):
                    state[i]=self.gym.get_dof_velocity(self.env,i) 
         
            elif observation=='full_state':
                    for i in range(2*self.num_dofs):
                        if i<self.num_dofs:
                                state[i]=self.gym.get_dof_position(self.env,i)  
                        else:
                                state[i]=self.gym.get_dof_velocity(self.env,i)  
            return state
        
        # Get the DOF position
        def get_dof_position(self):
            return self.compute_observation(observation='position')
        
        # Get Time in simulation
        def get_time(self):
            return self.gym.get_elapsed_time(self.sim)

        # Get the RGB Depth images
        def get_rgb_depth_images(self):
            color_image = None
            while color_image is None:
                color_image =self.gym.get_camera_image_gpu_tensor(self.sim,self.env, self.camera_handle, gymapi.IMAGE_COLOR)
                color_image=gymtorch.wrap_tensor(color_image)
                color_image=color_image.cpu().numpy()
                color_image=color_image[:,:,[2,1,0]]
                
                depth_image =self.gym.get_camera_image_gpu_tensor(self.sim, self.env,self.camera_handle,gymapi.IMAGE_DEPTH)
                depth_image =gymtorch.wrap_tensor(depth_image)
                depth_image=depth_image.cpu().numpy()
                time=self.get_time()
                    
            return color_image, depth_image, time
    
        # Set the DOF position
        def set_position(self, position):
            self.gym.set_dof_position_target_tensor(self.sim,  gymtorch.unwrap_tensor(position))
        
        # Get the endeffector position
        def get_endeff_position(self):
            state=self.gym.acquire_actor_root_state_tensor(self.sim)
            state=gymtorch.wrap_tensor(state)
            position=state.numpy()[0,0:7]
            print("End Effector Position",position)
            return position

        # Set the control mode
        def set_control_mode(self,props,mode=None):
            for k in range(self.num_dofs):
                if mode is not None:
                    if mode=='Position':
                            props["driveMode"][k] = gymapi.DOF_MODE_POS
                    elif mode=='Velocity':
                            props["driveMode"][k] = gymapi.DOF_MODE_VEL
                    elif mode=='Effort':
                            props["driveMode"][k] = gymapi.DOF_MODE_EFFORT
                    elif mode=='Position_Velocity':
                        
                            props["driveMode"][k] = gymapi.DOF_MODE_POS   

                else:
                        return
                        
        @property              
        def timer(self):
            return self._timer
       
        # Take Action             
        def take_action(self):
            joint_angles=self.joint_angle_subscriber.recv_keypoints()
            joint_angles=to_torch((joint_angles), device='cpu') 
            self.set_position(joint_angles)                        
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            self.gym.simulate(self.sim)
            self.actual_joint_angles= self.get_dof_position()
            self.actualanglepublisher.pub_keypoints(self.actual_joint_angles,'current_angles')
            self.gym.fetch_results(self.sim, True)
            self.gym.step_graphics(self.sim)
            self.gym.render_all_camera_sensors(self.sim)

        

        

                        

       
        