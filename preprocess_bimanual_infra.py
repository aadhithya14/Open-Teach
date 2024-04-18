import numpy as np
import pickle
import cv2
import pandas as pd
import os
import subprocess
import os
import re
import shutil
import h5py
from pathlib import Path

FOLDER_NAME = "2024.04.17"
DATA_PATH = Path(f"/mnt/robotlab/siddhant/projects/scaling_polytask//extracted_data/{FOLDER_NAME}")
SAVE_PATH = Path("/mnt/robotlab/siddhant/projects/scaling_polytask//processed_data")
num_demos = None
cam_indices = [1, 2, 3, 4, 51, 52]
states_file_name = "states"
task_names = ["lift_up_lid", "open_oven", "pick_bowl_from_rack"]

for TASK_NAME in task_names:
    print(f"#################### Processing task {TASK_NAME} ####################")

    # Make save directory
    (SAVE_PATH / TASK_NAME).mkdir(parents=True, exist_ok=True)

    # demo directories
    DEMO_DIRS = [f for f in (DATA_PATH / TASK_NAME).iterdir() if f.is_dir()]
    if num_demos is not None:
        DEMO_DIRS = DEMO_DIRS[:num_demos]

    for num, demo_dir in enumerate(DEMO_DIRS):
        print("Processing demonstration", demo_dir.name)
        output_path = f"{SAVE_PATH}/{TASK_NAME}/demonstration_{num}/"
        Path(output_path).mkdir(parents=True, exist_ok=True)
        
        cam_avis = [f"{demo_dir}/cam_{i}_rgb_video.avi" for i in cam_indices]

        cartesian = h5py.File(f"{demo_dir}/xarm_cartesian_states.h5", 'r')
        state_timestamps = cartesian['timestamps']
        state_positions = cartesian['cartesian_positions']

        gripper = h5py.File(f"{demo_dir}/xarm_gripper_states.h5", 'r')
        gripper_positions = gripper['gripper_positions']

        state_positions = np.array(state_positions)
        gripper_positions = np.array(gripper_positions)
        gripper_positions = gripper_positions.reshape(-1, 1)

        state_timestamps = np.array(state_timestamps)

        # Find indices of timestamps where the robot moves
        static_timestamps = []
        static = False
        start, end = None, None
        for i in range(1, len(state_positions)-1):       
            if np.array_equal(state_positions[i], state_positions[i+1]) and static==False:
                static = True
                start = i
            elif not np.array_equal(state_positions[i], state_positions[i+1]) and static==True:
                static = False
                end = i
                static_timestamps.append((start, end))
        if static:
            static_timestamps.append((start, len(state_positions)-1))

            
        # read metadata file
        CAM_TIMESTAMPS = []
        CAM_VALID_LENS = []
        skip = False
        for idx in cam_indices:
            cam_meta_file_path = f"{demo_dir}/cam_{idx}_rgb_video.metadata"
            with open(cam_meta_file_path, 'rb') as f:
                image_metadata = pickle.load(f)
                image_timestamps = np.asarray(image_metadata['timestamps']) / 1000.

                cam_timestamps = dict(
                    timestamps = image_timestamps
                )
            #convert to numpy array
            cam_timestamps = np.array(cam_timestamps['timestamps'])

            # Fish eye cam timestamps are divided by 1000
            if max(cam_timestamps) < state_timestamps[static_timestamps[0][1]]:
                cam_timestamps *= 1000
            elif min(cam_timestamps) > state_timestamps[static_timestamps[-1][0]]:
                cam_timestamps /= 1000

            valid_indices = []
            for k in range(len(static_timestamps)-1):
                start_idx = sum(cam_timestamps < state_timestamps[static_timestamps[k][1]])
                end_idx = sum(cam_timestamps < state_timestamps[static_timestamps[k+1][0]])
                valid_indices.extend([i for i in range(start_idx, end_idx+1)])
            cam_timestamps = cam_timestamps[valid_indices]
            
            # if no valid timestamps, skip
            if len(cam_timestamps) == 0:
                skip = True
                break

            CAM_VALID_LENS.append(valid_indices)
            CAM_TIMESTAMPS.append(cam_timestamps)
        if skip:
            continue
        
        # cam frames
        CAM_FRAMES = []
        for idx in range(len(cam_avis)): #cam_indices:
            cam_avi = cam_avis[idx]
            cam_frames = []
            cap_cap = cv2.VideoCapture(cam_avi)
            while(cap_cap.isOpened()):   
                ret, frame = cap_cap.read()
                if ret == False:
                        break
                cam_frames.append(frame)
            cap_cap.release()

            # save frames
            cam_frames = np.array(cam_frames)
            cam_frames = cam_frames[CAM_VALID_LENS[idx]]
            CAM_FRAMES.append(cam_frames)

        rgb_frames = CAM_FRAMES
        timestamps = CAM_TIMESTAMPS
        timestamps.append(state_timestamps)

        min_time_index = np.argmin([len(timestamp) for timestamp in timestamps])
        reference_timestamps = timestamps[min_time_index]
        align = []
        index = []
        for i in range(len(timestamps)):
            # aligning frames
            if i == min_time_index:
                align.append(timestamps[i])
                index.append(np.arange(len(timestamps[i])))
                continue
            curindex = []
            currrlist = []
            for j in range(len(reference_timestamps)):
                curlist = []
                for k in range(len(timestamps[i])):
                    curlist.append(abs(timestamps[i][k] - reference_timestamps[j]))
                min_index = curlist.index(min(curlist))
                currrlist.append(timestamps[i][min_index])
                curindex.append(min_index)
            align.append(currrlist)
            index.append(curindex)

        index = np.array(index)

        for i in range(len(index)):
            print(index[i])

        # convert left_state_timestamps and left_state_positions to a csv file with header "created timestamp", "pose_aa", "gripper_state"
        state_timestamps_test = pd.DataFrame(state_timestamps)
        # convert each pose_aa to a list
        state_positions_test = state_positions
        for i in range(len(state_positions_test)):
            state_positions_test[i] = np.array(state_positions_test[i])
        state_positions_test = pd.DataFrame({'column': [list(row) for row in state_positions_test]})
        # convert left_gripper to True and False
        gripper_positions_test = pd.DataFrame(gripper_positions)
        
        state_test = pd.concat([state_timestamps_test, state_positions_test, gripper_positions_test], axis=1)
        with open(output_path + f'big_{states_file_name}.csv', 'a') as f:
            state_test.to_csv(f, header=["created timestamp", "pose_aa", "gripper_state"], index=False)

        df = pd.read_csv(output_path + f'big_{states_file_name}.csv')
        for i in range(len(reference_timestamps)):
                curlist = []
                for j in range(len(state_timestamps)):
                    curlist.append(abs(state_timestamps[j] - reference_timestamps[i]))
                min_index = curlist.index(min(curlist))
                min_df = df.iloc[min_index]
                min_df = min_df.to_frame().transpose()
                with open(output_path + f'{states_file_name}.csv', 'a') as f:
                    min_df.to_csv(f, header=f.tell()==0, index=False)

        # Create folders for each camera if they don't exist
        output_folder = output_path + "videos"
        os.makedirs(output_folder, exist_ok=True)
        camera_folders = [f"camera{i}" for i in cam_indices]
        for folder in camera_folders:
            os.makedirs(os.path.join(output_folder, folder), exist_ok=True)

        # Iterate over each camera and extract the frames based on the indexes
        for camera_index, frames in enumerate(rgb_frames):
            camera_folder = camera_folders[camera_index]
            print(f"Extracting frames for {camera_folder}...")
            indexes = index[camera_index]

            # Iterate over the indexes and save the corresponding frames
            for i, indexx in enumerate(indexes):
                if i % 100 == 0:
                    print(f"Extracting frame {i}...")
                frame = frames[indexx]
                # name frame with its timestamp
                image_output_path = os.path.join(output_folder, camera_folder, f"frame_{i}_{timestamps[camera_index][indexx]}.jpg")
                cv2.imwrite(image_output_path, frame)


        csv_file = os.path.join(output_path, f"{states_file_name}.csv")

        def get_timestamp_from_filename(filename):
            # Extract the timestamp from the filename using regular expression
            timestamp_match = re.search(r'\d+\.\d+', filename)
            if timestamp_match:
                return float(timestamp_match.group())
            else:
                return None
            
        #add desired gripper states
        for file in [csv_file]:
            df = pd.read_csv(file)
            df['desired_gripper_state'] = df['gripper_state'].shift(-1)
            df.loc[df.index[-1], 'desired_gripper_state'] = df.loc[df.index[-2], 'gripper_state']
            df.to_csv(file, index=False)

        def save_only_videos(base_folder_path):
            base_folder_path = os.path.join(base_folder_path, 'videos')
            # Iterate over each camera folder
            for cam in cam_indices:
                cam_folder = f'camera{cam}'
                full_folder_path = os.path.join(base_folder_path, cam_folder)
                
                # Check if the folder exists
                if os.path.exists(full_folder_path):
                    # List all jpg files
                    all_files = [f for f in os.listdir(full_folder_path) if f.endswith('.jpg')]

                    # Sort files based on the floating-point number in their name
                    sorted_files = sorted(all_files, key=get_timestamp_from_filename)

                    # Write filenames to a temp file
                    temp_list_filename = os.path.join(base_folder_path, 'temp_list.txt')
                    with open(temp_list_filename, 'w') as f:
                        for filename in sorted_files:
                            f.write(f"file '{os.path.join(full_folder_path, filename)}'\n")
                    
                    # Use ffmpeg to convert sorted images to video
                    output_video_path = os.path.join(base_folder_path, f'camera{cam}.mp4')
                    cmd = [
                        'ffmpeg',
                        '-f', 'concat',
                        '-safe', '0',
                        '-i', temp_list_filename,
                        '-framerate', '30',  # assuming 24 fps, change if needed
                        '-vcodec', 'libx264',
                        '-crf', '18',  # quality, lower means better quality
                        '-pix_fmt', 'yuv420p',
                        output_video_path
                    ]
                    subprocess.run(cmd, check=True)

                    # Delete the temporary list file and the image folder
                    os.remove(temp_list_filename)
                    shutil.rmtree(full_folder_path)
                else:
                    print(f"Folder {cam_folder} does not exist!")

        save_only_videos(output_path)

    # copy labels.txt
    shutil.copy(f"{DATA_PATH}/{TASK_NAME}/label.txt", str(SAVE_PATH / TASK_NAME / "label.txt"))