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

DATA_PATH = Path("/home/siddhant/github/Open-Teach/extracted_data/pick_can_test_ot")
num_demos = 5
num_cams = 4
states_file_name = "states"

for num in range(1, num_demos+1):
    print("Processing demonstration", num)
    output_path = f"{DATA_PATH}/demonstration_{num}/"
    
    cam_avis = [f"{DATA_PATH}/demonstration_{num}/cam_{i}_rgb_video.avi" for i in range(num_cams)]

    cartesian = h5py.File(f"{DATA_PATH}/demonstration_{num}/xarm_cartesian_states.h5", 'r')
    state_timestamps = cartesian['timestamps']
    state_positions = cartesian['cartesian_positions']

    gripper = h5py.File(f"{DATA_PATH}/demonstration_{num}/xarm_gripper_states.h5", 'r')
    gripper_positions = gripper['gripper_positions']

    state_positions = np.array(state_positions)
    gripper_positions = np.array(gripper_positions)
    gripper_positions = gripper_positions.reshape(-1, 1)

    state_timestamps = np.array(state_timestamps)

    # Set minimum timestamp to the first instance the robot moves
    min_timestamp_idx = 0
    for i in range(1, len(state_positions)):
        if not np.array_equal(state_positions[i], state_positions[i-1]):
            min_timestamp_idx = i
            break
    
    # Set max timestamp to the last instance the robot moves
    max_timestamp_idx = len(state_positions) - 1
    for i in range(len(state_positions) - 2, min_timestamp_idx-1, -1):
        if not np.array_equal(state_positions[i], state_positions[i+1]):
            max_timestamp_idx = i
            break
    
    # read metadata file
    CAM_TIMESTAMPS = []
    CAM_VALID_LENS = []
    skip = False
    for idx in range(num_cams):
        cam_meta_file_path = f"{DATA_PATH}/demonstration_{num}/cam_{idx}_rgb_video.metadata"
        with open(cam_meta_file_path, 'rb') as f:
            image_metadata = pickle.load(f)
            image_timestamps = np.asarray(image_metadata['timestamps']) / 1000.

            cam_timestamps = dict(
                timestamps = image_timestamps
            )
        #convert to numpy array
        cam_timestamps = np.array(cam_timestamps['timestamps'])

        # Fish eye cam timestamps are divided by 1000
        if max(cam_timestamps) < state_timestamps[min_timestamp_idx]:
            cam_timestamps *= 1000

        # valid indices 
        discard_start = cam_timestamps < state_timestamps[min_timestamp_idx]
        discard_end = cam_timestamps > state_timestamps[max_timestamp_idx]
        valid_indices = ~(discard_start | discard_end)
        cam_timestamps = cam_timestamps[valid_indices]

        # # filter timestamps before the robot moves
        # cam_timestamps = cam_timestamps[cam_timestamps >= state_timestamps[min_timestamp_idx] and cam_timestamps <= state_timestamps[max_timestamp_idx]]

        # # filter timestamps after the robot stops moving
        # cam_timestamps = cam_timestamps[cam_timestamps <= state_timestamps[max_timestamp_idx]]

        # if no valid timestamps, skip
        if len(cam_timestamps) == 0:
            skip = True
            break

        # CAM_VALID_LENS.append(len(cam_timestamps))
        CAM_VALID_LENS.append((sum(discard_start), sum(discard_end)))
        CAM_TIMESTAMPS.append(cam_timestamps)
    if skip:
        continue
    
    # cam frames
    CAM_FRAMES = []
    for idx in range(num_cams):
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
        # cam_frames = cam_frames[-CAM_VALID_LENS[idx]:]
        cam_frames = cam_frames[CAM_VALID_LENS[idx][0]:-CAM_VALID_LENS[idx][1]]
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
    output_folder = output_path + "output"
    os.makedirs(output_folder, exist_ok=True)
    camera_folders = [f"camera{i}" for i in range(num_cams)]
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

    # def find_consecutive_positions(csv_file): ### need to be modified to consider gripper states
    #     df = pd.read_csv(csv_file)
    #     consecutive_ranges = []
    #     start_idx = None
    #     prev_position = None
    #     ending_idx = len(df)-1

    #     for idx, row in df.iterrows():
    #         current_position = row['pose_aa']

    #         if prev_position is None:
    #             start_idx = idx
    #             prev_position = current_position
    #         elif current_position != prev_position:
    #             if idx - start_idx >= 2:
    #                 consecutive_ranges.append((start_idx, idx - 1))
    #             start_idx = idx
    #             prev_position = current_position

    #     if prev_position is not None and len(df) - start_idx >= 2:
    #         consecutive_ranges.append((start_idx, len(df) - 1))


    #     return consecutive_ranges

    csv_file = os.path.join(output_path, f"{states_file_name}.csv")
    # left_indexs_to_delete = find_consecutive_positions(csv_file)

    def get_timestamp_from_filename(filename):
        # Extract the timestamp from the filename using regular expression
        timestamp_match = re.search(r'\d+\.\d+', filename)
        if timestamp_match:
            return float(timestamp_match.group())
        else:
            return None
        
    #add desired gripper states
    for file in [csv_file]: #, right_csv_file):
        df = pd.read_csv(file)
        df['desired_gripper_state'] = df['gripper_state'].shift(-1)
        # Step 3a: For the last timestamp, set "desired_gripper_flag" to the value of the previous row
        df.loc[df.index[-1], 'desired_gripper_state'] = df.loc[df.index[-2], 'gripper_state']
        # # Step 3b: Only save indices correspnding to those in index[-1]
        # df = df.iloc[index[-1]]
        # Step 4: Update the CSV file with the modified DataFrame
        df.to_csv(file, index=False)

    def save_only_videos(base_folder_path):
        base_folder_path = os.path.join(base_folder_path, 'output')
        # Iterate over each camera folder
        for cam in range(num_cams):  # Iterating from camera1 to camera6
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