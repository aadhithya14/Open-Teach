import h5py as h5
import numpy as np
from pandas import read_csv
import pickle as pkl
import cv2
from pathlib import Path
from sentence_transformers import SentenceTransformer

DATASET_PATH = Path("extracted_data/pick_honey_bottle")
# num_cams = 3
camera_indices = [0,1,2,3,4]
img_size = (128, 128)

# Get task name sentence
# task_name = DATASET_PATH.name.split("/")[-1]
# task_name = task_name.replace("_", " ")
label_path = DATASET_PATH / "label.txt"
task_name = label_path.read_text().strip()
print(f"Task name: {task_name}")
lang_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
task_emb = lang_model.encode(task_name)

# Init storing variables
observations = []

# Store max and min
max_cartesian, min_cartesian = None, None
# max_rel_cartesian, min_rel_cartesian = None, None
max_gripper, min_gripper = None, None

# Load each data point and save in a list
dirs = [x for x in DATASET_PATH.iterdir() if x.is_dir()]
for i, data_point in enumerate(dirs):
    print(f"Processing data point {i+1}/{len(dirs)}")
    observation = {}
    # images
    image_dir = data_point / "output"
    if not image_dir.exists():
        print(f"Data point {data_point} is incomplete")
        continue
    for idx in camera_indices:
        # Read the frames in the video
        video_path = image_dir / f"camera{idx}.mp4"
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Video {video_path} could not be opened")
            continue
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, img_size)
            frames.append(frame)
        observation[f"pixels{idx}"] = np.array(frames)
    # read cartesian and gripper states from csv
    csv_path = data_point / 'states.csv'
    state = read_csv(csv_path)
    # Read cartesian state where every element is a 6D pose
    # Separate the pose into values instead of string
    cartesian_states = state['pose_aa'].values
    cartesian_states = np.array([np.array([float(x.strip()) for x in pose[1:-1].split(',')]) for pose in cartesian_states], dtype=np.float32)
    # Convert roll-pitch-yaw to sin-cos
    cartesian_pos = cartesian_states[:, :3]
    cartesian_ori = cartesian_states[:, 3:]
    cartesian_ori = np.concatenate([np.sin(cartesian_ori), np.cos(cartesian_ori)], axis=1)
    cartesian_states = np.concatenate([cartesian_pos, cartesian_ori], axis=1)
    # rest
    gripper_states = state['gripper_state'].values.astype(np.float32)
    observation["cartesian_states"] = cartesian_states.astype(np.float32)
    observation["gripper_states"] = gripper_states.astype(np.float32)
    # # Relative cartesian states
    # relative_cartesian_states = np.diff(cartesian_states, axis=0)
    # relative_cartesian_states = np.concatenate([relative_cartesian_states, np.zeros((1, 9))], axis=0)
    # observation["relative_cartesian_states"] = relative_cartesian_states
    
    # update max and min
    if max_cartesian is None:
        max_cartesian = np.max(cartesian_states, axis=0)
        min_cartesian = np.min(cartesian_states, axis=0)
    else:
        max_cartesian = np.maximum(max_cartesian, np.max(cartesian_states, axis=0))
        min_cartesian = np.minimum(min_cartesian, np.min(cartesian_states, axis=0))
    # if max_rel_cartesian is None:
    #     max_rel_cartesian = np.max(relative_cartesian_states, axis=0)
    #     min_rel_cartesian = np.min(relative_cartesian_states, axis=0)
    # else:
    #     max_rel_cartesian = np.maximum(max_rel_cartesian, np.max(relative_cartesian_states, axis=0))
    #     min_rel_cartesian = np.minimum(min_rel_cartesian, np.min(relative_cartesian_states, axis=0))
    if max_gripper is None:
        max_gripper = np.max(gripper_states)
        min_gripper = np.min(gripper_states)
    else:
        max_gripper = np.maximum(max_gripper, np.max(gripper_states))
        min_gripper = np.minimum(min_gripper, np.min(gripper_states))
    
    # append to observations
    observations.append(observation)

# Save the data
data = {
    'observations': observations,
    'max_cartesian': max_cartesian,
    'min_cartesian': min_cartesian,
    'max_gripper': max_gripper,
    'min_gripper': min_gripper,
    'task_emb': task_emb
}
pkl.dump(data, open(DATASET_PATH / 'expert_demos.pkl', "wb"))    