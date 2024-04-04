import h5py as h5
import numpy as np
import pickle as pkl
import cv2
from pathlib import Path
from sentence_transformers import SentenceTransformer

DATASET_PATH = Path("extracted_data/put_toast_in_oven")
num_cams = 1
img_size = (128, 128)

# Get task name sentence
task_name = DATASET_PATH.name.split("/")[-1]
task_name = task_name.replace("_", " ")
lang_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
task_emb = lang_model.encode(task_name)

# Init storing variables
observations = []

# Store max and min
max_cartesian, min_cartesian = None, None
max_gripper, min_gripper = None, None
max_joint, min_joint = None, None

# Load each data point and save in a list
dirs = [x for x in DATASET_PATH.iterdir() if x.is_dir()]
for i, data_point in enumerate(dirs):
    print(f"Processing data point {i+1}/{len(dirs)}")
    observation = {}
    # images
    image_indices = np.array(pkl.load(open(data_point / "image_indices.pkl", "rb")))[:, 1]
    for idx in range(num_cams):
        cam_dir = data_point / f"cam_{idx}_rgb_images"
        # img_paths = sorted(cam_dir.iterdir())
        imgs = [cv2.resize(cv2.imread(str(cam_dir / 'frame_{:05d}.png'.format(num) )), img_size) for num in image_indices]
        observation[f"pixels{idx}"] = imgs
    # robot state
    indices = pkl.load(open(data_point / "xarm_indices.pkl", "rb"))
    cartesian_states = h5.File(data_point / "xarm_cartesian_states.h5", "r")["cartesian_positions"][()]
    gripper_states = h5.File(data_point / "xarm_gripper_states.h5", "r")["gripper_positions"][()]
    joint_states = h5.File(data_point / "xarm_joint_states.h5", "r")["joint_positions"][()]
    # convert to numpy arrays
    indices = np.array(indices)
    cartesian_states = np.array(cartesian_states)[indices[:, 1]]
    gripper_states = np.array(gripper_states)[indices[:, 1]]
    joint_states = np.array(joint_states)[indices[:, 1]]
    # save in observation
    observation["cartesian_states"] = cartesian_states
    observation["gripper_states"] = gripper_states
    observation["joint_states"] = joint_states

    # update max and min
    if max_cartesian is None:
        max_cartesian = np.max(cartesian_states, axis=0)
        min_cartesian = np.min(cartesian_states, axis=0)
    else:
        max_cartesian = np.maximum(max_cartesian, np.max(cartesian_states, axis=0))
        min_cartesian = np.minimum(min_cartesian, np.min(cartesian_states, axis=0))
    if max_gripper is None:
        max_gripper = np.max(gripper_states)
        min_gripper = np.min(gripper_states)
    else:
        max_gripper = np.maximum(max_gripper, np.max(gripper_states))
        min_gripper = np.minimum(min_gripper, np.min(gripper_states))
    if max_joint is None:
        max_joint = np.max(joint_states, axis=0)
        min_joint = np.min(joint_states, axis=0)
    else:
        max_joint = np.maximum(max_joint, np.max(joint_states, axis=0))
        min_joint = np.minimum(min_joint, np.min(joint_states, axis=0))

    # append to observations
    observations.append(observation)

# Save the data
data = {
    'observations': observations,
    'max_cartesian': max_cartesian,
    'min_cartesian': min_cartesian,
    'max_gripper': max_gripper,
    'min_gripper': min_gripper,
    'max_joint': max_joint,
    'min_joint': min_joint,
    'task_emb': task_emb
}
pkl.dump(data, open(DATASET_PATH / 'expert_demos.pkl', "wb"))    