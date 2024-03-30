import os 
import cv2
import numpy as np
import pickle 
import shutil

from tqdm import tqdm

from .prep_module import PreprocessorModule

# Dumping video to images
# Creating pickle files to pick images
def dump_video_to_images(root: str, video_type: str ='rgb', view_num: int=0, dump_all=True) -> None:
    # Convert the video into image sequences and name them with the frames
    video_path = os.path.join(root, f'cam_{view_num}_{video_type}_video.avi')
    images_path = os.path.join(root, f'cam_{view_num}_{video_type}_images')
    if os.path.exists(images_path):
        print(f'{images_path} exists it is being removed')
        shutil.rmtree(images_path)

    os.makedirs(images_path, exist_ok=True)

    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    assert os.path.exists(os.path.join(root, 'image_indices.pkl')) or dump_all, 'If not dump_all, Image Indices should have been dumped before converting video to images'
    if not dump_all: # Otherwise we dn't need desired indices
        with open(os.path.join(root, 'image_indices.pkl'), 'rb') as f:
            desired_indices = pickle.load(f)

    frame_id = 0
    desired_img_id = 0
    print(f'dumping video in {root}')
    pbar = tqdm(total = frame_count)
    while success: 
        pbar.update(1)
        if (not dump_all and frame_id == desired_indices[desired_img_id][1]) or dump_all:
            cv2.imwrite('{}.png'.format(os.path.join(images_path, 'frame_{}'.format(str(frame_id).zfill(5)))), image)
            curr_id = desired_img_id
            while desired_img_id < len(desired_indices)-1 and desired_indices[curr_id][1] == desired_indices[desired_img_id][1]:
                desired_img_id += 1
        success, image = vidcap.read()
        frame_id += 1

    print(f'Dumping finished in {root}')

class ImagePreprocessor(PreprocessorModule):
    def __init__(self, camera_id, is_ssl=False, time_difference=None):
        super().__init__(
            camera_id=camera_id,
            is_ssl=is_ssl, 
            time_difference=time_difference)

        self.load_file_name = f'cam_{camera_id}_rgb_video.metadata'
        self.dump_file_name = f'image_indices.pkl'
        print('inside image preprocessor - self.load_file_name: {}'.format(
            self.load_file_name
        ))
        self.current_id = 0
        self.indices = []

    def __repr__(self):
        return 'image_preprocessor'

    def load_data(self):
        file_path = os.path.join(self.root, self.load_file_name)
        with open(file_path, 'rb') as f:
            image_metadata = pickle.load(f)
            image_timestamps = np.asarray(image_metadata['timestamps']) / 1000.

        self.data = dict(
            timestamps = image_timestamps
        )

    def dump_images(self):
        print('in dump images - self.root: {}'.format(
            self.root
        ))
        dump_video_to_images(
            root = self.root,
            video_type = 'rgb',
            view_num = self.camera_id,
            dump_all = self.is_ssl
        )

    def get_next_timestamp(self):
        curr_ts = self.current_timestamp

        if not self.time_difference is None:
            desired_ts = curr_ts + self.time_difference
            next_id = self._get_closest_id(desired_ts)
            return self.data['timestamps'][next_id]
        else:
            return -1


        
    
