# Base class for the preprocessor modules

from abc import ABC, abstractmethod

import os 
import cv2
import h5py 
import numpy as np
import pickle 
import shutil


class PreprocessorModule(ABC):
    def __init__(self, **kwargs):
        self.__dict__.update(**kwargs)

    @property
    def current_timestamp(self):
        return self.data['timestamps'][self.current_id]
    
    def update_root(self, demo_id, root):
        self.root = root
        self.demo_id = demo_id
        self.indices = []
        self.load_data()
        print('Updated the root of PreprocessorModule ({}) for demo_id: {} - root: {}'.format(
            self, demo_id, root
        ))
        self.reset_current_id()

    def reset_current_id(self):
        self.current_id = 0

    def update_indices(self):
        self.indices.append([self.demo_id, self.current_id])

    def is_finished(self):
        if self.current_id >= len(self.data['timestamps'])-1: 
            return True
    
        return False

    # @abstractmethod
    def update_next_id(self, desired_timestamp):
        self.current_id = self._get_closest_id(desired_timestamp)

    def _get_closest_id(self, desired_ts): 
        for i in range(self.current_id, len(self.data['timestamps'])):
            if self.data['timestamps'][i] >= desired_ts:
                return i
            
        return i

    def dump_data_indices(self): 
        print('self.indices in image: {}'.format(
            self.indices
        ))
        file_path = os.path.join(self.root, self.dump_file_name)
        with open(file_path, 'wb') as f: 
            pickle.dump(self.indices, f)

    @abstractmethod
    def load_data(self):
        raise NotImplementedError

    @abstractmethod
    def get_next_timestamp(self):
        raise NotImplementedError


        
    