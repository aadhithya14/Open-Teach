import glob
import numpy as np


# from franka_allegro.datasets import *
# Main preprocessor module
# It gets each module's preprocessor, and preprocesses all of them

class Preprocessor:
    def __init__(self, data_path, modules, 
                 dump_data_indices=False, process_single_demo=False,
                 human_data=None, shorten_demo=None, repr_preprcessor=None, 
                 **kwargs):
        
        self.data_path = data_path
        self.modules = modules
        self.dump_data_indices = dump_data_indices
        self.process_single_demo = process_single_demo

        # self.human_data = human_data # TODO: Add these
        # self.shorten_demo = shorten_demo # TODO: Add these
        # self.repr_preprocess = repr_preprcessor # TODO: Add these

    def apply(self):

        if self.process_single_demo:
            roots = [self.data_path]
        else:
            roots = glob.glob(f'{self.data_path}/demonstration_*')
            roots = sorted(roots)

        for demo_id, root in enumerate(roots):  
            # Update the root of the module
            self._update_root(demo_id, root)
            # if 'robot' in self.modules:
            #     self.modules['robot'].dump_fingertips()

            if self.dump_data_indices:
                self.dump_indices()

            if 'image' in self.modules:
                self.modules['image'].dump_images()

    def _reset_indices(self): 
        for module in self.modules.values():
            module.reset_current_id() 

    def _update_root(self, demo_id, root):
        for module in self.modules.values():
            module.update_root(demo_id, root)

    def _find_latest_module(self):
        latest_ts = 0
        module_key = None
        for key, module in self.modules.items():
            current_ts = module.current_timestamp
            if current_ts > latest_ts:
                latest_ts = current_ts
                module_key = key 

        return module_key

    def dump_indices(self):
        # TODO: add shorteninig part

        self._reset_indices()

        latest_key = self._find_latest_module()
        metric_timestamp = self.modules[latest_key].current_timestamp

        # Find the beginning ids for each module
        for module in self.modules.values():
            module.update_next_id(metric_timestamp)
            module.update_indices()

            print(f'{module} - ts: {module.current_timestamp}')

        # Update timestamps and ids consequitively
        while True:

            # Each module returns a 'metric' timestamp
            # We will choose the closest timestamp to the curr_ts as the metric timestamp
            # If the module is not selective (for ex touch sensors are not important for preprocesing)
            # they will return -1
            module_ts_diff = 1e3
            cand_metric_ts = metric_timestamp
            for key, module in self.modules.items():

                next_ts = module.get_next_timestamp()
                
                if next_ts != -1 and next_ts - metric_timestamp < module_ts_diff:
                    module_ts_diff = next_ts - metric_timestamp 
                    cand_metric_ts = next_ts

            if cand_metric_ts == metric_timestamp: # If it hasn't changed at all
                break
            metric_timestamp = cand_metric_ts 
            
            # Update the ids of each module
            for module in self.modules.values():
                module.update_next_id(metric_timestamp)

            # Check if the loop should be completed or not
            finished = np.array([module.is_finished() for module in self.modules.values()]).any()
            if finished:
                break

            # If not add update the indices array of each module
            for module in self.modules.values():
                module.update_indices()

        for module in self.modules.values():
            module.dump_data_indices()
                    
                
