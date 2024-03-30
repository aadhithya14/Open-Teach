## Teleop
- Start camera servers.
- `python teleop.py`

# Data collection
- `python data_collect.py demo_num=1`

# Process data
- Change data_path in `preprocessor_module.data_path` in `configs/preprocess.yaml`
- Save as frames: `python preprocess.py`
- Then save as pkl: `python convert_to_pkl.py`