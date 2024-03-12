# Installation for Simulation Environment

### Allegro Simulation
1. Install the environment for Allegro-Isaac environment specifically using 

   `conda env create -f env_isaac.yml`

2. Activate the conda environment. You need to get IsaacGym from preview release. Change to isaacgym directory.
   `pip install -e .` 

3. IsaacGym Specific Errors

   1) If you see an error like this 

      `ImportError: libpython3.7m.so.1.0: cannot open object file: No such file or directory`

      Installing the corresponding Python lib should fix that problem.

      `sudo apt install libpython3.7`

      If you are running Ubuntu 20.04, which does not have a libpython3.7 package, you will instead need to set the LD_LIBRARY_PATH variable appropriately:

      `export LD_LIBRARY_PATH=/home/xyz/miniforge3/envs/openteach_isaac/lib`

   2. If you see an error like this

      `ImportError: /usr/lib/python3.8/lib-dynload/_ctypes.cpython-38-x86_64-linux-gnu.so: undefined symbol: ffi_closure_alloc, version LIBFFI_CLOSURE_7.0`

      You will need to set PYTHONPATH appropriately:

      `export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7`

   3. If you have multiple vulkan devices visible when running `vulkaninfo`, you may need to explicitly force use of your NVIDIA GPU. (Required)

      `export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json`

   4. Export Library path to avoid errors related to cublas (Required)

      1. `export LD_LIBRARY_PATH='/home/xyz/miniforge3/envs/openteach_isaac/lib/python3.7/site-packages/nvidia/cublas/lib/':$LD_LIBRARY_PATH`

   5. Export library path to avoid errors. (Required)

      1. `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:'/home/xyz/miniforge3/envs/openteach_isaac/lib'`

3. After fixing the above errors and paths related to IsaacGym.

   1. For teleoperation run

      For Allegro Sim 
      `python3 teleop.py robot=allegro_sim sim_env=True`
      
      For Moving Allegro Sim
      `python3 teleop.py robot=moving_allegro_sim sim_env=True`

   2. For Data Collection run

      For Allegro Sim
      `python3 data_collect.py robot=allegro_sim sim_env=True demo_num=1`

      For Moving Allegro Sim 
      `python3 teleop.py robot=moving_allegro_sim sim_env=True demo_num=1`


4. If everything works successfully you will be able to get the end effector position from the simulation and will be able to stream the camera view within the oculus and data will be saved.
Use SingleHandArm-APK for IsaacGym Hand environment.

### Libero Simulation
1. Install the LIBERO env from [here](https://github.com/Lifelong-Robot-Learning/LIBERO?tab=readme-ov-file#Installation). No need to install the conda environment from LIBERO. Just clone the LIBERO codebase and install LIBERO using `pip install -r requirements.txt`

2. For LIBERO sim , follow the official installation documentation. Then in a python shell,
   `import libero`
   `from libero.libero import benchmark, get_libero_path`
    when prompted for the following line 
   `answer = input("Do you want to specify a custom path for the dataset folder? (Y/N):").lower()`
   input N. 

3. Then,

   1. For teleoperation run
      `python3 teleop.py robot=libero_sim sim_env=True`

   2. For Data Collection run
      `python3 data_collect.py robot=libero_sim sim_env=True demo_num=1`

4. Use the Bimanual-APK for Libero

   

   

   