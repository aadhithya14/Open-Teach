# Policy Learning

Since the morphologies of robots and simulations vary so much , we use different learning methods to learn policies for different robots and simulations. Below we have listed down the Imitation learning and Inverse RL algorithms used for each robots and simulations with open-sourced codefiles. 

1) Franka-Allegro: We use TAVI to learn tactile-vision policies for the Franka-Allegro setup.[Code](https://github.com/NYU-robot-learning/FrankaAllegro-Policies)
2) Allegro-Sim: We use FISH for training vision policies from the images saved from the camera data from simulation. [Code](https://github.com/NYU-robot-learning/Allegro-Sim-Policies/tree/main)
3) Libero-Sim: We use BC with a GMM head to train imitation learning policies for the data collected in Libero-Sim environment.[Code](https://github.com/NYU-robot-learning/LiberoSim-Policies)
