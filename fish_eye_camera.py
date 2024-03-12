import hydra
from openteach.components.initializers import FishEyeCameras

@hydra.main(version_base = '1.2', config_path = 'configs', config_name = 'fisheyecamera')
def main(configs):
    cameras =FishEyeCameras(configs)
    processes = cameras.get_processes()

    for process in processes:
        process.start()

    for process in processes:
        process.join()

if __name__ == '__main__':
    main()