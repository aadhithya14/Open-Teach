import hydra
from openteach.components.initializers import EmeetCameras

@hydra.main(version_base = '1.2', config_path = 'configs', config_name = 'emeetcamera')
def main(configs):
    cameras = EmeetCameras(configs)
    processes = cameras.get_processes()

    for process in processes:
        process.start()

    for process in processes:
        process.join()

if __name__ == '__main__':
    main()