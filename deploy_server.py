import hydra
from openteach.components.deploy.deployer_xarm import DeployServer

@hydra.main(version_base = '1.2', config_path='./configs', config_name = 'deploy')
def deploy(configs):
    deploy_server_component = DeployServer(configs)
    deploy_server_component.stream()

if __name__ == '__main__':
    deploy()
