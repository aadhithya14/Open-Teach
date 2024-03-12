mkdir logs
nohup python3 robot_camera.py > logs/camera_log.txt &
cd server
nohup gunicorn -w 12 -b 0.0.0.0:5000 -k gevent --timeout 0 --worker-connections 2 'monitor:app' > ../logs/cam_server.txt &