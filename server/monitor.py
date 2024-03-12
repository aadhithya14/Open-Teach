import hydra
from flask import Flask, render_template, Response
from application_manager import MonitoringApplication


# Initializing the monitor class
hydra.initialize(config_path = '../configs', version_base = '1.2')
configs = hydra.compose('server')
monitor_info = MonitoringApplication(configs)

# Initializing the flask application
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', num_cams = monitor_info.num_cams)

@app.route('/cam_<int:id>_feed')
def video_feed(id):
    return Response(
        monitor_info.get_cam_streamer(id).yield_frames(), 
        mimetype = 'multipart/x-mixed-replace; boundary=frame'
    )

if __name__ == '__main__':
    app.run(threaded = True)