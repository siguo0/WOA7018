from flask import Flask, jsonify, request
from flask_cors import CORS
import base64
from PIL import Image
from io import BytesIO
import numpy as np
import RoboticMTCNN

app = Flask(__name__)
CORS(app)  # 启用CORS插件


@app.route('/face_detect', methods=['GET', 'POST'])
def hello():
    try:
        name = None
        json_data = request.files
        json_data = dict(json_data)
        state_data = dict(request.form)
        cam_state = eval(state_data['state'])
        if cam_state:
            name = state_data['name']
        images = []
        for image_idx in json_data:
            print(json_data[image_idx])
            image_data = base64.b64decode(json_data[image_idx].read())
            image = Image.open(BytesIO(image_data))
            images.append(np.array(image))
        state = RoboticMTCNN.process_images(images, cam_state, name)

        return jsonify(state)
    except Exception as e:
        return jsonify({"error format": str(e)})

@app.route('/')
def init():
    return 'Hello, World! This is my Flask app.'

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8087)

