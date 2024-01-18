from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
from flask import Flask, request, jsonify
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
        json_data = request.json
        json_data = dict(json_data)
        images = []
        for image_idx in json_data:
            image_data = base64.b64decode(json_data[image_idx][1])
            image = Image.open(BytesIO(image_data))
            images.append(np.array(image))
        state = RoboticMTCNN.process_images(images)

        return jsonify({"message": "Image received and processed successfully",
                        "state": state})
    except Exception as e:
        return jsonify({"error format": str(e)})

@app.route('/')
def init():
    return 'Hello, World! This is my Flask app.'

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8087)

