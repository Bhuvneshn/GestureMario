from flask import Flask, render_template, request
import numpy as np
import mediapipe as mp
import base64
import cv2
from PIL import Image
import io

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/image_prediction', methods=["GET", "POST"])
def predict_hand():
    if request.method == 'POST':
        b64_str = request.form['image']
        base64_img_string = b64_str.split(',')
        base64_img_string = base64_img_string[1]
        image = base64.b64decode(str(base64_img_string))

        mp_hands = mp.solutions.hands
        image = Image.open(io.BytesIO(image))
        image = np.asarray(image)
        image = cv2.flip(image, 1)
        with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:

            results = hands.process(image)
            if results.multi_handedness != None:
                if len(results.multi_handedness) == 2:
                    return ('Both Hands')
                else:
                    for idx, hand_handedness in enumerate(results.multi_handedness):
                        return str(hand_handedness.classification[0].label)
            else:
                return ('No Hands')


if __name__ == '__main__':
    app.run(debug=True)
