from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import cv2 as cv
import mediapipe as mp
import numpy as np
from slr.model.classifier import KeyPointClassifier
import base64

app = Flask(__name__)
CORS(app)

print("INFO: Initializing System")

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Initialize the classifier
keypoint_classifier = KeyPointClassifier()

# Load labels
try:
    with open('slr/model/label.csv', encoding='utf-8-sig') as f:
        labels = [row.strip() for row in f]
except Exception as e:
    print(f"Error loading labels: {e}")
    labels = []

def pre_process_landmark(landmark_list):
    temp_landmark_list = landmark_list.copy()
    base_x, base_y = landmark_list[0]
    
    for i in range(len(temp_landmark_list)):
        temp_landmark_list[i][0] = temp_landmark_list[i][0] - base_x
        temp_landmark_list[i][1] = temp_landmark_list[i][1] - base_y
    
    temp_landmark_list = np.array(temp_landmark_list).flatten()
    max_value = max(map(abs, temp_landmark_list))
    
    if max_value != 0:
        temp_landmark_list = temp_landmark_list / max_value
    
    return temp_landmark_list.tolist()

def process_frame(base64_frame):
    try:
        # Decode base64 image
        encoded_data = base64_frame.split(',')[1]
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        frame = cv.imdecode(nparr, cv.IMREAD_COLOR)
        
        # Convert to RGB for MediaPipe
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmark_list = []
                for landmark in hand_landmarks.landmark:
                    x = min(int(landmark.x * frame.shape[1]), frame.shape[1] - 1)
                    y = min(int(landmark.y * frame.shape[0]), frame.shape[0] - 1)
                    landmark_list.append([x, y])
                
                processed_landmark_list = pre_process_landmark(landmark_list)
                hand_sign_id = keypoint_classifier(processed_landmark_list)
                
                if 0 <= hand_sign_id < len(labels):
                    return labels[hand_sign_id]
        
        return ""
    
    except Exception as e:
        print(f"Error processing frame: {e}")
        return ""

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/sign')
def sign_page():
    return render_template('sign_language.html')

@app.route('/voice')
def voice_page():
    return render_template('voice_text.html')

@app.route('/recognize_sign', methods=['POST'])
def recognize_sign():
    try:
        data = request.json
        frame_data = data.get('frame')
        
        if not frame_data:
            return jsonify({'error': 'No frame data received'}), 400
        
        prediction = process_frame(frame_data)
        
        return jsonify({
            'success': True,
            'prediction': prediction
        })
    
    except Exception as e:
        print(f"Error in recognize_sign: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/translate', methods=['POST'])
def translate():
    text = request.json.get('text', '')
    return jsonify({'translation': text})

if __name__ == '__main__':
    app.run(debug=True)