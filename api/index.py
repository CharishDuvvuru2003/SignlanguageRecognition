from flask import Flask, render_template, jsonify, request, send_file
from flask_cors import CORS
import cv2 as cv
import mediapipe as mp
import numpy as np
from slr.model.classifier import KeyPointClassifier
import base64
import torch
from transformers import pipeline
from datasets import load_dataset
import soundfile as sf
import os
import tempfile
import time
from threading import Lock

# English to Hindi dictionary
ENGLISH_TO_HINDI = {
    'A': 'अ', 'B': 'ब', 'C': 'क', 'D': 'ड', 'E': 'ए',
    'F': 'फ', 'G': 'ग', 'H': 'ह', 'I': 'इ', 'J': 'ज',
    'K': 'क', 'L': 'ल', 'M': 'म', 'N': 'न', 'O': 'ओ',
    'P': 'प', 'Q': 'क्यू', 'R': 'र', 'S': 'स', 'T': 'ट',
    'U': 'उ', 'V': 'व', 'W': 'व', 'X': 'एक्स', 'Y': 'य',
    'Z': 'ज़', ' ': ' '
}

app = Flask(__name__)
CORS(app)

print("INFO: Initializing System")

# Initialize TTS
print("Initializing TTS engine...")
synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts")
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
temp_dir = tempfile.mkdtemp()
tts_lock = Lock()
print(f"TTS engine initialized. Using temporary directory: {temp_dir}")

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

def generate_speech(text, language="english"):
    try:
        with tts_lock:
            timestamp = int(time.time() * 1000)
            filename = os.path.join(temp_dir, f"speech_{timestamp}.wav")
            speech_text = f"In {language}: {text}"
            speech = synthesiser(speech_text, forward_params={"speaker_embeddings": speaker_embedding})
            sf.write(filename, speech["audio"], samplerate=speech["sampling_rate"])
            return filename
    except Exception as e:
        print(f"TTS Error: {e}")
        return None

def cleanup_old_files():
    try:
        current_time = time.time()
        for filename in os.listdir(temp_dir):
            filepath = os.path.join(temp_dir, filename)
            if os.path.getmtime(filepath) < current_time - 300:  # 5 minutes old
                os.remove(filepath)
    except Exception as e:
        print(f"Cleanup error: {e}")

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
        encoded_data = base64_frame.split(',')[1]
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        frame = cv.imdecode(nparr, cv.IMREAD_COLOR)
        
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
                    english_text = labels[hand_sign_id]
                    hindi_text = ENGLISH_TO_HINDI.get(english_text, '')
                    return {
                        'english': english_text,
                        'hindi': hindi_text
                    }
        
        return {'english': '', 'hindi': ''}
    
    except Exception as e:
        print(f"Error processing frame: {e}")
        return {'english': '', 'hindi': ''}

# Main routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/sign')
def sign_page():
    return render_template('sign_language.html')

@app.route('/voice')
def voice_page():
    return render_template('voice_text.html')

# API routes
@app.route('/text_to_speech', methods=['POST'])
def text_to_speech():
    try:
        data = request.json
        text = data.get('text', '')
        language = data.get('language', 'english')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
            
        audio_file = generate_speech(text, language)
        if audio_file and os.path.exists(audio_file):
            cleanup_old_files()
            return send_file(audio_file, mimetype='audio/wav')
        else:
            return jsonify({'error': 'Failed to generate speech'}), 500
            
    except Exception as e:
        print(f"Error in text_to_speech: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/recognize_sign', methods=['POST'])
def recognize_sign():
    try:
        data = request.json
        frame_data = data.get('frame')
        
        if not frame_data:
            return jsonify({'error': 'No frame data received'}), 400
        
        prediction = process_frame(frame_data)
        
        # Generate speech for both English and Hindi
        english_audio = None
        hindi_audio = None
        
        if prediction['english']:
            english_audio = generate_speech(prediction['english'], 'English')
        if prediction['hindi']:
            hindi_audio = generate_speech(prediction['hindi'], 'Hindi')
        
        return jsonify({
            'success': True,
            'prediction': prediction,
            'audio': {
                'english': english_audio,
                'hindi': hindi_audio
            }
        })
    
    except Exception as e:
        print(f"Error in recognize_sign: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/translate', methods=['POST'])
def translate():
    try:
        text = request.json.get('text', '')
        hindi_text = ''.join(ENGLISH_TO_HINDI.get(c.upper(), c) for c in text)
        
        # Generate audio for both languages
        english_audio = generate_speech(text, 'English') if text else None
        hindi_audio = generate_speech(hindi_text, 'Hindi') if hindi_text else None
        
        return jsonify({
            'english': text,
            'hindi': hindi_text,
            'audio': {
                'english': english_audio,
                'hindi': hindi_audio
            }
        })
    except Exception as e:
        print(f"Error in translate: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)