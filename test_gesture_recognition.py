import cv2 as cv
import mediapipe as mp
import numpy as np
from slr.model.classifier import KeyPointClassifier
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
import os
import torch
import soundfile as sf
import winsound
from transformers import pipeline
from datasets import load_dataset
from time import sleep
import threading
import time
import tempfile

# English to Hindi dictionary
ENGLISH_TO_HINDI = {
    'A': 'अ', 'B': 'ब', 'C': 'क', 'D': 'ड', 'E': 'ए',
    'F': 'फ', 'G': 'ग', 'H': 'ह', 'I': 'इ', 'J': 'ज',
    'K': 'क', 'L': 'ल', 'M': 'म', 'N': 'न', 'O': 'ओ',
    'P': 'प', 'Q': 'क्यू', 'R': 'र', 'S': 'स', 'T': 'ट',
    'U': 'उ', 'V': 'व', 'W': 'व', 'X': 'एक्स', 'Y': 'य',
    'Z': 'ज़', ' ': ' '
}

class TTSEngine:
    def __init__(self):
        print("Initializing TTS engine...")
        self.synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts")
        embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
        self.speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
        self.is_speaking = False
        self.temp_dir = tempfile.mkdtemp()
        print(f"TTS engine initialized. Using temporary directory: {self.temp_dir}")

    def speak(self, text):
        """Convert text to speech and play the audio in a separate thread."""
        if self.is_speaking:
            return
        
        def speak_thread():
            self.is_speaking = True
            try:
                # Generate unique filename in temp directory
                timestamp = int(time.time() * 1000)
                filename = os.path.join(self.temp_dir, f"speech_{timestamp}.wav")
                
                # Generate speech
                speech = self.synthesiser(text, forward_params={"speaker_embeddings": self.speaker_embedding})
                
                # Save audio
                sf.write(filename, speech["audio"], samplerate=speech["sampling_rate"])
                
                # Play audio using winsound
                winsound.PlaySound(filename, winsound.SND_FILENAME)
                
                # Clean up
                try:
                    if os.path.exists(filename):
                        os.remove(filename)
                except Exception as e:
                    print(f"Warning: Could not remove temporary file {filename}: {e}")
            
            except Exception as e:
                print(f"TTS Error: {e}")
            finally:
                self.is_speaking = False

        threading.Thread(target=speak_thread, daemon=True).start()

    def cleanup(self):
        """Clean up temporary directory"""
        try:
            if os.path.exists(self.temp_dir):
                import shutil
                shutil.rmtree(self.temp_dir, ignore_errors=True)
        except Exception as e:
            print(f"Warning: Could not clean up temporary directory: {e}")

class HindiDisplay:
    def __init__(self):
        try:
            self.font = ImageFont.truetype("C:/Windows/Fonts/Nirmala.ttf", 32)
        except:
            self.font = ImageFont.load_default()

    def draw_text(self, image, english_text, position=(10, 30)):
        try:
            hindi_text = ENGLISH_TO_HINDI.get(english_text, '')
            
            # Draw English text
            cv.putText(image, f"English: {english_text}", (position[0], position[1]), 
                      cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            
            # Draw Hindi text
            image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            draw = ImageDraw.Draw(pil_image)
            draw.text((position[0], position[1] + 40), f"हिंदी: {hindi_text}", 
                     font=self.font, fill=(255, 255, 255))
            return cv.cvtColor(np.array(pil_image), cv.COLOR_RGB2BGR)
        except Exception as e:
            print(f"Error drawing text: {e}")
            return image

def main():
    print("Starting camera initialization...")
    
    cap = cv.VideoCapture(0, cv.CAP_DSHOW)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
    
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    
    keypoint_classifier = KeyPointClassifier()
    hindi_display = HindiDisplay()
    tts_engine = TTSEngine()
    
    try:
        with open('slr/model/label.csv', encoding='utf-8-sig') as f:
            labels = [row.strip() for row in f]
    except Exception as e:
        print(f"Error loading labels: {e}")
        return

    # Buffer for continuous letter detection
    letter_buffer = ""
    buffer_timeout = 20
    frame_counter = 0
    last_prediction = ""
    last_spoken_text = ""

    print("System ready - press ESC to exit")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv.flip(frame, 1)
            debug_frame = frame.copy()
            
            frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        debug_frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS
                    )
                    
                    landmark_list = []
                    for landmark in hand_landmarks.landmark:
                        x = min(int(landmark.x * frame.shape[1]), frame.shape[1] - 1)
                        y = min(int(landmark.y * frame.shape[0]), frame.shape[0] - 1)
                        landmark_list.append([x, y])
                    
                    processed_landmark_list = pre_process_landmark(landmark_list)
                    hand_sign_id = keypoint_classifier(processed_landmark_list)
                    
                    if 0 <= hand_sign_id < len(labels):
                        current_prediction = labels[hand_sign_id]
                        
                        # Buffer management
                        if current_prediction != last_prediction:
                            letter_buffer += current_prediction
                            frame_counter = 0
                        last_prediction = current_prediction
                        
                        # Update display and TTS
                        if frame_counter >= buffer_timeout and letter_buffer:
                            debug_frame = hindi_display.draw_text(debug_frame, letter_buffer)
                            
                            # Speak the text if it's different from last spoken
                            if letter_buffer != last_spoken_text:
                                # Speak English
                                tts_engine.speak(f"In English: {letter_buffer}")
                                sleep(1)  # Brief pause between languages
                                
                                # Get Hindi text
                                hindi_text = ''.join(ENGLISH_TO_HINDI.get(c, '') for c in letter_buffer)
                                if hindi_text.strip():
                                    tts_engine.speak(f"In Hindi: {hindi_text}")
                                
                                last_spoken_text = letter_buffer
                            
                            letter_buffer = ""
                        else:
                            current_text = letter_buffer + current_prediction
                            debug_frame = hindi_display.draw_text(debug_frame, current_text)
                        
                        frame_counter += 1
            
            cv.imshow('Sign Language Recognition', debug_frame)
            
            if cv.waitKey(1) & 0xFF == 27:
                break
    
    finally:
        print("Cleaning up...")
        cap.release()
        cv.destroyAllWindows()
        tts_engine.cleanup()

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

if __name__ == "__main__":
    main()