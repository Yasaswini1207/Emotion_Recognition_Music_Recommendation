from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
import tensorflow as tf
import os
import mediapipe as mp
from pytube import Search

app = Flask(__name__)

# Load trained model
model_path = "emotion_recognition_model.keras"
if not os.path.exists(model_path):
    print("Error: Model not found! Train the model first.")
    exit()

model = tf.keras.models.load_model(model_path)

# Define image size
IMG_SIZE = 128

# Initialize MediaPipe solutions
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Get emotion labels from dataset dynamically
dataset_path = "Dataset"
emotion_labels = sorted([folder for folder in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, folder))])

if not emotion_labels:
    print("Error: No emotions found in the dataset folder!")
    exit()

print("Detected Emotion Classes:", emotion_labels)

@app.route("/")
def welcome_page():
    return render_template("welcome.html")

@app.route("/main")
def main_page():
    return render_template("main.html")

# Global variable to store predicted emotion
predicted_label = "Neutral"

def detect_emotion():
    global predicted_label  # Store predicted emotion globally
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose, \
         mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands, \
         mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break

            # Convert frame to RGB
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False

            # Process frame for keypoints
            pose_results = pose.process(image_rgb)
            hands_results = hands.process(image_rgb)
            face_results = face.process(image_rgb)

            image_rgb.flags.writeable = True
            frame = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            # Draw landmarks
            if face_results.multi_face_landmarks:
                for face_landmarks in face_results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)

            if pose_results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            if hands_results.multi_hand_landmarks:
                for hand_landmarks in hands_results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Resize and preprocess frame for prediction
            frame_resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            frame_resized = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            frame_resized = frame_resized / 255.0  # Normalize
            frame_resized = np.expand_dims(frame_resized, axis=0)  # Add batch dimension

            # Predict emotion
            predictions = model.predict(frame_resized)
            predicted_index = np.argmax(predictions)
            predicted_label = emotion_labels[predicted_index] if predicted_index < len(emotion_labels) else "Neutral"
            confidence = np.max(predictions) * 100  # Get confidence percentage

            # Display prediction
            cv2.putText(frame, f"Detected Emotion: {predicted_label} ({confidence:.2f}%)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            _, buffer = cv2.imencode(".jpg", frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()

@app.route("/video_feed")
def video_feed():
    return Response(detect_emotion(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/get_emotion")
def get_emotion():
    global predicted_label
    return jsonify({"emotion": f"Detected Emotion: {predicted_label}"})

@app.route("/get_youtube_link")
def get_youtube_link():
    query = request.args.get("query")
    if not query:
        return jsonify({"error": "Query parameter is required"}), 400

    search = Search(query)
    if not search.results:
        return jsonify({"error": "No results found"}), 404

    first_video = search.results[0]
    video_url = f"https://www.youtube.com/embed/{first_video.video_id}"

    return jsonify({"video_url": video_url})

if __name__ == "__main__":
    app.run(debug=True)

