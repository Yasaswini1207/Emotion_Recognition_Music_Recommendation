import cv2
import mediapipe as mp
import os

# Initialize MediaPipe solutions
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Function to open the webcam and collect data
def collect_data():
    # Ask user for label name
    label = input("Enter the emotion label: ").strip().lower()

    # Create dataset folder if it doesn't exist
    dataset_path = "Dataset"
    label_path = os.path.join(dataset_path, label)
    os.makedirs(label_path, exist_ok=True)

    # Try different camera indices if needed (0, 1, or 2)
    cap = cv2.VideoCapture(0)  # Change to 1 or 2 if the camera doesn't open

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    frame_count = 0

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose, \
         mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands, \
         mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break

            # Convert frame to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Process detections
            pose_results = pose.process(image)
            hands_results = hands.process(image)
            face_results = face.process(image)

            # Convert back to BGR for OpenCV
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Draw Face Landmarks
            if face_results.multi_face_landmarks:
                for face_landmarks in face_results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(image, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)

            # Draw Body Pose Landmarks
            if pose_results.pose_landmarks:
                mp_drawing.draw_landmarks(image, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Draw Hand Landmarks
            if hands_results.multi_hand_landmarks:
                for hand_landmarks in hands_results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Save frame as dataset image
            frame_count += 1
            img_path = os.path.join(label_path, f"{label}_{frame_count}.jpg")
            cv2.imwrite(img_path, image)

            # Show real-time webcam feed
            cv2.imshow('Collecting Data - Press "q" to Stop', image)

            # Press 'q' to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Data collection complete! {frame_count} images saved in '{label_path}'.")

# Run the function
collect_data()
