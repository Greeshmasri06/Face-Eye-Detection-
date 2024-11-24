import cv2
import os
import sys

# File paths
face_cascade_path = r"C:\Users\lenovo\Desktop\NIT FILES\21st- python to mysql connection\haar cascade classifier basic project\Haarcascades\haarcascade_frontalface_default.xml"
eye_cascade_path = r"C:\Users\lenovo\Desktop\NIT FILES\21st- python to mysql connection\haar cascade classifier basic project\Haarcascades\haarcascade_eye.xml"

# Verify paths
if not os.path.exists(face_cascade_path):
    print(f"Error: Face cascade file not found at {face_cascade_path}")
    sys.exit()

if not os.path.exists(eye_cascade_path):
    print(f"Error: Eye cascade file not found at {eye_cascade_path}")
    sys.exit()

# Load Haar cascade files
face_cascade = cv2.CascadeClassifier(face_cascade_path)
eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

# Verify cascade loading
if face_cascade.empty():
    print("Error: Could not load face cascade classifier.")
    sys.exit()

if eye_cascade.empty():
    print("Error: Could not load eye cascade classifier.")
    sys.exit()

# Function to detect faces and eyes
def detect_faces_and_eyes(gray, frame):
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
    return frame

# Initialize webcam
video_capture = cv2.VideoCapture(0)

# Check webcam access
if not video_capture.isOpened():
    print("Error: Could not access the webcam. Ensure it's connected.")
    video_capture.release()
    sys.exit()

print("Webcam opened successfully. Starting face and eye detection...")

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Error: Failed to capture frame. Exiting...")
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    result_frame = detect_faces_and_eyes(gray, frame)
    cv2.imshow('Face and Eye Detection', result_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break

# Release resources
video_capture.release()
cv2.destroyAllWindows()











