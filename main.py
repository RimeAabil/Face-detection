from face_detection_utils import start_webcam
import cv2

# Load Haar cascades
faceCascade = cv2.CascadeClassifier("classifiers/haarcascade_frontalface_default.xml")
eyeCascade = cv2.CascadeClassifier("classifiers/haarcascade_eye.xml")

# Start webcam detection
start_webcam(faceCascade, eyeCascade)