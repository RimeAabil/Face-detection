import os
import cv2
import pickle

from face_detection_utils import detect_faces_and_eyes

import face_recognition


DATASET_FOLDER = 'data/users/'
EMBEDDINGS_FILE = os.path.join(DATASET_FOLDER, "embeddings.pkl")
os.makedirs(DATASET_FOLDER, exist_ok=True)


# Load cascades
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eyeCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")


def capture_image(prompt="Look at the camera and press 'c'"):
    cap = cv2.VideoCapture(0)
    print(prompt)

    while True:
        ret, frame = cap.read()  # ret → a boolean (True or False) that tells you whether the frame was successfully captured.
        if not ret:
            break

        frame = detect_faces_and_eyes(frame, faceCascade, eyeCascade)

        cv2.imshow("Capture", frame)
        if cv2.waitKey(1) & 0xFF == ord('c'):
            img = frame.copy()
            break

    cap.release()
    cv2.destroyAllWindows()
    return img



def save_embeddings(user_name, face_encoding):
    """saves a user’s face encoding (a vector that represents their face) under their user_name."""
    if os.path.exists(EMBEDDINGS_FILE):
        with open(EMBEDDINGS_FILE, "rb") as f:  # rb: read binary (pickle works with bytes)
            data = pickle.load(f)  # reads (deserializes) the Python object stored inside (dict of {user_name: encoding}).
    else:
        data = {}   # dictionary of all stored users and their encodings. e.g, data = {"Alice": [...], "Bob": [...]}

    data[user_name] = face_encoding

    with open(EMBEDDINGS_FILE, "wb") as f:
        pickle.dump(data, f)
    

def load_embeddings():
    if os.path.exists(EMBEDDINGS_FILE):
        with open(EMBEDDINGS_FILE, 'rb') as f:
            return pickle.load(f)
    return {}


# LOGIN
def login_user():
    embeddings = load_embeddings()
    if len(embeddings)==0:
        print("No users registred yet.")
        return
    
    img = capture_image("Login: Look at the camera and press 'c'")
    
    # 1. Detect faces
    faces = face_recognition.face_locations(img)
    if len(faces)==0:  # faces is a Python list of tuples. Each tuple is (top, right, bottom, left) for a detected face.
        print("No faces detected.")
        return

    print(f"{len(faces)} face(s) detected")

    # 2. Encode face
    face_encoding = face_recognition.face_encodings(img, faces)[0]

    # 3. Compare with stored embeddings
    for name, stored_encoding in embeddings.items():
        match = face_recognition.compare_faces([stored_encoding], face_encoding, tolerance = 0.5)
        if match[0]:
            print(f"Welcome, {name}.")
            return
    print("Face not recognized.")


def register_user():
    user_name = input("Enter your name: ")
    img = capture_image(f"Registering {user_name}: Look at the camera and press 'c'")

    # Detect face
    faces = face_recognition.face_locations(img)
    if len(faces)==0:
        print("No face detected. Try again")
        return
    
    face_encoding = face_recognition.face_encodings(img, faces)[0]
   
    # Save embeddigns
    save_embeddings(user_name, face_encoding)
    print(f"{user_name} registered successfully!")



while True:
    choice = input("Choose: [1] Register  [2] Login  [q] Quit : ")
    if choice == "1":
        register_user()
    elif choice == "2":
        login_user()
    elif choice.lower() == "q":
        break
    else:
        print("Invalid choice. Try again.")