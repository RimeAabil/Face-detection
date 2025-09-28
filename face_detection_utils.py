import cv2


def draw_boundary(img, classifier, scaleFactor=1.1, minNeighbors=10, color=(255,0,0), text="Feature"):
    """Detect features, draw rectangles and return coordinates."""
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)
    coords = []

    for (x, y, w, h) in features:
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, text, (x, y - 4), cv2.FONT_HERSHEY_PLAIN, 0.8, color, 1, cv2.LINE_AA)
        coords.append([x, y, w, h])

    return coords, img


def detect_faces_and_eyes(img, faceCascade, eyeCascade):
    """Detect faces and eyes in an image and return annotated image."""
    colors = {"face": (255, 0, 0), "eye": (0, 255, 0)}
    faces, img = draw_boundary(img, faceCascade, color=colors["face"], text="Face")

    for (x, y, w, h) in faces:
        face_roi = img[y:y+h, x:x+w]
        eyes, face_roi = draw_boundary(face_roi, eyeCascade, color=colors["eye"], text="Eye")
        img[y:y+h, x:x+w] = face_roi

    return img


def start_webcam(faceCascade, eyeCascade):
    """Start webcam and show real-time face and eye detection."""
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, img = video_capture.read()
        if not ret:
            break

        img = detect_faces_and_eyes(img, faceCascade, eyeCascade)
        cv2.imshow("Face and Eye Detection", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()