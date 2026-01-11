import cv2

def detect_faces():
    # 1. Load the pre-trained Haar Cascade classifier
    # OpenCV comes with this file built-in
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    # 2. Open the Webcam (0 is usually the default camera)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        return

    print("Face Detection Active! Press 'q' to exit.")

    while True:
        # 3. Read a frame from the camera
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # 4. Convert to grayscale (Face detection works better on black & white)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 5. Detect faces
        # scaleFactor=1.1: Reduces image size by 10% each pass to find faces
        # minNeighbors=5: Higher number = fewer detections but higher quality
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=20, minSize=(100, 100))
        # 6. Draw rectangles around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, "Face", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # 7. Show the result
        cv2.imshow('Face Detection - Task 5', frame)

        # 8. Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 9. Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_faces()