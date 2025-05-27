from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import cvlib as cv

# Load the pre-trained gender detection model
model = load_model('gender_detection_model.model')

# Open the webcam
webcam = cv2.VideoCapture(0)

# Define the classes
classes = ['man', 'woman']

# Loop through frames
while webcam.isOpened():
    # Read frame from webcam
    status, frame = webcam.read()

    if not status:
        print("Could not read frame")
        break

    # Apply face detection
    faces, confidence = cv.detect_face(frame)

    # Draw rectangles around detected faces
    for f in faces:
        (startX, startY, endX, endY) = f
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

    # Display the video stream with detected faces
    cv2.imshow("Realtime Face Detection", frame)

    # Check for key press
    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        # Capture the current frame for gender detection
        captured_frame = frame.copy()

        # Perform gender detection on the captured frame
        for f in faces:
            (startX, startY, endX, endY) = f
            face_crop = np.copy(captured_frame[startY:endY, startX:endX])

            if face_crop.shape[0] < 10 or face_crop.shape[1] < 10:
                continue

            # Preprocessing for gender detection model
            face_crop = cv2.resize(face_crop, (96, 96))
            face_crop = face_crop.astype("float") / 255.0
            face_crop = img_to_array(face_crop)
            face_crop = np.expand_dims(face_crop, axis=0)

            # Apply gender detection on face
            conf = model.predict(face_crop)[0]  # model.predict returns a 2D matrix, e.g., [[0.9999, 0.0001]]
            idx = np.argmax(conf)
            label = classes[idx]

            label_text = "{}: {:.2f}%".format(label, conf[idx] * 100)

            # Print gender label to the terminal
            print(f"Detected {label_text} at ({startX}, {startY}, {endX}, {endY})")

            # Annotate the captured frame with the gender label
            Y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.putText(captured_frame, label_text, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Display the captured frame with gender annotations
        cv2.imshow("Captured Frame Gender Detection", captured_frame)

    # Press "Q" to stop
    if key == ord('q'):
        break

# Release resources
webcam.release()
cv2.destroyAllWindows()