from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import cvlib as cv
                    
# load model
model = load_model('gender_detection_model.model')

# open webcam
webcam = cv2.VideoCapture(0)
    
classes = ['man','woman']

# loop through frames
while webcam.isOpened():

    # read frame from webcam 
    status, frame = webcam.read()

    # apply face detection
    face, confidence = cv.detect_face(frame)


    # loop through detected faces
    for idx, f in enumerate(face):

        # get corner points of face rectangle        
        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]

        # draw rectangle over face
        cv2.rectangle(frame, (startX,startY), (endX,endY), (0,255,0), 2)

        # crop the detected face region
        face_crop = np.copy(frame[startY:endY,startX:endX])

        if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
            continue

        # preprocessing for gender detection model
        face_crop = cv2.resize(face_crop, (96,96))
        face_crop = face_crop.astype("float") / 255.0
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis=0)

        # apply gender detection on face
        conf = model.predict(face_crop)[0] # model.predict return a 2D matrix, ex: [[9.9993384e-01 7.4850512e-05]]

        # get label with max accuracy
        idx = np.argmax(conf)
        label = classes[idx]

        label = "{}: {:.2f}%".format(label, conf[idx] * 100)

        Y = startY - 10 if startY - 10 > 10 else startY + 10

        # write label and confidence above face rectangle
        cv2.putText(frame, label, (startX, Y),  cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)
        

    # display output
    cv2.imshow("Realtime_Gender_Detection", frame)

    # press "Q" to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release resources
webcam.release()
cv2.destroyAllWindows()





'''
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


def is_new_face(face_crop, previous_faces, threshold=0.5):
    for prev_face in previous_faces:
        # Compute similarity between the new face and previous faces
        similarity = np.sum(np.square(face_crop - prev_face))
        if similarity < threshold:
            return False
    return True

# Initialize memory for previous faces
previous_faces = []

# Loop through frames
try:
    while webcam.isOpened():
        # Read frame from webcam
        status, frame = webcam.read()

        if not status:
            print("Could not read frame")
            break

        # Apply face detection
        faces, confidence = cv.detect_face(frame)

        # Loop through detected faces
        for idx, f in enumerate(faces):
            # Get corner points of face rectangle        
            (startX, startY) = f[0], f[1]
            (endX, endY) = f[2], f[3]

            # Draw rectangle over face
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

            # Crop the detected face region
            face_crop = np.copy(frame[startY:endY, startX:endX])

            if face_crop.shape[0] < 10 or face_crop.shape[1] < 10:
                continue

            # Preprocessing for gender detection model
            face_crop_resized = cv2.resize(face_crop, (96, 96))
            face_crop_resized = face_crop_resized.astype("float") / 255.0
            face_crop_resized = img_to_array(face_crop_resized)
            face_crop_resized = np.expand_dims(face_crop_resized, axis=0)

            # Check if the face is new
            new_face = is_new_face(face_crop_resized, previous_faces)

            # Apply gender detection on face
            conf = model.predict(face_crop_resized)[0]  # model.predict returns a 2D matrix, e.g., [[0.9999, 0.0001]]

            # Get label with max accuracy
            idx = np.argmax(conf)
            label = classes[idx]

            label_text = "{}: {:.2f}%".format(label, conf[idx] * 100)
            
            # Print gender label to terminal at least once for each face
            if new_face:
                print(f"Detected {label_text}")

            Y = startY - 10 if startY - 10 > 10 else startY + 10

            # Write label and confidence above face rectangle
            cv2.putText(frame, label_text, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # If the face is new, add it to the list of previous faces
            if new_face:
                previous_faces.append(face_crop_resized)
               
        # Display output
        cv2.imshow("Realtime_Gender_Detection", frame)

        # Press "Q" to stop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    # Release resources
    webcam.release()
    cv2.destroyAllWindows()
'''


