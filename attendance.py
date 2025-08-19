import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
from ultralytics import YOLO

# -----------------------------
# Load YOLO model (for object detection)
# -----------------------------
yolo_model = YOLO("yolo11n.pt")

# -----------------------------
# Face dataset setup
# -----------------------------
validExtensions = ('.jpeg', '.jpg', '.png', '.gif')
path = "data"
images = []
peoplesNames = []

for file in os.listdir(path):
    if file.endswith(validExtensions):
        images.append(file)
        peoplesNames.append(os.path.splitext(file)[0])

# -----------------------------
# Encode known faces
# -----------------------------
encodeList = []
def imageEncoder(images):
    for image in images:
        img = face_recognition.load_image_file(f'data/{image}')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faceencodings = face_recognition.face_encodings(img)
        if faceencodings:
            encodeList.append(faceencodings[0])
        else:
            print(f"No face found in {image}")
    return encodeList

knownFaces = imageEncoder(images)
print("Encoding complete. Known faces:", peoplesNames)

# -----------------------------
# Attendance logging
# -----------------------------
def checkAttendance(matchName):
    with open('Attendance.csv', 'a+') as f:
        f.seek(0)
        myAttendance = f.readlines()
        class_Name_List = [line.split(',')[0] for line in myAttendance]

        if matchName not in class_Name_List:
            now = datetime.now()
            dtString = now.strftime('%Y-%m-%d %H:%M:%S')
            f.writelines(f'\n{matchName},{dtString}')

# -----------------------------
# Webcam loop
# -----------------------------
videocapture = cv2.VideoCapture(1)

while True:
    success, img = videocapture.read()
    if not success:
        print("Failed to grab frame")
        break

    # -----------------------------
    # YOLO OBJECT DETECTION
    # -----------------------------
    yolo_results = yolo_model.predict(img, verbose=False)
    for r in yolo_results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = yolo_model.names[cls]
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Draw YOLO detections
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(img, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # -----------------------------
    # OPENCV + FACE_RECOGNITION
    # -----------------------------
    imgSmall = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
    imgSmall = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)

    facesCurrentFrame = face_recognition.face_locations(imgSmall)
    faceEncodingsCurrentFrame = face_recognition.face_encodings(imgSmall, facesCurrentFrame)

    for faceEncoding, faceLocation in zip(faceEncodingsCurrentFrame, facesCurrentFrame):
        matches = face_recognition.compare_faces(knownFaces, faceEncoding)
        faceDistance = face_recognition.face_distance(knownFaces, faceEncoding)
        matchIndex = np.argmin(faceDistance)

        if matches[matchIndex]:
            matchName = peoplesNames[matchIndex].upper()
            checkAttendance(matchName)
        else:
            matchName = "UNKNOWN"

        # Scale back face location
        y1, x2, y2, x1 = faceLocation
        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4

        # Draw face box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, matchName, (x1 + 6, y2 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    # -----------------------------
    # Show combined output
    # -----------------------------
    cv2.imshow("YOLO + Face Recognition", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

videocapture.release()
cv2.destroyAllWindows()









