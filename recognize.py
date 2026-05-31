import cv2
import numpy as np
import pickle
import json
import os
import csv
from datetime import datetime

# Load models
with open("model/pca.pkl", "rb") as f:
    pca = pickle.load(f)

with open("model/svm.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/student_info.json", "r") as f:
    student_info = json.load(f)

# Convert keys to int
student_info = {int(k): v for k, v in student_info.items()}

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Attendance tracking
attendance_marked = set()

def mark_attendance(student_id, student_name):
    if student_id not in attendance_marked:
        attendance_marked.add(student_id)
        now = datetime.now()
        date = now.strftime("%Y-%m-%d")
        time = now.strftime("%H:%M:%S")

        os.makedirs("attendance", exist_ok=True)
        filename = f"attendance/attendance_{date}.csv"

        file_exists = os.path.isfile(filename)
        with open(filename, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["ID", "Name", "Date", "Time"])
            writer.writerow([student_id, student_name, date, time])

        print(f"Attendance marked for {student_name} at {time}")

# Open webcam
cap = cv2.VideoCapture(0)
print("Starting face recognition...")
print("Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (100, 100))
        face_flat = face.flatten().reshape(1, -1)

        # Apply PCA
        face_pca = pca.transform(face_flat)

        # Predict
        prediction = model.predict(face_pca)[0]
        confidence = model.predict_proba(face_pca).max()

        if confidence > 0.8:
            name = student_info.get(prediction, "Unknown")
            color = (0, 255, 0)
            label = f"ID:{prediction} Name:{name}"
            mark_attendance(prediction, name)
        else:
            name = "Unknown"
            color = (0, 0, 255)
            label = "Unknown"

        # Draw rectangle and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("FACETRACK - Press Q to quit", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Attendance saved!")