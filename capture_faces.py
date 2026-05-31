import cv2
import os

# Student details — change these for each student
# student_id = 1
# student_name = "Jyoti"
# roll_no = "22501102"
# department = "Computer Science"

# student_id = 2
# student_name = "Niharika"
# roll_no = "22501099"
# department = "Computer Science"

student_id = 3
student_name = "Nidhi"
roll_no = "22501098"
department = "Computer Science"

# student_id = 4
# student_name = "Aman"
# roll_no = "22501096"
# department = "Computer Science"


# Create folder for this student
save_path = f"dataset/{student_id}_{student_name}"
os.makedirs(save_path, exist_ok=True)

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Open webcam
cap = cv2.VideoCapture(0)
count = 0
total_photos = 100  # photos to capture per student

print(f"Capturing photos for {student_name}...")
print("Press Q to quit early")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        count += 1
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (100, 100))

        # Save face image
        filename = f"{save_path}/{count}.jpg"
        cv2.imwrite(filename, face)

        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"Photos: {count}/{total_photos}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Capturing Faces - Press Q to quit", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if count >= total_photos:
        print(f"Done! {count} photos saved for {student_name}")
        break

cap.release()
cv2.destroyAllWindows()