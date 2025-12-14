import cv2
import pickle
import numpy as np
from datetime import datetime

# =========================
# LOAD MODEL
# =========================
with open("knn_model.pkl", "rb") as f:
    knn, label_names, scaler = pickle.load(f)

# =========================
# LOAD FACE DETECTOR
# =========================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# =========================
# FUNGSI ABSENSI
# =========================
def save_absensi(nama):
    with open("absensi.csv", "a") as f:
        waktu = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"{nama},{waktu}\n")

# =========================
# WEBCAM
# =========================
cap = cv2.VideoCapture(0)
recorded = set()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=4,
        minSize=(80, 80)
    )

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (100, 100))
        features = face.flatten().reshape(1, -1)
        features = scaler.transform(features)

        pred = knn.predict(features)[0]
        name = label_names[pred]

        if name not in recorded:
            save_absensi(name)
            recorded.add(name)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(
            frame,
            name,
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

    cv2.imshow("Face Recognition KNN - Absensi", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
