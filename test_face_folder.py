import cv2
import pickle
import numpy as np
import os
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
# FOLDER INPUT
# =========================
input_folder = r"C:\FaceRecognition_KNN\input_images"

if not os.path.exists(input_folder):
    print("Folder input_images tidak ditemukan.")
    exit()

recorded = set()

# =========================
# PROSES SEMUA FOTO
# =========================
for file_name in os.listdir(input_folder):
    file_path = os.path.join(input_folder, file_name)

    if not file_name.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    print("Memproses:", file_name)

    img = cv2.imread(file_path)
    if img is None:
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=4,
        minSize=(80, 80)
    )

    if len(faces) == 0:
        print("  Wajah tidak terdeteksi.")
        continue

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

        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(
            img,
            name,
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

    cv2.imshow("Face Recognition - Folder Input", img)
    cv2.waitKey(5000)  # tampil 0.5 detik per foto

cv2.destroyAllWindows()
print("Selesai memproses semua gambar.")
