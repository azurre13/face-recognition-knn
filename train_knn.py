import cv2
import os
import numpy as np
import pickle

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# =========================
# LOAD FACE DETECTOR
# =========================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# =========================
# FUNGSI EKSTRAKSI FITUR
# =========================
def extract_face_features(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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
        return face.flatten()

    return None

# =========================
# LOAD DATASET
# =========================
dataset_path = r"C:\FaceRecognition_KNN\dataset"

X = []
y = []
label = 0
label_names = {}

for person_name in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person_name)

    if not os.path.isdir(person_folder):
        continue

    label_names[label] = person_name
    count = 0

    for img_name in os.listdir(person_folder):
        if count >= 30:   # batasi 30 gambar / orang
            break

        img_path = os.path.join(person_folder, img_name)
        print("Processing:", img_path)

        features = extract_face_features(img_path)
        if features is not None:
            X.append(features)
            y.append(label)
            count += 1

    label += 1

X = np.array(X)
y = np.array(y)

print("\nTotal data:", len(X))

# =========================
# NORMALISASI FITUR
# =========================
scaler = StandardScaler()
X = scaler.fit_transform(X)

# =========================
# SPLIT DATA
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# TRAIN KNN
# =========================
knn = KNeighborsClassifier(
    n_neighbors=5,
    metric="euclidean"
)

knn.fit(X_train, y_train)

# =========================
# EVALUASI
# =========================
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Akurasi sistem:", accuracy * 100, "%")

# =========================
# SIMPAN MODEL
# =========================
with open("knn_model.pkl", "wb") as f:
    pickle.dump((knn, label_names, scaler), f)

print("Model KNN berhasil disimpan.")
