import cv2
import numpy as np
import os
import pickle
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import json

# Load dataset
print("Loading dataset...")
faces = []
labels = []
student_info = {}

dataset_path = "dataset"

for folder in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, folder)
    if os.path.isdir(folder_path):
        # Get student ID and name from folder name
        parts = folder.split("_", 1)
        student_id = int(parts[0])
        student_name = parts[1]
        student_info[student_id] = student_name

        for img_file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (100, 100))
                faces.append(img.flatten())
                labels.append(student_id)

print(f"Total images loaded: {len(faces)}")
print(f"Students found: {student_info}")

# Convert to numpy arrays
X = np.array(faces)
y = np.array(labels)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Apply PCA — retaining 95% variance
print("Applying PCA...")
pca = PCA(n_components=0.95, whiten=True)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
print(f"PCA components: {pca.n_components_}")

# Train KNN
print("Training KNN...")
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_pca, y_train)
knn_acc = accuracy_score(y_test, knn.predict(X_test_pca))
print(f"KNN Accuracy: {knn_acc:.2f}")

# Train SVM
print("Training SVM...")
svm = SVC(kernel='rbf', probability=True)
svm.fit(X_train_pca, y_train)
svm_acc = accuracy_score(y_test, svm.predict(X_test_pca))
print(f"SVM Accuracy: {svm_acc:.2f}")

# Train Random Forest
print("Training Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_pca, y_train)
rf_acc = accuracy_score(y_test, rf.predict(X_test_pca))
print(f"Random Forest Accuracy: {rf_acc:.2f}")

# Save best model
print("Saving models...")
with open("model/pca.pkl", "wb") as f:
    pickle.dump(pca, f)

with open("model/svm.pkl", "wb") as f:
    pickle.dump(svm, f)

with open("model/knn.pkl", "wb") as f:
    pickle.dump(knn, f)

with open("model/rf.pkl", "wb") as f:
    pickle.dump(rf, f)

with open("model/student_info.json", "w") as f:
    json.dump(student_info, f)

print("\nAll models saved successfully!")
print(f"\nResults Summary:")
print(f"KNN Accuracy:          {knn_acc:.2f}")
print(f"SVM Accuracy:          {svm_acc:.2f}")
print(f"Random Forest Accuracy: {rf_acc:.2f}")