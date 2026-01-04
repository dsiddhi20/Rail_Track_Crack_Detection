import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ====== CHANGE THIS PATH IF NEEDED ======
BASE_DATA_PATH = r"D:\Project\Rail_Track_Crack_Detection\data"

IMAGE_SIZE = 128

def load_data(base_path):
    data = []
    labels = []

    class_map = {
        "Defective": 1,
        "Non defective": 0
    }

    for class_name, label in class_map.items():
        folder_path = os.path.join(base_path, class_name)

        if not os.path.exists(folder_path):
            print(f"⚠️ Folder not found: {folder_path}")
            continue

        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
            img = img.flatten()

            data.append(img)
            labels.append(label)

    return np.array(data), np.array(labels)


# ====== LOAD DATA ======
X_train, y_train = load_data(os.path.join(BASE_DATA_PATH, "Train"))
X_val, y_val     = load_data(os.path.join(BASE_DATA_PATH, "Validation"))
X_test, y_test   = load_data(os.path.join(BASE_DATA_PATH, "Test"))

print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)


# ====== TRAIN MODEL ======
model = SVC(kernel="linear")
model.fit(X_train, y_train)
print("✅ Model trained successfully")


# ====== VALIDATION ======
val_pred = model.predict(X_val)
print("Validation Accuracy:", accuracy_score(y_val, val_pred))


# ====== TESTING ======
test_pred = model.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, test_pred))

print("\nClassification Report:\n")
print(classification_report(y_test, test_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, test_pred))


# ====== VISUALIZATION ======
correct = np.sum(test_pred == y_test)
incorrect = np.sum(test_pred != y_test)

plt.bar(["Correct", "Incorrect"], [correct, incorrect])
plt.title("Rail Track Crack Detection Results")
plt.show()

import pickle

with open("rail_track_crack_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved as rail_track_crack_model.pkl")
