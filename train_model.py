import os
import cv2
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

base_project_dir = r"C:\Users\POORANA RAJENDRAN\OneDrive\Desktop\Urine_Test_AI"
dataset_dir = os.path.join(base_project_dir, "Dataset")

features = []
labels = []

print("🚀 Loading image arrays into Python 3.14 ML workspace...")

# Read sorted images from local directories
for target_class, class_label in [("Sensitive", 0), ("Resistant", 1)]:
    class_folder = os.path.join(dataset_dir, target_class)
    if os.path.exists(class_folder):
        for img_name in os.listdir(class_folder):
            img_path = os.path.join(class_folder, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            # Extract high-precision HSV metrics
            hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            h, w, _ = hsv_img.shape
            center_roi = hsv_img[int(h*0.25):int(h*0.75), int(w*0.25):int(w*0.75)]
            mean_metrics = center_roi.mean(axis=(0,1)) # Extract [H, S, V]
            
            features.append(mean_metrics)
            labels.append(class_label)

if len(features) == 0:
    print("❌ Error: No images found inside your Dataset/ folder. Run sort_word_dataset.py first!")
elif len(set(labels)) < 2:
    print("❌ Error: Training requires both Sensitive and Resistant images. Add blue and pink/purple well images, then rerun sort_word_dataset.py.")
else:
    X = np.array(features)
    y = np.array(labels)
    
    # Train a highly accurate Random Forest Boundary Classifier
    print(f"🧠 Training machine learning architecture on {len(X)} clinical samples...")
    ml_model = RandomForestClassifier(n_estimators=100, random_state=42)
    ml_model.fit(X, y)
    
    # Save the authentic AI brain matrix file
    model_output_path = os.path.join(base_project_dir, 'well_model.pkl')
    joblib.dump(ml_model, model_output_path)
    print(f"✨ Success! Authentic weight matrices compiled and saved as 'well_model.pkl'!")
