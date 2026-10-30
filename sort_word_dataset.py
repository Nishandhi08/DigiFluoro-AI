import os
import shutil
import cv2
import numpy as np

# 1. PATH CONFIGURATIONS
images_dir = r"C:\Users\POORANA RAJENDRAN\Downloads\doi_10_5061_dryad_5dv41nsfj__v20241013\images_measured"
target_base_dir = r"C:\Users\POORANA RAJENDRAN\OneDrive\Desktop\Urine_Test_AI\Dataset"

target_sensitive_dir = os.path.join(target_base_dir, "Sensitive")
target_resistant_dir = os.path.join(target_base_dir, "Resistant")

os.makedirs(target_sensitive_dir, exist_ok=True)
os.makedirs(target_resistant_dir, exist_ok=True)

# Rebuild both class folders so old classifications cannot remain after rerunning.
for target_dir in (target_sensitive_dir, target_resistant_dir):
    for existing_name in os.listdir(target_dir):
        existing_path = os.path.join(target_dir, existing_name)
        if os.path.isfile(existing_path):
            os.remove(existing_path)

print("🚀 Activating Bug-Free Color-Spectrum Clustering Engine...")

if not os.path.exists(images_dir):
    print(f"❌ Error: Source image folder path not found at {images_dir}")
else:
    actual_images = os.listdir(images_dir)
    sensitive_count = 0
    resistant_count = 0
    skipped_count = 0
    unclassified_count = 0
    
    for img_name in actual_images:
        if img_name.startswith(('.', '~$', '__MACOSX')):
            continue
            
        source_file_path = os.path.join(images_dir, img_name)
        img = cv2.imread(source_file_path)
        
        if img is None:
            skipped_count += 1
            continue
            
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, w, _ = hsv_img.shape
        
        center_roi = hsv_img[int(h*0.25):int(h*0.75), int(w*0.25):int(w*0.75)]
        mean_hsv = center_roi.mean(axis=(0,1))
        
        # Ignore grayscale scans; they do not contain enough color information to classify a well.
        pure_hue_value = mean_hsv[0]
        saturation_value = mean_hsv[1]

        if saturation_value < 40:
            unclassified_count += 1
            continue
        
        # Blue wells are sensitive; pink, purple, and red wells are resistant.
        if 85 <= pure_hue_value <= 130:
            shutil.copy(source_file_path, os.path.join(target_sensitive_dir, img_name))
            sensitive_count += 1
        elif pure_hue_value >= 140 or pure_hue_value <= 10:
            shutil.copy(source_file_path, os.path.join(target_resistant_dir, img_name))
            resistant_count += 1
        else:
            unclassified_count += 1

    print("---")
    print("✨ Colorimetric Sorting Operation Completed Successfully!")
    print(f"🔵 Sensitive (Effective Blue/Purple Wells) saved: `{sensitive_count}` files.")
    print(f"🔴 Resistant (Non-Effective Pink/Red Wells) saved: `{resistant_count}` files.")
    print(f"⏭️ Skipped unreadable files: `{skipped_count}` files.")
    print(f"⚪ Unclassified low-confidence/grayscale files: `{unclassified_count}` files.")
