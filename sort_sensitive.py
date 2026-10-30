import os
import shutil

# 1. Source and Target Exact System Paths Mapping
# Download panna mixed images extract folder path
mixed_dataset_dir = r"C:\Users\POORANA RAJENDRAN\Downloads\doi_10_5061_dryad_5dv41nsfj__v20241013\images_original"

# Custom Dataset dynamic base targeting path configuration
dataset_base = r"C:\Users\POORANA RAJENDRAN\OneDrive\Desktop\Urine_Test_AI\Dataset"
resistant_dir = os.path.join(dataset_base, "Resistant")
sensitive_dir = os.path.join(dataset_base, "Sensitive")

# Create Sensitive folder array if it doesn't exist layout logic
os.makedirs(sensitive_dir, exist_ok=True)

# 2. Extract existing Resistant items into memory hash map framework
resistant_images = set(os.listdir(resistant_dir))

print("⏳ Scanning source folder for sensitive profiles allocation workflow...")
count = 0

# 3. Running loop sorting exclusion tracking code execution logic
for filename in os.listdir(mixed_dataset_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        # Check matching filter tracking array parameters
        if filename not in resistant_images:
            src_path = os.path.join(mixed_dataset_dir, filename)
            dest_path = os.path.join(sensitive_dir, filename)
            
            shutil.copy(src_path, dest_path)
            print(f"✅ Moved to Sensitive: {filename}")
            count += 1

print(f"\n🎉 Done! Total {count} images successfully structured in Dataset/Sensitive folder.")
