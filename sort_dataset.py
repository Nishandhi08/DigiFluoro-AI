import os
import shutil
import pandas as pd

# 1. FIXED WORKSPACE PATHS
base_downloads_path = r"C:\Users\POORANA RAJENDRAN\Downloads\doi_10_5061_dryad_5dv41nsfj__v20241013"
target_base_dir = r"C:\Users\POORANA RAJENDRAN\OneDrive\Desktop\Urine_Test_AI\Dataset"

# Automatically create local folders
target_sensitive_dir = os.path.join(target_base_dir, "Sensitive")
target_resistant_dir = os.path.join(target_base_dir, "Resistant")
os.makedirs(target_sensitive_dir, exist_ok=True)
os.makedirs(target_resistant_dir, exist_ok=True)

# Path mapping for subfolders inside your Downloads
tables_folder_path = os.path.join(base_downloads_path, "tables")
images_folder_path = os.path.join(base_downloads_path, "images_measured")

# 2. AUTO-DETECT SPREADSHEET (No matter what it is named!)
csv_file_target = None
if os.path.exists(tables_folder_path):
    for current_file in os.listdir(tables_folder_path):
        if current_file.endswith(".csv"):
            csv_file_target = os.path.join(tables_folder_path, current_file)
            break

if not csv_file_target:
    print("❌ Error: Could not find any .csv files inside your tables folder. Please double check that you extracted Tables.zip!")
else:
    print(f"📡 Found spreadsheet file target: {os.path.basename(csv_file_target)}")
    try:
        # Load up the dataset tracking matrix
        df = pd.read_csv(csv_file_target)
        
        # Clean up column names to avoid matching errors
        df.columns = [col.strip().lower() for col in df.columns]
        
        # Find which column holds filenames and results automatically
        filename_col = [c for c in df.columns if 'file' in c or 'name' in c or 'id' in c][0]
        status_col = [c for c in df.columns if 'status' in c or 'result' in c or 'phenotype' in c or 'susceptibility' in c or 'sir' in c][0]
        
        print(f"🔗 Mapping column [{filename_col}] to column [{status_col}]...")
        success_count = 0
        
        for index, row_entry in df.iterrows():
            raw_file_name = str(row_entry[filename_col])
            label = str(row_entry[status_col]).strip().upper()
            
            # Format files accurately matching the dataset patterns
            final_filename = raw_file_name if raw_file_name.endswith(('.png', '.jpg', '.jpeg')) else f"{raw_file_name}.png"
            source_file_location = os.path.join(images_folder_path, final_filename)
            
            # If the raw extension is missing, check if it works with .png
            if not os.path.exists(source_file_location):
                source_file_location = os.path.join(images_folder_path, raw_file_name)
            
            if os.path.exists(source_file_location):
                # Route files into matching target bins
                if label in ['SENSITIVE', 'S', '0', 'SUSCEPTIBLE', 'SENS']:
                    shutil.copy(source_file_location, os.path.join(target_sensitive_dir, os.path.basename(source_file_location)))
                    success_count += 1
                elif label in ['RESISTANT', 'R', '1', 'RESIST']:
                    shutil.copy(source_file_location, os.path.join(target_resistant_dir, os.path.basename(source_file_location)))
                    success_count += 1
                    
        print(f"✨ Success! Segregated and stored `{success_count}` real validation images directly inside your target dataset folders.")
        
    except Exception as e:
        print(f"❌ Execution crash encountered: {e}")
