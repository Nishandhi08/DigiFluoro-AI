import os
try:
    from ultralytics import YOLO
except ImportError:
    print("❌ Error: Ultralytics not found. Please install it using: pip install ultralytics")
    exit(1)

def train_well_detector():
    """
    Trains a YOLOv8 model to detect individual wells on a plate.
    Once trained, the model can detect all wells, and we can infer 
    the n x m matrix size by counting unique X and Y coordinates.
    """
    print("🚀 Initializing YOLOv8 Well Detector Training...")
    
    # We use the YOLOv8 nano model as a fast, lightweight starting point
    model = YOLO('yolov8n.pt') 
    
    # Path to your dataset configuration file (YAML)
    # The YAML file should point to your train/val images and define the 'well' class.
    dataset_yaml = 'plate_dataset.yaml'
    
    if not os.path.exists(dataset_yaml):
        print(f"❌ Error: {dataset_yaml} not found.")
        print("Please create this file and a dataset (images + YOLO format labels) before training.")
        return

    print("🧠 Starting training process. This might take a while depending on your GPU...")
    # Train the model
    results = model.train(
        data=dataset_yaml,
        epochs=50,       # Adjust based on dataset size
        imgsz=640,       # Image size
        batch=16,        # Batch size
        name='well_detector'
    )
    
    print("✨ Training Complete!")
    print("Your trained model is saved in the 'runs/detect/well_detector/weights/best.pt' folder.")
    print("You can now integrate 'best.pt' into app.py to auto-detect matrix sizes.")

if __name__ == '__main__':
    train_well_detector()
