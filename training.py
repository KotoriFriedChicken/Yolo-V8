from ultralytics import YOLO

# Load the pre-trained YOLOv8 model (pre-trained on a large dataset like COCO)
model = YOLO('yolov8n.pt')  # You can choose a different variant like yolov8s.pt, yolov8m.pt, etc.

# Train the model on your custom dataset
model.train(
    data='datasets/my_food_dataset/data.yaml',  # Path to your dataset configuration file
    epochs=20,  # Number of training epochs
    imgsz=640,  # Image size for training (can be adjusted)
    batch=4, # Batch size (can be adjusted based on your GPU)
    amp=True, # Enables mixed precision training
    augment=False  # Disable augmentation (artificially increase batch size by imagination)
)
