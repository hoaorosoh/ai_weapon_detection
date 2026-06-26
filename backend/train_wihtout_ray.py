from ultralytics import YOLO
import os

# 1. Load your existing "best" weights
# This is called 'Fine-tuning'
model = YOLO("models/best.pt")

# 2. Run 10 more epochs
# We keep workers=0 and a small batch size for Windows stability
model.train(
    data=os.path.abspath("data/data.yaml"), 
    epochs=10,
    imgsz=640,
    batch=8,      
    workers=0,    
    device=0,     # Use your NVIDIA GPU
    project="weapon_system_v1",
    name="refined_run",
    # Important: Since we already have a good model, 
    # we don't want the learning rate to be too high.
)