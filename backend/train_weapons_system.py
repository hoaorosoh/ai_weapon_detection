import ray
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer, TorchConfig  # <-- 1. Add TorchConfig here
from ultralytics import YOLO
import os

def train_func(config):
    model = YOLO("yolov8n.pt") 
    model.train(
        data=config["data_path"],
        epochs=20,
        imgsz=640,
        batch=8,
        workers=0,
        device=0, # Still use your GPU!
        project="weapon_system_v1",
        name="ray_distributed_run"
    )

ray.init()

trainer = TorchTrainer(
    train_func,
    train_loop_config={
        "data_path": os.path.abspath("data/data.yaml") 
    },
    # 2. Add this line below to fix the NCCL error on Windows
    torch_config=TorchConfig(backend="gloo"), 
    scaling_config=ScalingConfig(
        num_workers=1,
        use_gpu=True
    )
)

print("🚀 Starting Ray Train with Gloo backend...")
result = trainer.fit()