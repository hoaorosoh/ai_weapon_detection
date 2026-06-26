from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import ray
import cv2
import numpy as np
import shutil
import os
import itertools
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "best.pt"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)
#ray initialization 
@ray.remote(num_gpus=0) #1 gpu accross 2 workers

class WeaponDetectorWorker:
    def __init__(self):
        self.model = YOLO(str(MODEL_PATH))

    def process_image(self, image_bytes):
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        results = self.model.predict(img, conf=0.5, verbose=False)
        annotated_frame = results[0].plot()
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        return buffer.tobytes()

    def process_video(self, input_path, output_path):
        cap = cv2.VideoCapture(input_path)
        fourcc = cv2.VideoWriter_fourcc(*'avc1') 
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
                
            results = self.model.predict(frame, conf=0.5, verbose=False)
            annotated_frame = results[0].plot()
            out.write(annotated_frame)

        cap.release()
        out.release()
        return True 
    
    def process_stream_frame(self, image_bytes):
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return b""
        results = self.model.predict(img, conf=0.4, verbose=False)
        annotated_frame = results[0].plot()
        success, buffer = cv2.imencode('.jpg', annotated_frame)
        return buffer.tobytes()
    
TEMP_DIR = os.path.abspath("temp_files")
os.makedirs(TEMP_DIR, exist_ok=True)

ray.init(ignore_reinit_error=True)
num_workers = 1
workers = [WeaponDetectorWorker.remote() for _ in range(num_workers)]
worker_cycle = itertools.cycle(workers) # changes between workers 1 and 2 accordingly

@app.post("/detect")
async def detect_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    selected_worker = next(worker_cycle)
    ref = selected_worker.process_image.remote(image_bytes)
    annotated_image_bytes = await ref 
    return Response(content=annotated_image_bytes, media_type="image/jpeg")

@app.post("/video")
async def detect_video(file: UploadFile = File(...)):
    input_path = os.path.join(TEMP_DIR, f"in_{file.filename}")
    output_path = os.path.join(TEMP_DIR, f"out_{file.filename}")
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    selected_worker = next(worker_cycle)
    await selected_worker.process_video.remote(input_path, output_path)
    os.remove(input_path)
    return FileResponse(output_path, media_type="video/mp4")

@app.websocket("/ws/live")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session_worker = next(worker_cycle) 
    try:
        while True:
            bytes_data = await websocket.receive_bytes()
            processed_bytes = await session_worker.process_stream_frame.remote(bytes_data)
            await websocket.send_bytes(processed_bytes)
    except WebSocketDisconnect:
        print("Live stream disconnected.")