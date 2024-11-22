import os

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

from ultralytics import YOLO

model = YOLO(r"C:\Users\admin\Desktop\yolo11n.pt")
model.to("cuda")
if __name__ == '__main__':
    results = model.train(data=r"C:\Users\admin\Desktop\Model\datasets\prepared\data.yaml", epochs=1000, imgsz=640,verbose=True)