from ultralytics import YOLO
from multiprocessing import freeze_support


model = YOLO(r"C:\Users\admin\Desktop\yolo11n.pt")
model.to("cuda")
if __name__ == '__main__':
    results = model.train(data=r"C:\Users\admin\Desktop\Model\datasets\prepared\data.yaml", epochs=1000, imgsz=640,verbose=True)
    #freeze_support()