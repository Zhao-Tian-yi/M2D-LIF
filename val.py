from ultralytics import YOLO
from ultralytics.utils import DEFAULT_CFG
from datetime import datetime

if __name__ == '__main__':
    model = YOLO(r"")
    data = r"./data/FLIR.yaml"
    batch = 1
    epochs = 1
    device = 4
    imgsz = 640

    DEFAULT_CFG.save_dir = f"./runs/v8m/val"

    model.val(data=data, batch=batch, imgsz=imgsz, device=device, save=True)


