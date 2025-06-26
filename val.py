from ultralytics import YOLO
from ultralytics.utils import DEFAULT_CFG
from datetime import datetime

if __name__ == '__main__':
    model = YOLO(r"/home/zhaotianyi/M2D-LIF/runs/FLIR/speed_test/weights/last.pt")
    data_yaml = r"/home/zhaotianyi/M2D-LIF/Feb_version/FLIR_data_Mul.yaml"
    batch = 64
    epochs = 1
    device = 4
    imgsz = 640

    DEFAULT_CFG.save_dir = f"../runs/v8m/val"

    model.val(data=data, batch=batch, imgsz=imgsz, device=device, save=True)

