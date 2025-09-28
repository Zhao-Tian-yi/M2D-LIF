from ultralytics import YOLO
from ultralytics.utils import DEFAULT_CFG
from datetime import datetime

from ultralytics.models.yolo.obb import OBBTrainer
if __name__ == '__main__':

    model_s = YOLO(r"./ultralytics/cfg/models/v8/yolov8m-obb.yaml")
    DEFAULT_CFG.save_dir = f""
    model_s.train(
        data=r"./Drone_IR.yaml",
        imgsz=640,
        epochs=100,
        batch=4,
        device=3,
        lr0=0.001,
        augment=True,
        workers=4,
        save=True,
        amp=False,
        rect=True
    )