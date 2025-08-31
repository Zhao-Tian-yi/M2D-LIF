from ultralytics import YOLO
from ultralytics.utils import DEFAULT_CFG
from datetime import datetime

if __name__ == '__main__':

    data = r"./data/DroneVehicle.yaml"

    from ultralytics.models.yolo.obb import OBBValidator

    args = dict(model="./checkpoint/multimodal/DroneVehicle.pt", data=data,
                device=5,
                imgsz=640, batch=1, save=True, rect=True)
    validator = OBBValidator(args=args)
    validator(model=args["model"])


