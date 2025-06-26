import os

import warnings

warnings.filterwarnings("ignore", category=UserWarning)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.models.yolo.obb import OBBTrainer
from ultralytics.nn.tasks import attempt_load_one_weight
from ultralytics.utils import DEFAULT_CFG
from ultralytics import YOLO

if __name__ == '__main__':
    """
    1.修改trainer.py中的特征通道数
    2.model_s为要训练的模型
    3.更改写在类Distillation_loss的 channels_s和channels_t，
        将通道数改成自己模型的通道数, 路径在
        ultralytics/engine/trainer.py
    4.如果不需要蒸馏, 则直接把Distillation设置成None即可
    5.多模态蒸馏, 把Distillation设置为"MultiDistillation",
      并设置Teacher_Model_RGB和Teacher_Model_IR
      单模态蒸馏, 把Distillation设置为"SingleDistillation",
      并设置Teacher_Model
    Attributes:
        Distillation: the distillation model
        loss_type: MGD, CWD, PKD
        amp: Automatic Mixed Precision
    """

    # DroneVehicle TEACHER
    _, model_t_rgb = attempt_load_one_weight(
        r'/home/zhaotianyi/M2D-LIF/M2D-LIF_open_source/checkpoint/dv_ir.pt')
    model_t_rgb["model"].info()
    _, model_t_ir = attempt_load_one_weight(
        r'/home/zhaotianyi/M2D-LIF/M2D-LIF_open_source/checkpoint/dv_ir.pt')
    model_t_ir["model"].info()


    args = dict(
        model=r"/home/zhaotianyi/M2D-LIF/M2D-LIF_open_source/model_yaml_obb/yolov8_LIF_obb.yaml",
        data=r"/home/zhaotianyi/M2D-LIF/M2D-LIF_open_source/data/DroneVehicle.yaml",  # DroneVehicle
        Distillation="MultiDistillation",
        distill_weight=0.8,
        Teacher_Model_RGB=model_t_rgb["model"],
        Teacher_Model_IR=model_t_ir["model"],
        loss_type="CWD",
        amp=False,
        imgsz=640,
        epochs=200,
        batch=8,
        device=5,
        lr0=0.001,
        online=False,
        augment=True,
        workers=4
    )

    DEFAULT_CFG.save_dir = f"./runs/DroneVehicle"

    model_s = OBBTrainer(overrides=args)
    model_s.train()
