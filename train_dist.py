import os

import warnings

warnings.filterwarnings("ignore", category=UserWarning)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import attempt_load_one_weight
from ultralytics.utils import DEFAULT_CFG
import torch

if __name__ == '__main__':

    # TEACHER
    _, model_t_rgb = attempt_load_one_weight(
        r'/home/zhaotianyi/M2D-LIF/M2D-LIF_open_source/checkpoint/monomodal/llvip_rgb.pt')  # the teacher model
    model_t_rgb["model"].info()
    _, model_t_ir = attempt_load_one_weight(
        r'/home/zhaotianyi/M2D-LIF/M2D-LIF_open_source/checkpoint/monomodal/llvip_ir.pt')  # the teacher model
    model_t_ir["model"].info()


    args = dict(
        model=r"/home/zhaotianyi/M2D-LIF/M2D-LIF_open_source/model_yaml/yolov8-LIF.yaml",
        data=r"/home/zhaotianyi/M2D-LIF/M2D-LIF_open_source/data/LLVIP.yaml",  # FLIR
        Distillation="MultiDistillation",
        distill_weight=0.8,
        Teacher_Model_RGB=model_t_rgb["model"],
        Teacher_Model_IR=model_t_ir["model"],
        loss_type="CWD",
        amp=False,
        imgsz=1280,
        epochs=100,
        batch=2,
        device=4,
        lr0=0.001,
        online=False,
        augment=True,
        workers=4
    )

    DEFAULT_CFG.save_dir = f"./runs/LLVIP"

    model_s = DetectionTrainer(overrides=args)
    model_s.train()
