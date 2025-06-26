from ultralytics import YOLO

# Load a model
model = YOLO(r"/home/zhangguiwei/KK/codes/maerceci/YOLO_MulDist_lby_1102/runs/cross_continue/weights/best.pt")  # build a new model from scratch
# model = YOLO("yolov8n_twostream.yaml")  # load a pretrained model (recommended for training)
# model = YOLO("/data/huangyao/1/codes/ultralytics-main/runs/detect/train9_rgb/weights/last.pt")  # build a new model from scratch
# Use the model
model.train(task='detect',data="FLIR_align.yaml", epochs=100, imgsz=1024,device='2', batch=32,
            workers=2,save_dir= '/data/huangyao/1/codes/ultralytics-main/runs/FLIR_detect/0618_tir' )  # train the model
metrics = model.val()  # evaluate model performance on the validation set
