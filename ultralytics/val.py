from ultralytics import YOLO

# Load a model
# model = YOLO("yolov8n.yaml")  # build a new model from scratch
# model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
model = YOLO("/data/huangyao/1/codes/ultralytics-main/runs/detect/two_stream_rgbt/weights/best.pt")  # build a new model from scratch
# Use the model
# model.train(task='detect',data="DroneVehicle_rgb.yaml", epochs=100, imgsz=1024,device='6,7', batch=32, workers=2)  # train the model
metrics = model.val(task='detect',data="DroneVehicle_com.yaml", conf=0.001, batch=32, device="3,4", workers=2,save_json=True
                    ,save_dir= '/data/huangyao/1/codes/ultralytics-main/runs/val/debug' )  # evaluate model performance on the validation set
print(metrics.box.map50)  # map50
print(metrics.box.map75 ) # map75
print(metrics.box.map)  # map50-95
print(metrics.box.maps ) # a list contains map50-95 of each category