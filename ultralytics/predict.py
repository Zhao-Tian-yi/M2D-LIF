from ultralytics import YOLO

model = YOLO("")
model.predict(source="./FLIR_datasets", save=True, save_conf=True, save_txt=True,
              name='output',save_json = True,imgsz=1024,device='0'
              , save_dir='')

# source后为要预测的图片数据集的的路径
# save=True为保存预测结果
# save_conf=True为保存坐标信息

# save_txt=True为保存txt结果，但是yolov8本身当图片中预测不到异物时，不产生txt文件
