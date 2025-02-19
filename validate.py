from ultralytics import YOLO
from ultralytics import RTDETR

# model = YOLO("./YOLOv8/yolov8s/weights/best.pt")
model = YOLO("./configuration/models/yolov8_nc10.yaml").load("./YOLOv8/yolov8s/weights/best.pt")
# model = YOLO("./YOLOv8/yolov8s3/weights/best.pt")

# print(model.ckpt['train_args'])

result = model.val(data='./configuration/datasets/VOC.yaml')
breakpoint()