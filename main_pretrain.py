from ultralytics import YOLO
from ultralytics import RTDETR

# Load a model

# YOLOv8
# model = YOLO("yolov8l.pt")
# print(model.ckpt['train_args'])
# train_args = model.ckpt['train_args']
# model = YOLO('yolov8l.yaml')

model = YOLO("yolov8s.pt")
print(model.ckpt['train_args'])
train_args = model.ckpt['train_args']
model = YOLO('yolov8s.yaml')

# RTDETR
# model = RTDETR("rtdetr-l.pt")
# print(model.ckpt['train_args'])
# breakpoint()
# train_args = model.ckpt['train_args']
# model = RTDETR('rtdetr-l.yaml')

# YOLO11
# model = YOLO("yolo11l.pt")
# print(model.ckpt['train_args'])
# train_args = model.ckpt['train_args']
# model = YOLO('yolo11l.yaml')

# breakpoint()
train_args['data'] = './configuration/datasets/VOC.yaml'
train_args['device']= "6"
# Train the model
train_results = model.train(**train_args)

