## Online CLOD (Continual Learning Object Detection)

### Environment
```bash
conda create -n clod python==3.10 -y
conda activate clod
pip install ultralytics
```
### Create pretraining model
1. Download coco/VOC dataset
2. Preprocess coco/VOC dataset 
```bash
python preprocess_coco40.py # split for 40+40 cl
python preprocess_voc10.py # split for 10+10 cl
```
```bash
python main_pretrain.py
```
