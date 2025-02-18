import json
import os
import random
import numpy as np
import scipy.stats as stats
import glob

dataset = 'VOC'
output_dataset = 'VOC_10_10'
dataset_dir = f'datasets/{dataset}'
train_dir = os.path.join(dataset_dir, 'train')
val_dir = os.path.join(dataset_dir, 'test')

repeats = [1]
sigmas = [0.1]
seeds = [1, 2, 3]
init_cls = 1

clses = list(range(10, 20))
cls_list = {10: 'diningtable',
  11: 'dog',
  12: 'horse',
  13: 'motorbike',
  14: 'person',
  15: 'pottedplant',
  16: 'sheep',
  17: 'sofa',
  18: 'train',
  19: 'tvmonitor',
}

cls_datalist = {}
for id in cls_list.keys():
    cls_datalist[id] = []

labels_dir = os.path.join(dataset_dir, 'labels')
images_dir = os.path.join(dataset_dir, 'images')
label_files = glob.glob(labels_dir + "/train*/*.txt") + glob.glob(labels_dir + "/val*/*.txt")
breakpoint()
for cls_id in cls_list.keys():
    for label_file in label_files:
        split_name = label_file.split('/')[-2]
        base_name = label_file.split('/')[-1][:-4]
        # Read lines and filter
        with open(label_file, "r") as lf:
            lines = lf.readlines()
        valid_lines = []
        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                continue  # Skip empty lines if any

            # Split line and parse the first token as class ID
            parts = line_stripped.split()
            class_id = int(parts[0])
            
            if class_id == cls_id:
                valid_lines.append(line_stripped)
        if valid_lines:
            cls_datalist[cls_id].append(split_name + '/' + base_name)

for repeat in repeats:
    for sigma in sigmas:
        for seed in seeds:
            random.seed(seed)
            np.random.seed(seed)
            n_classes = len(clses)
            cls_increment_time = np.zeros(n_classes)
            samples_list = []
            for cls in clses:
                datalist = cls_datalist[cls]
                random.shuffle(datalist)
                samples_list.append(datalist)
            stream = []
            for i in range(n_classes):
                times = np.random.normal(i/n_classes, sigma, size=len(samples_list[i]))
                choice = np.random.choice(repeat, size=len(samples_list[i]))
                times += choice
                for ii, sample in enumerate(samples_list[i]):
                    if choice[ii] >= cls_increment_time[i]:
                        stream.append({'file_name': samples_list[i][ii], 'klass': cls_list[clses[i]], 'label':clses[i], 'time':times[ii]})
            random.shuffle(stream)
            stream = sorted(stream, key=lambda d: d['time'])
            data = {'cls_dict':cls_list, 'stream':stream, 'cls_addition':list(cls_increment_time)}

            with open(f'collections/{output_dataset}/{output_dataset}_sigma{int(sigma*100)}_repeat{repeat}_seed{seed}.json', 'w') as fp:
                json.dump(data, fp)
                
                
                
# val = []
# cls_list = os.listdir(train_dir)
# n_classes = len(cls_list)
# cls_dict = {cls:i for i, cls in enumerate(cls_list)}
# for i in range(n_classes):
#     cls_val_list = os.listdir(os.path.join(val_dir, cls_list[i]))
#     for ii, sample in enumerate(cls_val_list):
#         val.append({'file_name': os.path.join('val/', cls_val_list[ii]), 'klass': cls_list[i], 'label':i})

# with open(f'collections/{dataset}/{dataset}_val2.json', 'w') as fp:
#     json.dump(val, fp)