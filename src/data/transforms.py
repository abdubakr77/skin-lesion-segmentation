# Augmentation and preprocessing pipelines (train/val/test transforms).

import cv2
import os
import numpy as np

def read_image_and_label(filename_no_ext, data_yaml):
    img_path = os.path.join(data_yaml['train'], filename_no_ext + ".jpg")
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    labels_path = data_yaml['train'].replace('images', 'labels')
    label_path = os.path.join(labels_path, filename_no_ext + ".txt")

    polygons = []
    class_labels = []

    with open(label_path, 'r') as f:
        for line in f.readlines():
            values = line.split()
            cls_id = int(float(values[0]))
            coords = list(map(float, values[1:]))
            points = np.array(coords).reshape(-1, 2)

            polygons.append(points)
            class_labels.append(cls_id)

    return image, polygons, class_labels