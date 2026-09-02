# Augmentation and preprocessing pipelines (train/val/test transforms).

import cv2
import os
import numpy as np
import albumentations as A

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


def build_transform(config):
    return A.Compose([
        A.HorizontalFlip(p=config['hflip_p']),
        A.VerticalFlip(p=config['vflip_p']),
        A.Rotate(limit=config['rotate_limit'], p=config['rotate_p'], border_mode=cv2.BORDER_REFLECT_101),
        A.RandomBrightnessContrast(
            brightness_limit=config['brightness_limit'],
            contrast_limit=config['contrast_limit'],
            p=config['brightness_contrast_p']
        ),
        A.HueSaturationValue(
            hue_shift_limit=config['hue_shift_limit'],
            sat_shift_limit=config['sat_shift_limit'],
            val_shift_limit=config['val_shift_limit'],
            p=config['hue_sat_val_p']
        ),
        A.CLAHE(clip_limit=config['clahe_clip_limit'], p=config['clahe_p']),
        A.ShiftScaleRotate(
            shift_limit=config['shift_limit'],
            scale_limit=config['scale_limit'],
            rotate_limit=0,
            p=config['shift_scale_rotate_p'],
            border_mode=cv2.BORDER_REFLECT_101
        ),
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))