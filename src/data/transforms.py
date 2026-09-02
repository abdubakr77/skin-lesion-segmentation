# Augmentation and preprocessing pipelines (train/val/test transforms).

import cv2
import os
import numpy as np
import albumentations as A
from src.utils.visualization import visualize_augmentation

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


def augment_and_save(image, polygons, class_labels, n_copies, base_filename,
                      output_images, output_labels, aug_config, debugging=False):

    img_h, img_w = image.shape[:2]
    transform = build_transform(aug_config)

    
    points_per_polygon = [len(p) for p in polygons]
    all_points = [(x * img_w, y * img_h) for p in polygons for x, y in p]

    all_images, all_polygons, all_labels, all_filenames = [], [], [], []

    for n in range(n_copies):
        augmented = transform(image=image, keypoints=all_points)
        new_img = augmented['image']
        new_points = augmented['keypoints']
        new_h, new_w = new_img.shape[:2]

        new_polygons = []
        i = 0
        for count in points_per_polygon:
            pts = new_points[i:i + count]
            pts = [(np.clip(x / new_w, 0, 1), np.clip(y / new_h, 0, 1)) for x, y in pts]
            new_polygons.append(pts)
            i += count

        all_images.append(new_img)
        all_polygons.append(new_polygons)
        all_labels.append(class_labels)
        all_filenames.append(f"{base_filename}_aug{n}")

    if debugging:
        visualize_augmentation(all_images, all_polygons, all_labels, titles=all_filenames)
        return

    for new_img, new_polygons, new_labels, new_filename in zip(all_images, all_polygons, all_labels, all_filenames):
        cv2.imwrite(os.path.join(output_images, f"{new_filename}.jpg"),
                    cv2.cvtColor(new_img, cv2.COLOR_RGB2BGR))

        with open(os.path.join(output_labels, f"{new_filename}.txt"), 'w') as f:
            for cls_id, polygon in zip(new_labels, new_polygons):
                coords_str = '  '.join(f"{x} {y}" for x, y in polygon)
                f.write(f"{cls_id}  {coords_str}\n")