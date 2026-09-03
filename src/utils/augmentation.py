import cv2
import os
import numpy as np
import albumentations as A
from tqdm import tqdm
from src.utils.visualization import visualize_augmentation
from src.data.dataset import clear_dataset_images

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

def add_hair_noise(image, **kwargs):
    img = image.copy()
    h, w = img.shape[:2]
    n_hairs = np.random.randint(1, 4)

    for _ in range(n_hairs):
        x1, y1 = np.random.randint(0, w), np.random.randint(0, h)
        x2, y2 = x1 + np.random.randint(-100, 100), y1 + np.random.randint(-100, 100)
        color = tuple(np.random.randint(10, 60, 3).tolist())
        thickness = np.random.randint(1, 2)
        cv2.line(img, (x1, y1), (x2, y2), color, thickness)

    return img

def build_transform(config):
    return A.Compose([
        A.HorizontalFlip(p=config['hflip_p']),
        A.VerticalFlip(p=config['vflip_p']),
        A.Rotate(limit=config['rotate_limit'], p=config['rotate_p']),
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
        ),
        A.Lambda(image=add_hair_noise, p=config['hair_noise_p']),
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



def apply_smart_aug(data_yaml, aug_config, n_copies_per_class=None, apply_debug=False, clear_existing=False):

    main_images_path = data_yaml['train']
    labels_path = main_images_path.replace('images', 'labels')
    all_files_no_ext = [f.split('.')[0] for f in os.listdir(main_images_path)]

    aug_exists = any('aug' in f for f in all_files_no_ext)

    if aug_exists and not apply_debug:
        if clear_existing:
            clear_dataset_images(data_yaml, target='augmented', confirm_prompt=False)
        else:
            print("Warning: Found existing augmented images. Pass clear_existing=True to clear them, "
                  "otherwise new copies get added on top of the existing ones.")

    if apply_debug:
        rand_fname = np.random.choice(all_files_no_ext)
        image, polygons, class_labels = read_image_and_label(rand_fname, data_yaml)
        augment_and_save(image, polygons, class_labels, n_copies=3,
                          base_filename=rand_fname, output_images=None, output_labels=None,
                          aug_config=aug_config, debugging=True)
        return

    if n_copies_per_class is None:
        raise ValueError("n_copies_per_class is required. Use suggest_n_copies(data_yaml) to get a starting point.")

    for fname in tqdm(all_files_no_ext, desc='Augmenting Images Now...'):
        if 'aug' in fname:
            continue

        image, polygons, class_labels = read_image_and_label(fname, data_yaml)

        # decide n_copies for this image based on the rarest class it contains
        n = max((n_copies_per_class.get(cls_id, 0) for cls_id in class_labels), default=0)
        if n <= 0:
            continue

        augment_and_save(image, polygons, class_labels, n_copies=n,
                          base_filename=fname, output_images=main_images_path,
                          output_labels=labels_path, aug_config=aug_config)


def suggest_n_copies(data_yaml, class_names=None):
    """
    Counts how many label instances each class has across all label files,
    and suggests how many augmented copies per image would bring every class
    close to the largest class count.

    Args:
        data_yaml: dict with a 'train' key pointing to the images folder
        class_names: optional dict/list mapping class_id -> class name, for readable output

    Returns:
        counts: dict of class_id -> current instance count
        suggestions: dict of class_id -> suggested n_copies
    """
    labels_path = data_yaml['train'].replace('images', 'labels')
    all_files_no_ext = [f.split('.')[0] for f in os.listdir(data_yaml['train'])]

    counts = {}
    for fname in all_files_no_ext:
        label_path = os.path.join(labels_path, fname + '.txt')
        if not os.path.exists(label_path):
            continue
        with open(label_path, 'r') as f:
            for line in f.readlines():
                cls_id = int(float(line.split()[0]))
                counts[cls_id] = counts.get(cls_id, 0) + 1

    max_count = max(counts.values())

    suggestions = {}
    for cls_id, count in counts.items():
        suggestions[cls_id] = max(0, round(max_count / count) - 1) if count else 0

    if class_names:
        counts = {class_names[k]: v for k, v in counts.items()}
        suggestions = {class_names[k]: v for k, v in suggestions.items()}

    return counts, suggestions