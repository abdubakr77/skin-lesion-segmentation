import os
from tqdm import tqdm
import cv2
import numpy as np

def convert_to_classifier_crops(images_path, output_root, class_to_folder,
                                train_df=None, valid_df=None, test_df=None,
                                padding=0.15, remove_background=False):

    splits = {'train': train_df, 'val': valid_df, 'test': test_df}
    counts = {}

    for split_name, df in splits.items():
        if df is None:
            continue

        saved = 0

        for idx, row in tqdm(df.iterrows(), total=len(df),
                             desc=f'{split_name} is processing now...'):
            disease_name = row['dx']

            if disease_name not in class_to_folder:
                continue

            image_id = row['image_id']
            img = cv2.imread(os.path.join(images_path, image_id + '.jpg'))

            if img is None:
                print(f'WARNING: Could not read image: {image_id}')
                continue

            img_h, img_w = img.shape[:2]

            polygon = row['polygons'][0]
            points = list(zip(polygon[::2], polygon[1::2]))
            points = [(int(x * img_w), int(y * img_h)) for x, y in points]

            xs = [x for x, y in points]
            ys = [y for x, y in points]

            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)

            bbox_w = x_max - x_min
            bbox_h = y_max - y_min

            pad_x = bbox_w * padding
            pad_y = bbox_h * padding

            x1 = max(0, int(x_min - pad_x))
            y1 = max(0, int(y_min - pad_y))
            x2 = min(img_w, int(x_max + pad_x))
            y2 = min(img_h, int(y_max + pad_y))

            cropped_image = img[y1:y2, x1:x2]

            if cropped_image.size == 0:
                print(f'WARNING: Empty crop at index {idx}')
                continue

            if remove_background:
                mask = np.zeros(cropped_image.shape[:2], dtype=np.uint8)

                crop_points = np.array([
                    [x - x1, y - y1] for x, y in points
                ], dtype=np.int32)

                cv2.fillPoly(mask, [crop_points], 255)

                cropped_image[mask == 0] = 0

            output_dir = os.path.join(
                output_root,
                split_name,
                class_to_folder[disease_name]
            )
            os.makedirs(output_dir, exist_ok=True)

            output_img_name = f'{image_id}_{disease_name}.png'
            output_path = os.path.join(output_dir, output_img_name)

            if os.path.exists(output_path):
                raise FileExistsError(
                    f'File already exists at: {output_path}. '
                    f'Delete the old crops first if you want to re-run this.'
                )

            cv2.imwrite(output_path, cropped_image)
            saved += 1

        counts[split_name] = saved

    return counts