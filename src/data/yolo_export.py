
import os
from tqdm import tqdm
import shutil
from src.data.dataset import clear_dataset_images

def convert_to_yolo_segmentation(images_path: str, output_root: str,
                                  train_df, valid_df, test_df=None,
                                  clear_existing=True, target='all'):

    ds_partitions = {'train': train_df,
                      'val': valid_df,
                      'test': test_df}

    # ---- check + clear existing images/labels using clear_dataset_images ----
    if clear_existing:
        for split, df in ds_partitions.items():
            if df is None:
                continue

            split_images_path = os.path.join(output_root, split, 'images')
            if not os.path.exists(split_images_path) or len(os.listdir(split_images_path)) == 0:
                continue

            clear_dataset_images(data_yaml={'train': split_images_path}, target=target, confirm_prompt=True)

    # ---- make sure output folders exist ----
    for split, df in ds_partitions.items():
        if df is None:
            continue
        os.makedirs(os.path.join(output_root, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_root, split, 'labels'), exist_ok=True)

    # ---- Real Converting Here ----
    for split, df in ds_partitions.items():
        if df is None:
            continue

        for idx in tqdm(range(len(df)), desc=f'{split} Is Processing Now...'):

            image_id = df['image_id'].iloc[idx]
            polygons = df['polygons'].iloc[idx]
            class_ids = df['class_id'].iloc[idx]

            imgs_path = os.path.join(output_root, split, 'images', image_id + '.jpg')
            labels_path = os.path.join(output_root, split, 'labels', image_id + '.txt')

            if os.path.exists(imgs_path):
                raise FileExistsError(
                    f"Error: File already exists at {imgs_path}.\n"
                    f"Set clear_existing=True and re-run, or delete manually first."
                )

            with open(labels_path, 'w') as f:
                for cls_id, polygon in zip(class_ids, polygons):
                    coords_str = '  '.join(map(str, polygon))
                    f.write(f'{cls_id}  {coords_str}\n')

            shutil.copy2(os.path.join(images_path, image_id + '.jpg'), imgs_path)

        print(f"{split} Done!")