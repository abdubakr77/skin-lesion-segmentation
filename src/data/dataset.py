# Dataset class that loads paired (image, mask) samples for training/validation/test.

from sklearn.preprocessing import MultiLabelBinarizer
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
import cv2
import os
import pickle
import numpy as np
from src.utils.polygon_validation import _is_valid_polygon
from tqdm import tqdm

def masks_to_polygons_dataset(masks_path, df, y_target, stage=1,epsilon_threshold=0.001):

    all_classes_id = []
    polygons_list = []

    df = df.copy()

    polygons_length = []

    df.drop(['age','sex','localization'],inplace=True,axis=1)

    cls2idx = {x:y for y,x in enumerate(sorted(df[y_target].unique()))}

    for idx in range(len(df)):
        
        if stage == 1:
            cls_id = int(1 if df[y_target].iloc[idx] == 'mel' else 0)
        elif stage == 2:
            cls_id = int(cls2idx[df[y_target].iloc[idx]])

        mask = cv2.imread(os.path.join(masks_path, df['image_id'].iloc[idx] + '_segmentation.png'),cv2.IMREAD_GRAYSCALE)

        img_h, img_w = mask.shape[:2]

        contours, _ = cv2.findContours(mask.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        classes = []
        image_polygons = []

        for contour in contours:
            contour = _is_valid_polygon(contour,img_h,img_w,df['image_id'].iloc[idx])
            if isinstance(contour,str) and contour == 'continue':
                continue

            epsilon = epsilon_threshold * cv2.arcLength(contour, True)
            simplified = cv2.approxPolyDP(contour, epsilon, True)

            contour_points = simplified.reshape(-1, 2)  # shape (N, 2)
            
            normalized = [coord for x, y in contour_points for coord in (x / img_w, y / img_h)]

            polygons_length.append(len(normalized))
            image_polygons.append(normalized)
            classes.append(cls_id)

        all_classes_id.append(classes)
        polygons_list.append(image_polygons)
    
    # print(len(polygons_list))

    df['class_id'] = all_classes_id
    df['polygons'] = polygons_list

    print('='*35)

    print(f'Max Value: {np.max(polygons_length)}')
    print(f'Min Value: {np.min(polygons_length)}')
    print(f'Mean: {np.mean(polygons_length)}')

    empty_count = sum(1 for p in polygons_list if len(p) == 0)
    print(f'Empty polygons: {empty_count} / {len(df)}')

    return pickle.loads(pickle.dumps(df))



def split_dataset(df,y_target,test_size=0.2,apply_leakage_check=False):
    grouped = (
        df.groupby("lesion_id")[y_target]
        .apply(list)
        .reset_index()
    )

    mlb = MultiLabelBinarizer()

    Y = mlb.fit_transform(grouped[y_target])

    msss = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=42
    )

    train_idx, temp_idx = next(msss.split(grouped["lesion_id"], Y))

    train_images = grouped.iloc[train_idx]
    temp_images = grouped.iloc[temp_idx]


    Y_temp = Y[temp_idx]

    msss2 = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=0.5,
        random_state=42
    )

    val_idx, test_idx = next(
        msss2.split(temp_images["lesion_id"], Y_temp)
    )

    val_images = temp_images.iloc[val_idx]
    test_images = temp_images.iloc[test_idx]


    train_df = df[df["lesion_id"].isin(train_images["lesion_id"])].reset_index(drop=True)

    val_df = df[df["lesion_id"].isin(val_images["lesion_id"])].reset_index(drop=True)

    test_df = df[df["lesion_id"].isin(test_images["lesion_id"])].reset_index(drop=True)

    if apply_leakage_check:
        print()

        print("Train Patient ID :", train_images.shape[0])
        print("Val Patient ID   :", val_images.shape[0])
        print("Test Patient ID  :", test_images.shape[0])

        print()

        print("Train Images :", len(train_df))
        print("Val Images   :", len(val_df))
        print("Test Images  :", len(test_df))

        print()

        train_files = set(train_df["lesion_id"])
        val_files = set(val_df["lesion_id"])
        test_files = set(test_df["lesion_id"])

        assert train_files.isdisjoint(val_files)
        assert train_files.isdisjoint(test_files)
        assert val_files.isdisjoint(test_files)

        print("Perfect! No Data Leakage Found.")

    return train_df, val_df , test_df


def clear_dataset_images(data_yaml, target='augmented', confirm_prompt=True):
    """
    Deletes images (and matching label files, if applicable) from the dataset.

    Parameters:
        data_yaml: dict with 'train' key pointing to the images path
        target: 'augmented' -> only files with 'aug' in the name
                'original'  -> only files WITHOUT 'aug' in the name
                'all'       -> everything
        confirm_prompt: if True, asks for y/n confirmation before deleting
    """

    main_images_path = data_yaml['train']

    def matches_target(fname):
        if target == 'augmented':
            return 'aug' in fname
        elif target == 'original':
            return 'aug' not in fname
        elif target == 'all':
            return True
        else:
            raise ValueError(f"Invalid target: {target}. Use 'augmented', 'original', or 'all'.")

    # ---- collect files to delete ----
    files_to_delete = []

    all_files_no_ext = [item.split('.')[0] for item in os.listdir(main_images_path)]
    files_to_delete = [
        os.path.join(main_images_path, f + '.jpg')
        for f in all_files_no_ext if matches_target(f)
    ]

    n_files = len(files_to_delete)

    if n_files == 0:
        print(f"No '{target}' images found. Nothing to delete.")
        return

    print(f"Found {n_files} images matching target='{target}'.")

    if confirm_prompt:
        confirm = input(f"Delete all {n_files} images? - (y or n): ").lower().strip()
        if confirm != 'y':
            print("Cancelled. No files deleted.")
            return

    # ---- delete ----
    deleted_count = 0
    failed_count = 0

    for fpath in tqdm(files_to_delete, desc=f'Deleting {target} images...'):
        try:
            os.remove(fpath)

            
            label_path = fpath.replace('images', 'labels').replace('.jpg', '.txt')
            if os.path.exists(label_path):
                os.remove(label_path)

            deleted_count += 1
        except Exception as e:
            print(f"Failed to remove {fpath}: {e}")
            failed_count += 1

    print(f"Deleted {deleted_count} images. Failed: {failed_count}.")
    if failed_count > 0:
        print("Check the failed deletions above and re-run if needed.")