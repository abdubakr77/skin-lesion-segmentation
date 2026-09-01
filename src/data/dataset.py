# Dataset class that loads paired (image, mask) samples for training/validation/test.

from sklearn.preprocessing import MultiLabelBinarizer
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
import cv2
import os
import pickle
from shapely.geometry import Polygon
import numpy as np


def masks_to_polygons_dataset(masks_path, df, y_target, stage=1):

    all_classes_id = []
    polygons_list = []

    df = df.copy()

    polygons_length = []

    df.drop(['age','sex','localization'],inplace=True,axis=1)

    cls2idx = {x:y for y,x in enumerate(df[y_target].unique())}

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
            if len(contour) < 3:
                print(f"WARNING: CONTOUR LENTGH IS {len(contour)} in {df['image_id'].iloc[idx]}, WILL BE SKIPPED") 
                continue

            contour = contour.squeeze()

            if contour.ndim != 2:
                print(f"N_DIM WARNING: CONTOUR DIMENSIONS ({contour.ndim}) IS NOT 2, WILL BE SKIPPED")
                continue

            polygon = Polygon(contour)

            if not polygon.is_valid:
                print(
                    f"WARNING: THIS POLYGON IS NOT VALID "
                    f"IN THIS ROW INDEX: {idx}, TRYING TO FIX..."
                )
                polygon = polygon.buffer(0)

                if polygon.is_empty:
                    print("Failed To Fix...")
                    continue

                if polygon.geom_type != "Polygon":
                    print(f"Failed To Fix... Result is {polygon.geom_type}")
                    continue

                print("Fixed Successfully!")

            contour = list(polygon.exterior.coords)
            
            normalized = [coord for x, y in contour for coord in (x / img_w, y / img_h)]

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

    return pickle.loads(pickle.dumps(df))

