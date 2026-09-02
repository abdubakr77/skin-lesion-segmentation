import cv2
import os
from shapely.geometry import Polygon
import numpy as np
import matplotlib.pyplot as plt


def _is_valid_polygon(contour, img_h, img_w, image_id='Your Image'):

    min_area_ratio = 0.001

    if len(contour) < 3:
        print(f"WARNING: CONTOUR LENTGH IS {len(contour)} in {image_id}, WILL BE SKIPPED") 
        return 'continue'

    # skip tiny noise contours
    contour_area = cv2.contourArea(contour)
    if contour_area < (min_area_ratio * img_h * img_w):
        return 'continue'

    contour = contour.squeeze()

    if contour.ndim != 2:
        print(f"N_DIM WARNING: CONTOUR DIMENSIONS ({contour.ndim}) IS NOT 2, WILL BE SKIPPED")
        return 'continue'

    polygon = Polygon(contour)
    
    if not polygon.is_valid:
        print(
            f"WARNING: THIS POLYGON IS NOT VALID "
            f"IN THIS IMAGE ID: {image_id}, TRYING TO FIX..."
        )
        polygon = polygon.buffer(0)

        if polygon.is_empty:
            print("  - Failed To Fix...")
            return 'continue'

        if polygon.geom_type != "Polygon":
            if polygon.geom_type == "MultiPolygon":
                # take the largest sub-polygon instead of dropping it
                polygon = max(polygon.geoms, key=lambda p: p.area)
                print(f"Recovered largest part from MultiPolygon")
            else:
                print(f"  - Failed To Fix... Result is {polygon.geom_type}")
                return 'continue'

        print("  - Fixed Successfully!")

    # convert to a proper cv2-compatible array before simplifying
    contour = np.array(polygon.exterior.coords, dtype=np.float32).reshape(-1, 1, 2)

    return contour


def compare_polygon_to_mask(row, images_path, masks_path, area_threshold=0.05, plot=True):

    image_id = row['image_id']
    polygons = row['polygons']

    # skip rows with no valid polygon
    if not polygons or len(polygons) == 0:
        print(f"SKIPPED: {image_id} has empty polygons")
        return None

    image = cv2.imread(os.path.join(images_path, f"{image_id}.jpg"))
    img_h, img_w = image.shape[:2]

    mask_path = os.path.join(masks_path, f"{image_id}_segmentation.png")
    true_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    true_mask = (true_mask > 127).astype(np.uint8)

    poly_mask = np.zeros((img_h, img_w), dtype=np.uint8)

    # handle multiple contours per image (list of lists) vs a single flat list
    is_multi = isinstance(polygons[0], list)
    contours_list = polygons if is_multi else [polygons]

    for contour in contours_list:
        points = np.array(contour).reshape(-1, 2)
        points[:, 0] *= img_w
        points[:, 1] *= img_h
        points = points.astype(np.int32)
        cv2.fillPoly(poly_mask, [points], 1)

    true_area = true_mask.astype(np.int64).sum()
    poly_area = poly_mask.astype(np.int64).sum()
    area_diff_ratio = abs(true_area - poly_area) / true_area

    if area_diff_ratio > area_threshold:
        print(f"WARNING: {image_id} area diff = {area_diff_ratio:.2%}")

    if not plot:
        return area_diff_ratio

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(true_mask, cmap='gray')
    axes[0].set_title(f"{image_id} | True Mask")
    axes[0].axis("off")

    axes[1].imshow(poly_mask, cmap='gray')
    axes[1].set_title(f"{image_id} | Polygon Mask")
    axes[1].axis("off")

    overlay = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    overlay[..., 0] = true_mask * 255
    overlay[..., 1] = poly_mask * 255
    axes[2].imshow(overlay)
    axes[2].set_title(f"Overlay | Area Diff: {area_diff_ratio:.2%}")
    axes[2].axis("off")

    plt.show()

    return area_diff_ratio