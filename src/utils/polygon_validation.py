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
        return polygon