# Helper functions to visualize images, masks, and predictions.

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

def show_image(
    df,
    images_path,
    masks_path=None,
    target=None,
    draw_mask=False,
    draw_box=False,
    show_raw_only=False,
    mask_opacity=0.3
):
    import matplotlib

    # Select image
    if target and len(target) == 2:
        rows = df[df[target[0]] == target[1]]
        if len(rows) == 0:
            raise ValueError(f"No data found for {target[0]} = {target[1]}")
        row = rows.iloc[np.random.randint(len(rows))]
        image_id = row['image_id']

    elif target is None:
        row = df.iloc[np.random.randint(len(df))]
        image_id = row['image_id']

    else:
        raise IndexError(f"Index out of range! Must be only 2. Got {len(target)}")

    data = df[df['image_id'] == image_id]

    # Read raw image
    image = cv2.imread(os.path.join(images_path, f"{image_id}.jpg"))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_h, img_w = image.shape[:2]

    disease_colors = {
        'bkl': 'orange',
        'nv': 'green',
        'df': 'purple',
        'mel': 'red',
        'vasc': 'cyan',
        'bcc': 'blue',
        'akiec': 'yellow'
    }

    if show_raw_only:
        mask_path = os.path.join(masks_path, f"{image_id}_segmentation.png")
        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        fig, axes = plt.subplots(1, 2, figsize=(15, 15))
        axes[0].imshow(image)
        axes[0].set_title(f"{image_id} | Raw Image")
        axes[0].axis("off")

        axes[1].imshow(mask_img, cmap='gray')
        axes[1].set_title(f"{image_id} | Raw Mask")
        axes[1].axis("off")

        plt.show()
        return

    fig, ax = plt.subplots(figsize=(15, 15))

    for _, row in data.iterrows():
        dx = row['dx']
        color = disease_colors.get(dx, 'gray')
        label = dx

        # polygons are normalized [0-1], scale to image size
        points = np.array(row['polygons']).reshape(-1, 2)
        points[:, 0] *= img_w
        points[:, 1] *= img_h
        points = points.astype(np.int32)

        xmin = points[:, 0].min()
        ymin = points[:, 1].min()
        xmax = points[:, 0].max()
        ymax = points[:, 1].max()

        if draw_mask:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [points], 255)

            rgb = np.array(matplotlib.colors.to_rgb(color)) * 255
            image[mask == 255] = (
                image[mask == 255] * (1 - mask_opacity) + rgb * mask_opacity
            ).astype(np.uint8)

        if draw_box:
            rect = patches.Rectangle(
                (xmin, ymin), xmax - xmin, ymax - ymin,
                linewidth=2, edgecolor=color, facecolor='none'
            )
            ax.add_patch(rect)

        if draw_mask or draw_box:
            ax.text(
                xmin, max(ymin - 5, 10), label,
                color=color, fontsize=9,
                bbox=dict(facecolor='black', alpha=0.5, pad=0.5)
            )

    ax.imshow(image)

    legend_elements = [
        patches.Patch(facecolor='none', edgecolor=color, label=disease)
        for disease, color in disease_colors.items()
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    ax.set_title(f"{image_id} | dx_type: {row['dx_type']} | Images for this ID: {len(data)}")

    ax.axis("off")
    plt.show()
