# yolo semantic model utils for lesion segmentation.

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib import patches

def get_background_class_id(names):
    for cls_id, cls_name in names.items():
        if cls_name.lower() == 'background':
            return cls_id
    raise ValueError("No class named 'background' found in model.names")

def predict(yolo_model, images_path, specific_image_name=None,
            save_dir: str = None, remove_background=False):

    if not os.path.exists(images_path):
        raise FileNotFoundError('Images Dir not existed! Please Check the path if is it correct')

    rand_image_name = np.random.choice(os.listdir(images_path)).split('.')[0]
    image_path = os.path.join(images_path, rand_image_name + '.jpg')

    if specific_image_name:
        rand_image_name = specific_image_name.split('.')[0]
        image_path = os.path.join(images_path, rand_image_name + '.jpg')

    original_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

    outputs = yolo_model.predict(original_image)
    output = outputs[0]
    names = output.names

    semantic_mask = output.semantic_mask
    if semantic_mask is None:
        print(f"No semantic mask returned for this image: {rand_image_name}.")
        return

    background_class_id = get_background_class_id(names)

    mask_data = semantic_mask.data.cpu().numpy()
    present_classes = [c for c in np.unique(mask_data) if c != background_class_id]

    print("Classes present in this image:", {int(c): names[int(c)] for c in present_classes})

    if len(present_classes) == 0:
        print(f"Only background detected in this image: {rand_image_name}.")
        return

    if save_dir:
        os.makedirs(os.path.join(save_dir, 'cropped_images'), exist_ok=True)

    colors = plt.cm.get_cmap('tab10', len(present_classes))

    _, ax = plt.subplots(1, 2, figsize=(18, 12))
    ax[0].set_title('Original Image')
    ax[0].imshow(original_image)
    ax[0].axis('off')

    ax[1].set_title('Segmented Classes: ' + ', '.join(names[c] for c in present_classes))
    ax[1].imshow(original_image)
    ax[1].axis('off')

    legend_patches = []

    for i, cls_id in enumerate(present_classes):
        binary_mask = (mask_data == cls_id).astype('uint8')
        ys, xs = np.where(binary_mask)
        x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()

        print(f"Class: {names[cls_id]} (id={cls_id}) | BBox: ({x1}, {y1}, {x2}, {y2}) | Pixels: {len(xs)}")

        color_mask = np.zeros((*binary_mask.shape, 4))
        color_mask[binary_mask == 1] = (*colors(i)[:3], 0.45)
        ax[1].imshow(color_mask)

        legend_patches.append(patches.Patch(color=colors(i), label=names[cls_id]))

        if save_dir:
            cropped_image = original_image[y1:y2 + 1, x1:x2 + 1]
            if remove_background:
                mask_3ch = np.stack([binary_mask] * 3, axis=-1)
                cropped_image = (original_image * mask_3ch)[y1:y2 + 1, x1:x2 + 1]

            cv2.imwrite(
                os.path.join(save_dir, 'cropped_images', f'{rand_image_name}_{names[cls_id]}.png'),
                cv2.cvtColor(cropped_image.astype('uint8'), cv2.COLOR_RGB2BGR)
            )

        if save_dir:
            cropped_image = original_image[y1:y2 + 1, x1:x2 + 1]
            if remove_background:
                mask_3ch = np.stack([binary_mask] * 3, axis=-1)
                cropped_image = (original_image * mask_3ch)[y1:y2 + 1, x1:x2 + 1]

            cv2.imwrite(
                os.path.join(save_dir, 'cropped_images', f'{rand_image_name}_{names[cls_id]}.png'),
                cv2.cvtColor(cropped_image.astype('uint8'), cv2.COLOR_RGB2BGR)
            )
            
    ax[1].legend(handles=legend_patches, loc='upper right')
