<div align="center">

# Skin Lesion Segmentation

A two stage deep learning pipeline that first segments and detects melanoma versus non melanoma lesions, then classifies the specific skin disease for non melanoma cases, combining semantic segmentation and image classification on dermatological images.

![Python](https://img.shields.io/badge/Python-3.13-blue)
![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO-purple)
![PyTorch](https://img.shields.io/badge/PyTorch-Swin%20Transformer-red)
![OpenCV](https://img.shields.io/badge/OpenCV-Image%20Processing-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

![Project Banner](assets/banner.png)

</div>

## Table of Contents

- [Overview](#overview)
- [Why Two Stages](#why-two-stages)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Pipeline](#pipeline)
- [Augmentation Strategy](#augmentation-strategy)
- [Evaluation](#evaluation)
- [Inference Pipeline](#inference-pipeline)
- [Installation](#installation)
- [Usage](#usage)
- [Notebooks](#notebooks)
- [Tech Stack](#tech-stack)
- [Sample Results](#sample-results)
- [Roadmap](#roadmap)
- [License](#license)
- [Author](#author)

## Overview

Melanoma is the deadliest form of skin cancer, and early detection from dermatological photographs can make a real difference in patient outcomes. This project builds a two stage pipeline that:

1. Segments the lesion boundary in a skin image using semantic segmentation.
2. Decides whether the lesion is melanoma or not.
3. If not melanoma, passes the same image to a dedicated image classifier that identifies which of several common skin conditions it actually is, while keeping the boundary already found in step 1.

The pipeline is built end to end: raw mask preprocessing, polygon extraction and validation, dataset balancing through smart augmentation, model training for both a segmentation model and a classifier, and a full inference pipeline with ground truth comparison and evaluation statistics.

## Why Two Stages

The dataset is heavily imbalanced. Melanoma cases are a small minority compared to non melanoma lesions, and within the non melanoma group the class distribution across diseases (nevus, benign keratosis, basal cell carcinoma, actinic keratosis, vascular lesions, dermatofibroma) is uneven as well. Training a single model on all classes at once would let the majority classes dominate learning and hurt melanoma detection specifically, which is the most clinically important class.

The two stages also solve genuinely different problems, so they use different model types on purpose:

- Stage 1 needs pixel level localization: where exactly is the lesion, and is it melanoma. This is a semantic segmentation task.
- Stage 2 only needs a whole image decision: given a lesion that is not melanoma, which specific disease is it. This is a plain image classification task, so a dedicated classifier is simpler and more accurate here than reusing a segmentation model.

Splitting the problem this way keeps each model focused on what it is best suited for, instead of forcing one architecture to do both jobs at once.

## Dataset

The dataset consists of dermoscopic images paired with segmentation masks and diagnosis labels, following a scheme similar to HAM10000 style skin lesion datasets.

- Raw data: original images, binary segmentation masks, and a metadata CSV with diagnosis (`dx`) and diagnosis confirmation method (`dx_type`).
- Stage 1 processed data: binary classes, melanoma vs not melanoma, exported as YOLO segmentation style label files (one or more polygons per lesion, with a class id) for training the segmentation model.
- Stage 2 processed data: organized as one folder per disease class (`01_AKIEC`, `02_BCC`, `03_BKL`, `04_DF`, `05_NV`, `06_VASC`), covering the six non melanoma diagnoses, for training the classifier in a standard image folder format.

## Project Structure

```
skin-lesion-segmentation/
├── data/
│   ├── raw/
│   │   ├── images/
│   │   ├── masks/
│   │   └── metadata.csv
│   └── processed/
│       ├── stage1/
│       │   └── {train,val,test}/{images,labels}
│       └── stage2/
│           └── {train,val,test}/
│               ├── 01_AKIEC/
│               ├── 02_BCC/
│               ├── 03_BKL/
│               ├── 04_DF/
│               ├── 05_NV/
│               └── 06_VASC/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_evaluation.ipynb
│   └── 05_inference_pipeline.ipynb
├── src/
│   ├── data/
│   │   ├── dataset.py
│   │   ├── transforms.py
│   │   └── yolo_export.py
│   ├── classifier/
│   │   ├── model_builder.py
│   │   ├── transforms.py
│   │   └── predict.py
│   ├── inference_utils/
│   │   ├── inference_io.py
│   │   ├── inference_visualization.py
│   │   └── inference_pipeline.py
│   └── utils/
│       ├── metrics.py
│       ├── polygon_preprocessing.py
│       ├── polygon_validation.py
│       ├── visualization.py
│       ├── augmentation.py
│       └── evaluation.py
├── outputs/
│   ├── checkpoints/
│   ├── predictions/
│   └── logs/
├── requirements.txt
├── .gitignore
└── README.md
```

## Pipeline

1. **Mask to polygon conversion.** Raw binary masks are converted into polygon annotations using contour detection, with a validation and repair step for invalid or self intersecting shapes, and a filter that removes tiny noise contours.
2. **Dataset splitting.** Data is split into train, validation, and test sets at the patient level (grouped by `lesion_id`) using multilabel stratified splitting, so the same lesion never leaks across splits.
3. **Export.** Stage 1 polygons and class ids are exported into YOLO segmentation label format. Stage 2 crops or copies images into a class per folder layout ready for a standard image classification dataset loader.
4. **Augmentation.** Underrepresented classes are boosted through targeted augmentation, described below.
5. **Training.**
   - Stage 1 is trained as a semantic segmentation model using Ultralytics, tracking cross entropy loss, Dice loss, mIoU, and pixel accuracy across epochs.
   - Stage 2 is a Swin V2 S image classifier built with a generic transfer learning setup: the pretrained backbone is frozen by default, only the replaced classification head trains at first, and specific late backbone blocks can be selectively unfrozen once the head has adapted, with clear per layer trainable parameter counts to guide that decision.
6. **Evaluation.** Training curves, confusion matrices, and metric comparisons between the two stages are reviewed in a dedicated notebook.
7. **Inference.** A cascading inference pipeline runs Stage 1, then Stage 2 when needed, compares predictions against ground truth when available, and organizes outputs for easy review.

## Augmentation Strategy

Beyond standard geometric and color augmentations (flips, rotation, brightness and contrast jitter, CLAHE, hue and saturation shifts), this project includes a custom augmentation designed specifically for dermoscopic images:

- **Hair overlay augmentation.** Real dermoscopic images often contain body hair crossing the lesion, which can confuse a model that has never seen it. Thin curved hair strands, extracted from reference images and simplified into smooth vector curves, are blended onto training images at random positions, angles, and scales using alpha blending rather than hard masking, so they look natural rather than like pasted graphics.
- **Class aware copy count.** Instead of augmenting every image the same number of times, the pipeline counts how many label instances exist per class and computes how many augmented copies each image needs so that rare classes get closer to the volume of the majority class.

## Evaluation

The evaluation notebook loads the training results for each stage and provides:

- Train versus validation loss curves for both stages.
- Stage 1 segmentation metrics: mIoU and pixel accuracy, with a bar chart comparison across training runs.
- Stage 2 classifier metrics: accuracy and confusion matrix across the six disease classes.
- Confusion matrices and sample prediction grids saved automatically during training.

## Inference Pipeline

The inference pipeline in `05_inference_pipeline.ipynb` runs both stages together as a cascade, all built around one core function:

- Accepts a single random image, one specific image, a list of specific images, or an entire folder in a single call.
- Stage 1 segments the lesion and decides melanoma or not melanoma. If it cannot find a lesion at all, the image is moved to a clearly labeled no detection folder instead of being forced through a low quality prediction.
- If Stage 1 predicts not melanoma, the same raw image (uncropped, no mask applied) is passed to the Stage 2 classifier. Stage 2 never produces its own mask, so Stage 1's mask is kept and simply relabeled with the specific disease Stage 2 identifies, colored from a fixed per disease palette.
- An optional confidence threshold routes low confidence Stage 2 predictions to their own folder instead of reporting an unreliable diagnosis.
- Ground truth is read from a single path that is auto detected as either raw segmentation masks or YOLO label files, used for the True Mask panel and for IoU and Dice scores.
- A separate metadata lookup (image id to true diagnosis) drives the correctness check for both stages, including the honest case where the true diagnosis is melanoma but Stage 1 missed it and Stage 2 still returns its best guess at the specific disease.
- Produces a four panel comparison per image: the predicted mask alone, the true mask alone, the image with the predicted mask overlaid, and the image with both predicted and true masks overlaid together, with a title colored green for a correct prediction and red for an incorrect one, plus an IoU and Dice comparison box.
- Saves cropped lesion images for further analysis through one shared cropping utility used by both stages.
- Aggregates everything into a results DataFrame and a statistics summary covering Stage 1 detection rate, Stage 1 and Stage 2 classification accuracy, mean IoU and Dice, and an end to end accuracy that reflects the pipeline's final answer as a whole.

## Installation

```bash
git clone https://github.com/abdubakr77/skin-lesion-segmentation.git
cd skin-lesion-segmentation
pip install -r requirements.txt
```

## Usage

Run the notebooks in order from the `notebooks/` folder:

```
01_data_exploration.ipynb      Explore class balance, image resolutions, and sample lesions
02_preprocessing.ipynb         Convert masks to polygons, split the dataset, export both stages
03_model_training.ipynb        Train the Stage 1 segmentation model and the Stage 2 classifier
04_evaluation.ipynb            Compare training metrics between stages
05_inference_pipeline.ipynb    Run the full two stage inference pipeline
```

## Notebooks

| Notebook | Purpose |
|---|---|
| `01_data_exploration.ipynb` | Dataset statistics and visual inspection |
| `02_preprocessing.ipynb` | Polygon extraction, validation, splitting, dataset export |
| `03_model_training.ipynb` | Training configuration and execution for both stages |
| `04_evaluation.ipynb` | Metric curves, confusion matrices, stage comparison |
| `05_inference_pipeline.ipynb` | End to end two stage inference with ground truth comparison |

## Tech Stack

- Python
- Ultralytics (semantic segmentation, Stage 1)
- PyTorch and Torchvision (Swin V2 S classifier, Stage 2)
- OpenCV
- Shapely
- Albumentations
- Pandas and NumPy
- Matplotlib
- Scikit learn and iterative stratification

## Sample Results

![Sample Prediction](assets/sample_prediction.png)

## Roadmap

- [ ] Expand Stage 2 to include additional rare disease classes as more labeled data becomes available.
- [ ] Experiment with instance level segmentation for overlapping lesions.
- [ ] Package the two stage pipeline behind a simple web interface for quick testing.

## License

This project is released under the MIT License. The dataset follows the license terms of its original source.

## Author

**Abdullah Bakr**
AI/ML Engineering Student

- GitHub: [@abdubakr77](https://github.com/abdubakr77)
- LinkedIn: [abdubakr](https://linkedin.com/in/abdubakr)
- Kaggle: [abdullahbakr7](https://kaggle.com/abdullahbakr7)
- Portfolio: [abdubakr77.github.io](https://abdubakr77.github.io)