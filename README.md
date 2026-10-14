# Unsupervised Product Image Quality Control

**Unsupervised anomaly detection for e-commerce product images using Vision Transformers and PatchCore-style feature matching.**

## Overview

Product images are central to e-commerce catalog quality. Blurry images, watermarks, missing-image placeholders, and images associated with the wrong product can degrade the shopping experience.

This project explores an unsupervised computer-vision pipeline that learns visual patterns from normal product images and identifies potential anomalies.

The approach combines patch-level anomaly detection using pretrained Vision Transformer (ViT) features with complementary image-quality signals and a global-embedding baseline.

The current implementation is an experimental research prototype evaluated using synthetically generated image defects.

## Objectives

- Develop an unsupervised pipeline for detecting anomalous product images.
- Extract local visual features using a pretrained Vision Transformer.
- Implement PatchCore-style nearest-neighbor anomaly detection.
- Use a coreset memory bank to represent normal image features efficiently.
- Combine patch-level anomaly scores with image-quality and global-embedding signals.
- Compare the proposed approach against patch-only detection and a global-embedding baseline.
- Generate anomaly heatmaps to visualize potentially problematic image regions.

## Defect Categories

The notebook evaluates four types of image defects.

| Defect type | Description |
|---|---|
| Blur | Gaussian blur applied to a normal product image. |
| Watermark | Synthetic watermark text overlaid on an image. |
| Placeholder | A synthetic missing-image placeholder replacing the original image. |
| Wrong product | An image sampled from a product category different from the target category. |

Clean images serve as the normal evaluation class.

**Note:** These defects are synthetically constructed. Their detection performance does not necessarily represent performance on naturally occurring catalog defects.

## Methodology

### 1. Dataset preparation

The project uses the [Amazon Berkeley Objects (ABO) dataset](https://amazon-berkeley-objects.s3.amazonaws.com/index.html), which provides product images and associated metadata.

The notebook downloads the small-image archive and product listing metadata, selects a target product category, and prepares normal and anomalous evaluation samples.

The current configuration uses:

- Up to 2,000 images from the target product category.
- Up to 600 images from other product categories for wrong-product sampling.
- A 60/40 split of target-category images into normal training and evaluation pools.
- Synthetic defects introduced into a subset of evaluation images.

Actual sample counts depend on data availability and successful image loading.

### 2. Synthetic defect generation

Controlled defects are introduced through image transformations or by substituting images from other product categories.

This produces image-level labels for evaluating anomaly detection under a known experimental setup.

### 3. Vision Transformer feature extraction

The pipeline uses a pretrained `vit_base_patch16_224` model through the `timm` library.

Intermediate-layer patch tokens are extracted from two transformer blocks. The features undergo:

- Classification-token removal.
- Reshaping into a 14 × 14 patch grid.
- Local 3 × 3 average pooling.
- L2 normalization.
- Feature concatenation across the selected layers.

This representation captures local visual patterns at multiple transformer depths.

### 4. PatchCore-style anomaly detection

A memory bank is constructed from patch features extracted from normal training images.

Greedy coreset subsampling reduces the memory-bank size while retaining representative feature vectors.

For each test image, the pipeline:

1. Extracts patch features.
2. Calculates nearest-neighbor distances to the normal memory bank.
3. Aggregates the five highest patch distances into an image-level anomaly score.
4. Reshapes patch distances into a 14 × 14 anomaly map.
5. Upsamples the map to 224 × 224 pixels for visualization.

Higher scores indicate greater deviation from the normal reference features.

The current configuration uses a coreset sampling ratio of 0.25 and top-five patch-distance aggregation.

### 5. Image-quality signals and score fusion

Patch-level anomaly detection may not capture every type of image defect equally well. The notebook explores complementary signals:

- **Patch anomaly score:** Mean of the highest patch-to-memory-bank distances.
- **Maximum patch anomaly score:** Largest patch-level distance.
- **Global embedding distance:** Distance from the test image's global representation to the normal training-image representations.
- **Sharpness signal:** Image sharpness estimated using Laplacian variance.

Additional exploratory signals include re-blur response, grayscale entropy, and high-frequency image content.

Selected signals are normalized using robust statistics derived from held-out normal calibration images. The fusion strategy uses the maximum of the selected non-negative standardized scores.

### 6. Threshold calibration and evaluation

The anomaly threshold is calibrated using held-out normal images, targeting a nominal false-positive rate of 10%.

The notebook also examines an operating point targeting 70% recall.

Evaluation metrics include:

- ROC-AUC.
- Precision, recall, and F1-score.
- Recall by defect category.
- False-positive rate on clean images.
- Precision-recall curves.
- False-positive counts at a common recall target.
- Anomaly heatmap visualizations.

The notebook compares three approaches:

1. **Proposed fusion:** PatchCore-style detection combined with selected image-quality and global-embedding signals.
2. **Patch-only detection:** PatchCore-style anomaly scoring without score fusion.
3. **Global-embedding baseline:** Distance between a test image's global embedding and the mean embedding of normal training images.

Performance claims should be added only after the notebook has been executed and the results verified.

## Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Deep learning | PyTorch |
| Vision Transformer models | `timm` |
| Image processing | OpenCV, Pillow |
| Data processing | NumPy, pandas |
| Evaluation | scikit-learn |
| Visualization | Matplotlib |
| Dataset | Amazon Berkeley Objects (ABO) |

## Repository Structure

The notebook is located directly in the repository root.

```text
unsupervised-product-image-qc/
├── README.md
├── unsupervised_product_image_qc_v2 (1).ipynb
├── requirements.txt
├── .gitignore
├── data/
│   └── README.md
├── outputs/
└── results/
    └── README.md
```

The Jupyter notebook contains the current end-to-end implementation. Separate Python modules, automated tests, and a standalone inference interface have not yet been extracted.

## Getting Started

### Prerequisites

- Python 3.10 or 3.11 recommended.
- Jupyter Notebook or JupyterLab.
- Internet access for dataset and pretrained-model downloads.
- A CUDA-enabled GPU recommended for faster feature extraction.

CPU execution may be possible but can be substantially slower.

### Installation

Clone or download the repository from GitHub and open a terminal in the project directory.

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Launch Jupyter:

```bash
jupyter notebook
```

Open `unsupervised_product_image_qc_v2 (1).ipynb` from the repository root and execute the cells in sequence.

The notebook downloads and extracts the dataset automatically. Keep the downloaded archives and extracted image data out of Git version control.

## Experimental Considerations

### Unsupervised training

The anomaly detector builds its memory bank from normal training images without using defect labels to fit the detector.

Synthetic labels are used for evaluation. The experiment therefore evaluates anomaly detection rather than a fully supervised defect-classification model.

### Data separation

The design separates normal training, calibration, and evaluation data. Threshold selection uses held-out normal calibration images rather than synthetic test labels.

Any selection of scoring signals or fusion settings informed by test-set performance must be documented and separated from final held-out evaluation.

### Interpretation of results

Synthetic defects provide a controlled initial evaluation. They do not establish equivalent performance on real-world catalog defects. Validation on independently labeled real-world data is required to assess practical effectiveness.

## Limitations

- **Synthetic evaluation:** Injected defects may not reflect the diversity of real-world image-quality problems.
- **Category dependence:** Performance may vary with the target category and its visual diversity.
- **Feature representation:** The pretrained ViT is not specifically trained to identify e-commerce image-quality defects.
- **Computational cost:** Patch-level nearest-neighbor comparisons can require substantial memory and processing.
- **Threshold sensitivity:** Results depend on the calibration distribution and operating threshold.
- **Limited defect taxonomy:** The experiment covers four constructed defect types.
- **Reproducibility:** Results may vary with package versions, hardware, random sampling, and dataset availability.
- **Production readiness:** The implementation remains notebook-based and is not a production-grade application.

## Future Work

- Evaluate performance on independently labeled real-world defective product images.
- Compare alternative vision encoders and anomaly-detection architectures.
- Conduct ablation studies to quantify the contribution of each scoring signal.
- Measure inference latency, memory consumption, and scalability.
- Pin dependency versions and document experiment configurations.
- Extract reusable Python modules and add automated tests.
- Develop a review interface showing anomaly scores and visual explanations.

## Project Status

**Research prototype — experimental implementation.**

This project investigates whether local visual anomaly detection, complemented by image-level quality signals, can support automated product image quality control. Its effectiveness requires reproducible evaluation and validation on representative real-world data.

## Acknowledgements

This project uses the Amazon Berkeley Objects dataset and pretrained Vision Transformer models available through the `timm` ecosystem.

Please consult the original dataset and model documentation for applicable citation, licensing, and usage requirements.
