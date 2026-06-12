# Multimodal Fusion Framework for Explainable Radiology Assessment

This repository contains a multimodal deep learning framework for radiology exam classification. The model combines image features, clinical text information, and anatomical metadata to predict whether a radiology study is normal or abnormal. The framework was evaluated on radiology data from multiple anatomical regions, including chest, spine, and extremity examinations.

## Model Overview

The proposed architecture consists of three input branches:

### Image Encoder

Radiology images are processed using an EfficientNet-B0 backbone to extract visual features.

### Text Encoder

Clinical text is encoded using a pre-trained Bio_ClinicalBERT model. The encoder is used to generate contextual representations from clinical descriptions and reports.

### Anatomical Metadata Encoder

Categorical anatomical information is transformed into dense feature representations through a fully connected projection network.

### Feature Fusion and Classification

Features from all three modalities are concatenated and passed through a classification network consisting of fully connected layers and dropout regularization. The final layer performs binary classification (Normal vs. Abnormal).

## Experimental Results

| Metric    | Score  |
| --------- | ------ |
| Accuracy  | 91.06% |
| Precision | 92.36% |
| Recall    | 89.52% |
| F1-Score  | 90.92% |
| ROC-AUC   | 0.9689 |

## Explainability Methods

To improve model interpretability, the following explanation techniques are included:

### Grad-CAM++

Generates visual attention maps highlighting image regions that contribute most to the prediction.

### LIME

Provides local explanations by estimating the contribution of textual features to individual predictions.

### SHAP

Measures feature importance across the multimodal feature space and quantifies modality contributions.

## Repository Structure

```text
├── src/
│   ├── model.py
│   └── dataset.py
├── train.py
├── predict.py
├── requirements.txt
└── README.md
```

### File Description

* `src/model.py` – Model architecture implementation
* `src/dataset.py` – Dataset loading and preprocessing
* `train.py` – Training pipeline
* `predict.py` – Evaluation and inference pipeline
* `requirements.txt` – Required Python packages

## Installation

```bash
pip install -r requirements.txt
```

## Training

```bash
python train.py
```

## Inference

```bash
python predict.py
```

## Notes

The repository is intended for research and educational purposes. Performance may vary depending on dataset composition, preprocessing procedures, and training configurations.
