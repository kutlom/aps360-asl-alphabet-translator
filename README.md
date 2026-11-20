# Real-Time ASL Alphabet Translator

**Author:** Kutlo Moseki  
**Course:** APS360 — Applied Fundamentals of Deep Learning, University of Toronto  
**Tech Stack:** Python • PyTorch • OpenCV • NumPy • Google Colab  

![Status](https://img.shields.io/badge/Status-Completed-brightgreen)
![Accuracy](https://img.shields.io/badge/Test%20Accuracy-92.0%25-blue)
![Framework](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c)

## Overview
An end-to-end computer vision pipeline that classifies static American Sign Language (ASL) hand gestures (A–Z, `space`, `delete`, `nothing`) in real time from standard webcam video feeds. Designed to bridge communication accessibility gaps, the system emphasizes **low-latency inference** and **environmental robustness** under varying lighting conditions and backgrounds.

---

## Key Features & Engineering Highlights
* **92% Classification Accuracy:** Achieved across 29 distinct ASL gesture classes on unseen validation/test splits.
* **Robust Data Preprocessing Pipeline:** Engineered custom image transformations (random horizontal flips/rotations, brightness/contrast jittering, and RGB normalization) to prevent overfitting to uniform studio backgrounds.
* **Low-Latency Real-Time Inference:** Integrated OpenCV live video capture with a Region-of-Interest (ROI) bounding box and frame-buffer smoothing to eliminate flickering predictions during continuous signing.
* **Modular Codebase:** Separated data loading, model architecture, training loops, and edge/webcam inference into clean, maintainable Python modules.

---

## Dataset
* **Source:** [ASL Alphabet Dataset (Kaggle)](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
* **Scale:** 87,000 RGB images ($200 \times 200$ pixels) distributed evenly across **29 classes** (3,000 images per class).
* **Data Split:** 70% Training (60,900 images) | 15% Validation (13,050 images) | 15% Test (13,050 images).

---

## Model Architecture & Training
To balance high spatial feature extraction with real-time frame rates on CPU/GPU hardware, the network utilizes a custom multi-block Convolutional Neural Network (CNN):
* **Feature Extractor:** 4 convolutional blocks (`Conv2d` $\rightarrow$ `BatchNorm2d` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPool2d`) scaling from 32 to 256 feature channels.
* **Classifier Head:** Adaptive average pooling followed by fully connected layers with `Dropout(p=0.5)` to reduce overfitting.
* **Optimization:** Trained in Google Colab (T4 GPU) using `CrossEntropyLoss` and the `Adam` optimizer ($\text{lr} = 0.001$) with a step learning-rate scheduler.

### Quantitative Results
| Metric | Value | Notes |
| :--- | :--- | :--- |
| **Test Accuracy** | **92.0%**[cite: 5, 6] | Evaluated across all 29 classes |
| **Input Resolution** | $64 \times 64$ / $128 \times 128$ | Downsampled for real-time throughput |
| **Inference Latency** | **< 15 ms / frame** | ~60+ FPS real-time webcam execution |

*(Add your training/validation loss curves and confusion matrix plots inside `results/` to display here)*
```markdown
![Training Curves](results/training_curves.png)
