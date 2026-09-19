
# 😷 Face Mask Detection using Custom CNN & Transfer Learning

<p align="center">
  A complete deep learning study of binary face-mask classification using a custom CNN,
  controlled experimentation, pretrained CNN comparison, and ResNet50 transfer learning.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow"/>
  <img src="https://img.shields.io/badge/Keras-Deep%20Learning-D00000?style=for-the-badge&logo=keras&logoColor=white" alt="Keras"/>
  <img src="https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV"/>
  <img src="https://img.shields.io/badge/Scikit--Learn-Evaluation-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-Learn"/>
</p>

---

## 📌 Project Overview

This project develops a binary image-classification system capable of distinguishing between:

- 😷 **Mask**
- 🙂 **No Mask**

The project was designed as more than a single model-training exercise.

Instead, it follows an experimental workflow to understand how different choices affect CNN-based image classification:

```text
Dataset Construction
        ↓
Image Preprocessing
        ↓
Custom CNN Baseline
        ↓
Augmentation Experiment
        ↓
Regularization Experiment
        ↓
Combined Experiment
        ↓
Pretrained Model Screening
        ↓
Preprocessing Correction
        ↓
ResNet50 Selection
        ↓
Transfer Learning
        ↓
Controlled E5–E8 Experiments
        ↓
Final Evaluation
        ↓
External Image Testing
```

The project therefore studies both:

1. Model performance  
2. The effect of different preprocessing and training strategies

### 🎯 Objectives

The main objectives of this project are:

- Build a face-mask classifier from scratch
- Establish a custom CNN baseline
- Study the effect of data augmentation
- Study the effect of regularization
- Compare augmentation and regularization independently
- Study their combined effect
- Evaluate multiple pretrained CNN architectures
- Identify the impact of architecture-specific preprocessing
- Select one pretrained architecture for detailed experimentation
- Build a transfer-learning pipeline
- Evaluate the models using multiple metrics
- Analyze errors using confusion matrices
- Compare training and validation behavior
- Test the trained classifier using previously unseen external images
- Document the complete evolution of the project

---

## 🧠 Problem Definition

This is a binary image classification problem.

The classifier receives an image and produces a probability representing the likelihood that the image belongs to the positive **Mask** class.

```text
Input Image
     ↓
Image Preprocessing
     ↓
CNN
     ↓
Sigmoid Probability
     ↓
Threshold = 0.5
     ↓
┌───────────────┐
│               │
│  No Mask      │  Mask
│   0           │   1
│               │
└───────────────┘
```

**Final output:**

- `0` → No Mask  
- `1` → Mask

### 🏷️ Class Definition

#### 😷 Mask — Label 1

The positive class contains different types of face coverings, including:

- Medical masks
- Surgical masks
- N95 masks
- Cloth masks
- Respirator masks
- Other face-covering masks

All mask types are intentionally treated as one class.  
The objective is to determine whether a face is wearing a mask, **not** to identify the exact mask type.

#### 🙂 No Mask — Label 0

The negative class contains:

- Uncovered faces
- Faces without masks
- Different face poses
- Different orientations
- Images with hands near the face
- Other non-mask facial images

---

## 📊 Dataset

The final dataset was created by merging multiple publicly available datasets.

### Dataset Sources

| Dataset               | Source            | Main Contribution |
|-----------------------|-------------------|-------------------|
| Face Mask 12K         | Kaggle            | Mask + No Mask    |
| Mendeley Mask Dataset | Mendeley Data     | Mask              |
| Human Faces           | Kaggle            | No Mask           |
| HandoverFace9 RGB     | VLM HandoverFace9 | No Mask           |

The purpose of combining the sources was to create a larger and more varied binary classification dataset.

### 📈 Dataset Statistics

The final dataset used during experimentation contained approximately:

```text
35,710 images
```

#### Class Distribution

| Class      | Label | Images |
|------------|-------|--------|
| 😷 Mask    | 1     | 18,197 |
| 🙂 No Mask | 0     | 17,513 |
| **Total**  |       | **35,710** |

The classes are relatively balanced, with no class dominating the dataset.

### 🧪 Test Set

The held-out test set contained:

| Class   | Images |
|---------|--------|
| No Mask | 1,752  |
| Mask    | 1,819  |
| **Total** | **3,571** |

The test set was kept separate from training during model evaluation.  
This allowed the experiments to report performance on data that was not used for gradient updates.

---

## 🔄 Dataset Processing

The project went through several preprocessing stages.  
The preprocessing strategy was **not fixed** from the beginning.  
It evolved as the experiments exposed problems and opportunities for improvement.

### 🧹 Preprocessing Evolution

#### Stage 1 — Face Detection with MediaPipe / BlazeFace

The original approach used a face-detection stage before classification.

```text
Original Image
      ↓
MediaPipe / BlazeFace
      ↓
Face Detection
      ↓
Largest Face Selection
      ↓
Face Crop
      ↓
224 × 224
      ↓
CNN
```

The preprocessing experiment used a BlazeFace TFLite detector through the MediaPipe-based pipeline.

The process included:

```text
Image
 ↓
Temporary 2× resize for detection
 ↓
Face detection
 ↓
Select largest detected face
 ↓
Minimum face-size filtering
 ↓
20% crop margin
 ↓
Crop from original image
 ↓
RGB conversion
 ↓
224 × 224 resize
```

The purpose was to provide the classifier with a face-focused input.

#### ❌ Why MediaPipe Was Removed

MediaPipe was eventually removed from the final classification pipeline.  
This was an intentional project change.

**Problems with the face-detection dependency:**

- Added an additional preprocessing stage
- Introduced another model into the pipeline
- Face-detection failures could prevent classification
- Increased implementation complexity
- Added another dependency that needed to remain consistent during inference
- Made the pipeline harder to reproduce
- Required the detector and classifier to work correctly together

The project therefore moved toward a simpler **direct-image classification** pipeline.

#### ✅ Final Preprocessing Pipeline

The final classifier does **not** depend on MediaPipe face detection.

```text
Original Image
      ↓
Read Image
      ↓
RGB Conversion
      ↓
Resize → 224 × 224
      ↓
Model-Specific Preprocessing
      ↓
CNN
      ↓
Mask / No Mask
```

- For the **custom CNN**, the input pipeline uses normalized image values.
- For **pretrained architectures**, the preprocessing is aligned with the specific pretrained network.
- For **ResNet50**:

```python
tf.keras.applications.resnet50.preprocess_input()
```

### ⚠️ Major Preprocessing Discovery

One of the most important discoveries happened during pretrained-model evaluation.

Initially, the pretrained models were evaluated using an input preprocessing approach that was **not properly aligned** with the expectations of the pretrained architectures.

The resulting model performance was unexpectedly poor.  
Some models appeared to perform close to random classification.

This initially suggested that something was wrong with the models.  
However, the actual issue was the **input preprocessing**.

### 🔧 Preprocessing Correction

The pretrained-model pipelines were corrected so that each architecture received input in the format expected by its pretrained weights.

After the correction, the same model families produced dramatically better results:

```text
Before correction
        ↓
Poor pretrained-model performance
        ↓
Preprocessing investigation
        ↓
Architecture-specific preprocessing
        ↓
Re-evaluation
        ↓
~99.75% – 99.89% accuracy
```

**Key lesson:**  
A pretrained model is not just the architecture.  
**Its expected input preprocessing is part of the pretrained pipeline.**

---

## 🧠 Custom CNN

Before using transfer learning, a CNN was developed from scratch.  
The purpose was to establish an independent baseline.  
The custom CNN did **not** use ImageNet pretrained features.

A simplified representation is:

```text
224 × 224 × 3
       ↓
Conv2D + ReLU
       ↓
MaxPooling
       ↓
Conv2D + ReLU
       ↓
MaxPooling
       ↓
Conv2D + ReLU
       ↓
MaxPooling
       ↓
Flatten
       ↓
Dense
       ↓
Dropout
       ↓
Sigmoid
       ↓
Mask / No Mask
```

The baseline was then modified through controlled experiments.

### 🧪 Custom CNN Experiments — E1 to E4

| Experiment | Configuration                              |
|------------|--------------------------------------------|
| E1         | Custom CNN                                 |
| E2         | Custom CNN + Augmentation                  |
| E3         | Custom CNN + Regularization                |
| E4         | Custom CNN + Augmentation + Regularization |

The purpose was to isolate the effect of augmentation and regularization.

### 📊 E1–E4 Results

| Experiment | Configuration                       | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|------------|-------------------------------------|----------|-----------|--------|----------|---------|
| E1         | Custom CNN                          | 97.76%   | 99.77%    | 95.82% | 97.76%   | 99.89%  |
| E2         | CNN + Augmentation                  | 99.24%   | 98.75%    | 99.78% | 99.26%   | 99.98%  |
| E3         | CNN + Regularization                | 99.41%   | 99.83%    | 99.01% | 99.42%   | 99.97%  |
| E4         | CNN + Augmentation + Regularization | 99.02%   | 99.78%    | 98.30% | 99.03%   | 99.94%  |

#### 🔬 E1 — Custom CNN Baseline

```text
Accuracy  : 97.76%
Precision : 99.77%
Recall    : 95.82%
F1-Score  : 97.76%
ROC-AUC   : 99.89%
```

**Confusion matrix:**

```text
                 Predicted
              No Mask   Mask

Actual
No Mask         1748      4
Mask              76   1743
```

**Key observation:** Relatively lower recall (76 False Negatives).

#### 🔬 E2 — CNN + Augmentation

Data augmentation was introduced **only during training**.

```text
Training      → Augmentation ✓
Validation    → Augmentation ✗
Testing       → Augmentation ✗
```

```text
Accuracy  : 99.24%
Precision : 98.75%
Recall    : 99.78%
F1-Score  : 99.26%
ROC-AUC   : 99.98%
```

**Confusion matrix:**

```text
                 Predicted
              No Mask   Mask

Actual
No Mask         1729      23
Mask               4    1815
```

**Observation:**

- Accuracy: 97.76% → 99.24%
- Recall: 95.82% → 99.78%
- Precision decreased slightly (99.77% → 98.75%)

#### 🔬 E3 — CNN + Regularization

```text
Accuracy  : 99.41%
Precision : 99.83%
Recall    : 99.01%
F1-Score  : 99.42%
ROC-AUC   : 99.97%
```

**Confusion matrix:**

```text
                 Predicted
              No Mask   Mask

Actual
No Mask         1749      3
Mask              18    1801
```

Total errors: **21** out of 3,571 test images.

**E3 produced the strongest overall custom-CNN configuration.**

#### 🔬 E4 — CNN + Augmentation + Regularization

```text
Accuracy  : 99.02%
Precision : 99.78%
Recall    : 98.30%
F1-Score  : 99.03%
ROC-AUC   : 99.94%
```

**Confusion matrix:**

```text
                 Predicted
              No Mask   Mask

Actual
No Mask         1748      4
Mask              31    1788
```

**Observation:** Combining both techniques did **not** improve over E3.

```text
E3 Accuracy = 99.41%
E4 Accuracy = 99.02%
```

**Important conclusion:**  
Adding more techniques does not automatically produce a better model.

### 🧠 What the Custom CNN Experiments Taught

1. The custom CNN was already highly capable (**97.76%** without transfer learning).
2. Augmentation had a particularly strong effect on **recall**.
3. Regularization produced the strongest recorded custom-CNN configuration.
4. The combined configuration did not automatically improve performance.

---

## 🚀 Transfer Learning

After the custom CNN experiments, pretrained CNN architectures were evaluated.

**Candidate models:**

- ResNet50
- MobileNetV2
- EfficientNetB0
- DenseNet121
- InceptionV3

The models were first screened using the same evaluation protocol.  
After comparison, **one architecture** was selected for the detailed transfer-learning experiments.

### 🧪 Pretrained Model Screening

After correcting the preprocessing pipeline:

| Model          | Accuracy | Precision | Recall  | F1-Score | ROC-AUC |
|----------------|----------|-----------|---------|----------|---------|
| ResNet50       | 99.89%   | 99.84%    | 99.95%  | 99.89%   | 100.00% |
| MobileNetV2    | 99.83%   | 99.89%    | 99.78%  | 99.83%   | 100.00% |
| EfficientNetB0 | 99.75%   | 99.94%    | 99.56%  | 99.75%   | 100.00% |
| DenseNet121    | 99.83%   | 99.89%    | 99.78%  | 99.83%   | 100.00% |
| InceptionV3    | 99.89%   | 99.78%    | 100.00% | 99.89%   | 100.00% |

### 🧠 Why ResNet50 Was Selected

ResNet50 was carried forward as the single architecture for the detailed transfer-learning experiments based on:

- Accuracy
- Precision
- Recall
- F1-score
- Overall consistency
- Desire to keep the detailed experiment focused on one architecture

### 🏗️ ResNet50 Transfer-Learning Architecture

```text
Input Image
     ↓
ResNet50 (ImageNet Pretrained Weights)
     ↓
Global Average Pooling
     ↓
Dense(128, ReLU)
     ↓
Dropout(0.3)
     ↓
Dense(1, Sigmoid)
     ↓
Mask / No Mask
```

The initial transfer-learning stage freezes the pretrained backbone.  
Only the newly added classification layers are trained.

### 🧪 Transfer Learning Experimental Design (E5–E8)

| Experiment | Model    | Augmentation | Regularization | Purpose                    |
|------------|----------|--------------|----------------|----------------------------|
| E5         | ResNet50 | ❌            | ❌              | Transfer-learning baseline |
| E6         | ResNet50 | ✅            | ❌              | Effect of augmentation     |
| E7         | ResNet50 | ❌            | ✅              | Effect of regularization   |
| E8         | ResNet50 | ✅            | ✅              | Combined configuration     |

This creates a consistent 2×2 experimental structure:

```text
                 Regularization
                  No       Yes

Augmentation No   E5       E7
Augmentation Yes  E6       E8
```

### ⚙️ Training Configuration

| Parameter          | Value                |
|--------------------|----------------------|
| Input Size         | 224 × 224 × 3        |
| Batch Size         | 32                   |
| Optimizer          | Adam                 |
| Loss               | Binary Cross-Entropy |
| Output             | Sigmoid              |
| Epochs             | 5                    |
| Pretrained Model   | ResNet50             |
| Pretrained Weights | ImageNet             |
| Backbone           | Initially Frozen     |

Five epochs were intentionally used for the standardized experiments.  
Early stopping was **not** used in the final standardized experiment configuration.

---

## 📏 Evaluation Metrics

The models are evaluated using multiple metrics rather than accuracy alone:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix

### Confusion Matrix Structure

```text
                         Predicted
                    No Mask       Mask

Actual No Mask         TN           FP
Actual Mask            FN           TP
```

- **Positive Class** = Mask  
- **Negative Class** = No Mask

---

## 🧪 External Image Testing

After completing the benchmark evaluation, the model can be tested using images uploaded directly from a local computer.

```text
User Uploads Image
        ↓
Read Image
        ↓
RGB Conversion
        ↓
Resize → 224 × 224
        ↓
ResNet50 Preprocessing
        ↓
Trained Model
        ↓
Prediction Probability
        ↓
Mask / No Mask
```

These images are **not** used during model training.

### 🌍 Why External Testing Matters

External images can introduce:

- Different lighting
- Different backgrounds
- Different camera quality
- Different face angles
- Different mask types
- Different image resolutions
- Different compositions
- Different visual distributions

External testing acts as a practical sanity check, but should **not** be interpreted as a replacement for a large independent external benchmark.

---

## ⚠️ Interpreting the Very High Accuracy

The final benchmark results are extremely high.  
However, high accuracy should be interpreted carefully.

**Potential factors include:**

- Dataset Similarity
- Near-Duplicate Images
- Source Bias
- Distribution Shift
- Limited External Generalization

**Correct interpretation:**  
The model demonstrates extremely strong performance on the assembled benchmark dataset, while broader generalization still requires careful validation.

---

## 🔍 Important Experimental Observations

1. The custom CNN was already strong (**97.76%** baseline).
2. Augmentation improved recall dramatically (95.82% → 99.78%).
3. Regularization produced the strongest custom-CNN configuration (E3).
4. Combining techniques did **not** automatically improve performance.
5. Preprocessing had a major effect on pretrained models.
6. All five pretrained models performed very closely after preprocessing correction.
7. ResNet50 was selected for detailed transfer-learning experiments.
8. Accuracy alone is not enough — multiple metrics are necessary.

---

## 🔄 Complete Project Evolution

1. **Dataset Expansion** — Combined multiple public sources → 35,710 images  
2. **MediaPipe / BlazeFace** — Face-detection stage introduced  
3. **MediaPipe Removal** — Simplified to direct image classification  
4. **Custom CNN Experiments** — E1 → E2 → E3 → E4  
5. **Pretrained Model Screening** — 5 architectures compared  
6. **Preprocessing Correction** — Dramatic performance improvement  
7. **Single Architecture Selection** — ResNet50 chosen  
8. **Training Standardization** — 5 epochs, no early stopping  

---

## 🛠️ Technologies Used

- **Programming:** Python  
- **Deep Learning:** TensorFlow, Keras  
- **Computer Vision:** OpenCV, MediaPipe / BlazeFace (earlier stage)  
- **Data Processing:** NumPy, Pandas  
- **Visualization:** Matplotlib  
- **Evaluation:** Scikit-learn  
- **Training Environment:** Kaggle Notebook + NVIDIA Tesla T4 GPU  

---

## 📦 Main Model Families

```text
Custom CNN
     ↓
ResNet50
MobileNetV2
EfficientNetB0
DenseNet121
InceptionV3
```

Final detailed transfer-learning experiments focus on **ResNet50**.

---

## 🔬 Reproducibility Notes

To reproduce the reported experiments as closely as possible:

1. Use the same dataset version.
2. Keep the train, validation, and test partitions fixed.
3. Keep random seeds fixed where possible.
4. Do **not** augment validation data.
5. Do **not** augment test data.
6. Use architecture-specific preprocessing for pretrained models.
7. Keep test images completely separate from training.
8. Record the exact experiment configuration.
9. Record training time.
10. Save model weights and notebook outputs.
11. Keep external test images separate from the benchmark dataset.
12. Use the same image size of **224 × 224**.
13. Use the same binary label mapping:  
    `0 = No Mask` `1 = Mask`

---

## ⚠️ Limitations

- **Dataset Dependence** — Performance can change with input distribution.
- **Dataset Source Bias** — Models may learn source-specific characteristics.
- **Near-Duplicate Risk** — Possible inflation of benchmark performance.
- **External Generalization** — A few external images are not a rigorous external evaluation.
- **Binary Classification Limitation** — Only answers “Is a mask present?”  
  Does **not** classify:
  - Correctly worn mask
  - Mask below nose / chin
  - Improper placement
  - Partial coverage
  - Mask type

---

## 🚧 Possible Future Improvements

- Stronger duplicate and near-duplicate detection
- Grouped dataset splitting
- Independent external benchmark datasets
- More challenging real-world images
- Fine-tuning deeper ResNet50 layers
- Learning-rate scheduling
- Threshold optimization
- Calibration analysis
- Grad-CAM visual explanations
- Error-category analysis
- Mask-position classification
- Multi-class mask detection
- Deployment as a web application
- Real-time webcam inference

---

## 🧠 Final Takeaways

1. Start with a baseline.
2. Change one major component at a time.
3. Augmentation and regularization solve different problems.
4. More complexity does not guarantee better results.
5. Preprocessing is part of the model.
6. Compare before committing.
7. Accuracy is not the whole story.
8. Very high benchmark performance requires careful interpretation.
9. External testing is useful as a sanity check.
10. The project is an **experimental study**, not just a classifier.

---

## 🏁 Final Project Summary

This project developed a complete binary face-mask classification pipeline using both a custom CNN and transfer learning.

**Custom CNN best result (E3):**

```text
Accuracy : 99.41%
Precision: 99.83%
Recall   : 99.01%
F1       : 99.42%
ROC-AUC  : 99.97%
```

**Pretrained screening (after preprocessing correction):**

```text
ResNet50       → 99.89%
MobileNetV2    → 99.83%
EfficientNetB0 → 99.75%
DenseNet121    → 99.83%
InceptionV3    → 99.89%
```

ResNet50 was selected for the detailed transfer-learning experiments (E5–E8).

The central lesson of the project is that model performance depends not only on the architecture, but also on the **dataset**, **preprocessing pipeline**, **training configuration**, **evaluation methodology**, and **experimental design**.

---

## 📚 Resources

**Dataset Sources**

- [Face Mask 12K](https://www.kaggle.com/datasets/ashishjangra27/face-mask-12k-images-dataset)
- [Mendeley Mask Dataset](https://data.mendeley.com/datasets/8pn3hg99t4/2)
- [Human Faces](https://www.kaggle.com/datasets/ashwingupta3012/human-faces)
- [HandoverFace9 RGB](https://sites.google.com/view/sghanem/vlm-handoverface)

---


```

