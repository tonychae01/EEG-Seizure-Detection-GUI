# EEG-Seizure-Detection-GUI

---
## 📄 EEG-Seizure-Detection with Automated Graphic User Interface

[![View PDF Report](https://img.shields.io/badge/View%20Report-PDF-blue?style=flat-square&logo=adobe)](Kaggle%20Report.pdf)

> **Note:** You can also click the badge above or use the link below  
> [**Open the full PDF report**](Official Repo)
>

# Seizure Detection System

**Authors:** Tony Chae, Daniel Yoon, Yee Hong Pua, Caroline Hughes, Eric Goode, Arnav Surjan  
**Institution:** The University of Texas at Austin, Department of Electrical & Computer Engineering  
**Project Duration:** Fall 2024 (Data Science Lab Final Project)

---

## Overview
A machine learning–driven pipeline for binary seizure detection using EEG data from focal epilepsy patients. We evaluated multiple models (LSTM, CNN, XGBoost) on preprocessed 19‑channel recordings and developed a prototype GUI for real‑time alarm demonstration.  
**Key Achievements:**  
- Achieved up to **90.50% accuracy** and **0.96 AUC** (CNN).  
- Demonstrated real‑time feasibility via a Tkinter‑based GUI prototype.  

---

## Dataset
- **Source:** American University of Beirut Medical Center (Jan 2014–Jul 2015)  
- **Patients:** 6 focal epilepsy subjects undergoing presurgical evaluation  
- **Channels:** 21 scalp electrodes (10‑20 system), processed to 19 channels  
- **Sampling:** 500 Hz, band‑pass filtered (∼1.6 Hz–70 Hz) + 50 Hz notch filter  
- **Labels:**
  - `1` (Seizure: complex partial, electrographic, video‑detected)  
  - `0` (Normal)  
- **Size:** 7,790 one‑second epochs (19×500)  
  - Training: 90% (7,011 samples)  
  - Testing: 10% (779 samples)

---

## Preprocessing & Feature Prep
1. **Reshape & Labeling**  
   - Grouped all seizure types into class `1`, normal as `0`.  
   - Reshaped data to (samples, channels, time) → (7011, 19, 500).  
2. **Normalization & Filtering**  
   - Logarithmic scaling to spread amplitude distribution.  
   - Omitted noisy channels based on visual and statistical inspection.  
3. **Epoching & Raw Input**  
   - Prepared MNE RawArray for time‑frequency and correlation analyses.  

---

## Exploratory Analyses
- **Time-Series Plots:** Visualized 1 s (500 samples) per channel.  
- **PSD:** Computed power spectral density across 0–250 Hz (Nyquist).  
- **TFR:** Morlet wavelet transform (1–50 Hz, n_cycles=freq/2) for channel 0.  
- **Correlation Map:** Pairwise channel correlations via heatmap.

---

## Model Architectures & Training
| Model  | Key Characteristics                             | Test Accuracy | AUC  |
|-------:|:------------------------------------------------|--------------:|:----:|
| **LSTM**   | 1-layer LSTM (units=128), dropout=0.3, Adam | 89.98%        | 0.95 |
| **CNN**    | 3‑layer 1D CNN (filters [32,64,128]), ReLU, Adam | **90.50%**   | 0.96 |
| **XGBoost**| PCA→100 features, Optuna‑tuned boosting trees   | 90.24%        | 0.96 |

- **Loss:** Binary cross‑entropy (NNs), log loss (XGBoost).  
- **Validation:** 10‑fold cross‑validation, early stopping for deep models.  

---

## Results & Discussion
- **Best Performer:** CNN with 90.50% accuracy and 0.96 AUC.  
- **Trade‑offs:**  
  - LSTM excels in sequence learning but slightly lower accuracy.  
  - XGBoost offers computational efficiency and interpretability.  
- **Limitations:**  
  - Small subject pool (n=6), potential intra‑subject variability.  
  - Channel exclusion choices introduce manual bias.  
- **Future Directions:**  
  - Transfer learning for cross-subject generalization.  
  - Seizure prediction models (proictal detection).  

---

## Real‑Time GUI Prototype
- **Framework:** Python Tkinter  
- **Features:** Continuous signal plot, detection alarm, simulated hospital notification.  
- **Purpose:** Demonstrate feasibility of a portable, real‑time seizure monitoring device.

---

## Usage (Internal Prototype)
> **Note:** This code is proprietary and unpublished. Unauthorized copying or distribution is prohibited.  
1. Run `python gui_seizure_detection.py`.  
2. Ensure the test EEG stream is correctly formatted as 1 s windows (19×500).  
3. Adjust detection threshold and notification channels in `config.py`.

---

## License & Acknowledgments
© 2024 Tony Chae et al. All rights reserved.  
Thank you to Dr. Amy Zhang’s Data Science Lab for guidance and support.

