# 1D Conv Layer Approach for Blood Glucose Estimation Using PPG Signals

## Overview
This project, developed by Vaibhav Aggarwal and Khushi under the supervision of Dr. Mohit Sajwan at Netaji Subhas University of Technology, Delhi, implements a hybrid deep learning model combining 1D Convolutional Neural Networks (CNNs) and Long Short-Term Memory (LSTM) networks for non-invasive blood glucose level (BGL) estimation using Photoplethysmography (PPG) signals. The project leverages the VitalDB dataset and a smaller PPG dataset for model training and evaluation.

## Project Details
- **Title**: 1D Conv Layer Approach to Estimate Blood Glucose Level Using Photoplethysmography Signals (PPG)
- **Authors**: Vaibhav Aggarwal (2021UIT3043), Khushi (2021UIT3046)
- **Supervisor**: Dr. Mohit Sajwan
- **Institution**: Netaji Subhas University of Technology, Delhi
- **Date**: May 2025
- **Degree**: B.Tech. in Information Technology

## Repository Structure
- **Notebooks**:
  - `method_1.ipynb`: Manual feature engineering-based machine learning approach using Random Forest, Extremely Randomized Trees, and AdaBoost.
  - `method_2.ipynb`: Hybrid 1D CNN-LSTM model with self-attention for BGL estimation.
  - `helper.ipynb`: Helper functions for data preprocessing, feature extraction, and evaluation.
  - `EMD_and_baselineDriftRemoval.ipynb`: Empirical Mode Decomposition (EMD) and baseline drift removal for PPG signal preprocessing.
- **Data Files**:
  - `df_cases.csv`: VitalDB patient metadata (age, sex, weight, height, surgical details).
  - `df_labs.csv`: VitalDB laboratory test results, including blood glucose levels.
  - `df_trks.csv`: VitalDB high-resolution physiological time-series data (PPG, ECG, etc.).
  - `ppg_ecg_trk_data.csv`: Raw PPG and ECG tracking data.
  - `ppg_ecg_preprocessedSignal.csv`: Preprocessed PPG and ECG signals.
  - `final_df2.csv`: Final dataset with extracted features and BGL labels.
- **Model**:
  - `lstm_bp_model.h5`: Pre-trained LSTM model for blood pressure estimation.

## Key Features
- **Datasets**:
  - VitalDB: High-resolution PPG signals from 6,000+ surgical patients (`https://api.vitaldb.net/`).
  - Smaller PPG dataset: 657 subjects, 7,884 segments from Guilin People's Hospital.
- **Preprocessing**:
  - Noise removal using Butterworth filters and Discrete Wavelet Transform (DWT).
  - Segmentation into 16-minute, 10-second, and 1-second windows.
  - Z-score and Min-Max normalization.
- **Models**:
  - **Method 1**: Manual feature engineering (63 features) with tree-based models.
  - **Method 2**: Hybrid 1D CNN-LSTM with ResNet-50 inspired architecture and BiLSTM.
- **Evaluation Metrics**:
  - Huber Loss, MAE, MSE, RMSE, MAPE, Clarke Error Grid Zone A accuracy.
- **Results**:
  - Hybrid 1D Conv-LSTM (1s windows): RMSE 27.07 mg/dL, MAE 28.12 mg/dL.
  - LSTM (10s segments): RMSE 47.13 mg/dL, MAE 42.45 mg/dL.

## Requirements
- **Python**: 3.x
- **Libraries**:
  - `numpy`, `scipy`, `pandas` (data processing)
  - `tensorflow`/`keras` (model implementation)
  - `pywt` (DWT)
  - `emd` (EMD)
- **Dataset Access**: VitalDB API (`https://api.vitaldb.net/`)

## Usage
1. **Setup**:
   - Install libraries: `pip install numpy scipy pandas tensorflow pywt emd`
   - Download VitalDB data or use provided CSVs.
2. **Preprocessing**:
   - Run `EMD_and_baselineDriftRemoval.ipynb` for EMD and noise removal.
   - Use `helper.ipynb` for segmentation, feature extraction, and normalization.
   - Generate `ppg_ecg_preprocessedSignal.csv` and `final_df2.csv`.
3. **Training**:
   - Run `method_1.ipynb` for tree-based models.
   - Run `method_2.ipynb` for hybrid 1D CNN-LSTM (50 epochs, batch size 64, Adam optimizer, Huber loss).
4. **Evaluation**:
   - Use `helper.ipynb` for metrics and visualizations.
   - Compare performance using tables and figures.
5. **Deployment**:
   - Adapt `lstm_bp_model.h5` for transfer learning or use the hybrid model for low-latency inference.

## Results
- **Hybrid 1D Conv-LSTM (1s windows)**:
  - Huber Loss: 29.87
  - MAE: 28.12 mg/dL
  - MSE: 732.54 mg/dL
  - RMSE: 27.07 mg/dL
- **LSTM (10s segments)**:
  - Huber Loss: 45.23
  - MAE: 42.45 mg/dL
  - MSE: 2210.76 mg/dL
  - RMSE: 47.13 mg/dL
- **Note**: The hybrid model achieves >90% Clarke Error Grid Zone A accuracy.

## Future Work
- Improve dataset diversity for better generalizability.
- Optimize for embedded hardware.
- Validate through clinical trials.

## References
- Zeynali et al., "Non-invasive blood glucose monitoring using PPG signals with various deep learning models and implementation using TinyML," *Scientific Reports*, 2025.
- VitalDB: https://www.nature.com/articles/s41597-022-01411-5
- Full references in `BTP_Vaibhav_Khushi_8th_sem.pdf`.

## Acknowledgments
- Dr. Mohit Sajwan for guidance.
- Netaji Subhas University of Technology for resources.
- VitalDB team for dataset access.

For details, refer to `BTP_Vaibhav_Khushi_8th_sem.pdf` or contact the authors.
