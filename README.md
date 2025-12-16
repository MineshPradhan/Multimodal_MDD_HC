# Multimodal EEG and Speech Biomarker Extraction for MDD Detection

This repository presents an end-to-end multimodal machine learning framework for detecting **Major Depressive Disorder (MDD)** using **electroencephalography (EEG)** and **speech signals**. The project is built on the **MODMA (Multi-Modal Open Dataset for Mental Disorders Analysis)** dataset and integrates neurophysiological and behavioral biomarkers to improve diagnostic robustness and interpretability.

## 🔍 Key Features
- Audio feature extraction using Librosa and Praat-based voice analysis
- EEG feature extraction from resting-state and ERP signals using MNE-Python
- Subject-level multimodal feature fusion
- Supervised classification using Logistic Regression and Random Forest
- Model serialization using joblib
- Interactive Streamlit dashboard for visualization and prediction

## 🧠 Motivation
Traditional depression diagnosis relies heavily on subjective assessments. This project aims to provide an **objective, data-driven, and explainable approach** by leveraging multimodal biosignals associated with depressive disorders.

## 📂 Project Structure
- `notebooks/` – Feature extraction and model training notebooks
- `data_processed/` – Extracted and integrated feature datasets
- `Model_Output_Joblib/` – Trained models and preprocessing pipelines
- `Streamlit/` – Deployment dashboard and utilities

## 🚀 Applications
- Mental health research
- Clinical decision support systems
- Multimodal machine learning experimentation

## ⚠️ Disclaimer
This project is intended for research purposes only and is not a substitute for professional medical diagnosis.

-----------------------------------------------------------------------------------------------------------------

## Setup
Primary Data Downlaod Link : https://drive.google.com/file/d/1YP4zk7f0UGo6EiwFbtI6qS350J5fXaRT/view?usp=drive_link

## Streamlit Dashboard Link
https://multimodal-mddhc-app-cpahejyamwhcv6wsqjpsty.streamlit.app

## Folder Structure
Arrange the downloaded Dataset according to the given folder structure and do not tamper the files that are outside.

## 1_Audio Feature Extraction

### |______ audio_lanzhou

|__________ 02010001

|_______________ .wav

|__________ 02010002

## 2_EEG

### |_____ EEG_3channels_resting_lanzhou

|__________ 02010001_still.txt

|__________ 02010002_still.txt

### |_____ EEG_128channels_ERP_lanzhou

|__________ 02010002erp 20150416 1131.raw

|__________ 02010004erp 20141219 1602.raw

### |_____ EEG_128channels_resting_lanzhou

|__________ 02010002rest 20150416 1017.mat

|__________ 02010004rest 20150427 1335.mat

## 3_data_integration_and_classification
