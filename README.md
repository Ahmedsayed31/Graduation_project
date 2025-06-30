# 🧠 Kidney Disease Detection Project
## 📌 Overview
This project aims to build a deep learning pipeline for identifying and locating kidney diseases — specifically tumor and stone — in CT scan images. The project evolved through two main phases:

Image Classification using CNN to identify the class of a kidney image.

Object Detection using YOLO to locate the position of tumor or stone within the image.

## 🧪 Phase 1: Classification
🔹 Objective
Build a model to classify kidney CT scans into one of the following categories:

Tumor

Stone

Cyst

Normal

### 📁 Dataset
Initially gathered and organized from [Kaggle](https://www.kaggle.com/datasets/nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone/data) : .

Images were split into train, test, and valid folders.

A large number of high-quality CT scans were used per class.

### 🧪 Techniques
Used **Convolutional Neural Networks (CNNs)** , **VGG-!6** , **RestNet** , **MobilNet**.

Applied Image Augmentation using ImageDataGenerator to increase dataset size and reduce overfitting.

Achieved high accuracy **(~98%)** on test data.

📥 Transition to Object Detection
🔄 Why shift to detection?
While classification told us what type of disease was in the image, it didn't show where it was — a critical requirement for real-world medical analysis. Especially:

Tumors tend to be large and occupy clear regions.

Stones are small and require precise localization.

## 🎯 Phase 2: Object Detection with YOLO
📦 Tools Used
YOLOv8 from Ultralytics

Labeling with LabelMe

Data preparation: Manual creation of bounding boxes for Tumor and Stone images.

### 🧾 Annotation Process
Labeled only Tumor and Stone images.

Used LabelMe to draw bounding boxes.

Converted .json files to YOLO .txt format using a custom Python script.

### 🧠 Model Details
Model trained with YOLOv8.

Classes:

0: Tumor

1: Stone

Augmentation applied for both categories to balance data.

