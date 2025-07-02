# 🎥 Human-Action-Video-Classification

This project implements a two-stage deep learning pipeline that combines human action recognition with video style transfer.

---

## 🧠 Stage 1: Video Classification with ResNet

Classify human actions in videos using transfer learning with ResNet.

### 🔹 Dataset
- **UCF101**: A benchmark dataset of 13,000+ video clips across 101 human action categories (e.g., basketball dunk, archery, jumping jack).

### 🔹 Preprocessing
- Extracted and resized frames from video clips.
- Prepared frame sequences suitable for CNN input.

### 🔹 Model
- Utilized pre-trained **ResNet50** as the feature extractor.
- Appended a custom classification head for multi-class prediction.

### 🔹 Training
- Fine-tuned on a subset of UCF101 using standard data augmentation and learning rate scheduling.

---

## 📊 Model Evaluation

- **Validation Accuracy**: ~96%

---

## 🎨 Stage 2: Style Transfer with CycleGAN

Stylize video frames while preserving their semantic content (pose/action).

### 🔹 Objective
- Transfer video appearance to another visual domain (e.g., natural to artistic) **without requiring paired data**.

### 🔹 Architecture
- Implemented **CycleGAN** with ResNet-based generators and cycle-consistency loss.

### 🔹 Result
- Maintains motion and pose accuracy while transforming frame appearance into a target artistic style.

---

## ✨ Features

- ✅ Frame-by-frame transformation using pre-trained **CycleGAN** weights.
- ✅ Option to run **real-time video stylization** via `streamlit_cyclegan_video.py`.
- ✅ Supports applying models to individual frames or full video sequences.

---

## 📁 Project Structure

- `video_classification/` — ResNet training, evaluation, and preprocessing
- `style_transfer/` — CycleGAN scripts for training and stylization
- `notebooks/` — Visualizations and experiments
- `streamlit_cyclegan_video.py` — Real-time stylization web app
- `data/` — Sample input frames and class labels
