# 🚗 AI-Based Accident Detection and Severity Classification System

## 📌 Overview

This project presents a **two-stage deep learning system** for:

1. **Accident Detection & Localization** using YOLOv8
2. **Accident Severity Classification** using a CNN (MobileNetV3 Small)

The system is designed for real-time smart traffic monitoring and emergency response support. It detects accident regions in images or video streams and classifies the detected accident into **Moderate** or **Severe** categories.

The project is implemented entirely in **PyTorch** and deployed using **Streamlit** for an interactive web-based demo.

---

# 🏗 System Architecture

### 🔹 Stage 1 – Object Detection

* Model: **YOLOv8s**
* Framework: Ultralytics YOLO
* Output: Bounding box localization of accident regions

### 🔹 Stage 2 – Severity Classification

* Model: **MobileNetV3 Small**
* Framework: PyTorch
* Output: Binary classification

  * Class 0 → Moderate
  * Class 1 → Severe

---

# 🧠 Tech Stack

## Programming Language

* Python

## Deep Learning Framework

* PyTorch (`torch`)
* Torchvision

## Object Detection Framework

* Ultralytics YOLOv8

## Pre-trained Models

* YOLOv8s (`yolov8s.pt`)
* MobileNetV3 Small (`IMAGENET1K_V1` weights)

## Computer Vision Libraries

* OpenCV (`cv2`)
* Pillow (PIL)

## Data Handling

* PyYAML
* OS module
* shutil

## Optimization

* AdamW Optimizer
* CrossEntropyLoss

## Deployment

* Streamlit (Web-based live demo)

## Hardware Acceleration

* Apple Silicon GPU (MPS backend)
* CUDA (if available)
* CPU fallback

---

# 📂 Project Structure

```
├── best.pt                         # Fine-tuned YOLOv8 model
├── cnn_severity_model.pth          # Trained CNN model
├── final.py                        # Streamlit deployment app
├── code.ipynb                      # Complete model training notebook
├── requirements.txt
├── README.md

├── dataset/
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── valid/
│   │   ├── images/
│   │   └── labels/
│   └── data.yaml
```

---

# 📊 Dataset Details

## Source

Dataset obtained from **Roboflow Universe**:

**Project Name:** accident-gtowx
License: Creative Commons Attribution 4.0
Format: YOLO format

The dataset was exported in YOLO format and used directly for training.

---

## 📦 YOLOv8 Dataset Statistics

| Split      | Images       | Annotations |
| ---------- | ------------ | ----------- |
| Train      | 12,115       | 12,115      |
| Validation | 2,982        | 2,982       |

**Total Images:** 15,097

Annotations follow standard YOLO format:

```
class_id x_center y_center width height
```

(Normalized coordinates)

---

## 🧠 CNN Dataset Statistics

The same dataset images were used for severity classification.

| Split      | Images      |
| ---------- | ----------- |
| Train      | 12,115      |
| Validation | 2,982       |

Total images used: 15,097

### Classes:

* Class 0 → Moderate
* Class 1 → Severe

---

# ⚙️ Training Details

## 🔹 YOLOv8 Training

* Base Model: yolov8s.pt
* Fine-tuned on accident dataset
* Optimized using Ultralytics training pipeline
* Output: `best.pt`

## 🔹 CNN Training

* Base Model: MobileNetV3 Small
* Pre-trained on ImageNet (IMAGENET1K_V1)
* Final fully connected layer modified for 2-class output
* Optimizer: AdamW
* Loss Function: CrossEntropyLoss
* Output: `cnn_severity_model.pth`

All training code is available in:

```
code.ipynb
```

---

# 🚀 Deployment

The trained models are integrated into a **Streamlit web application**.

### To Run the App:

```bash
pip install -r requirements.txt
streamlit run final.py
```

The app supports:

* Image upload
* Real-time accident detection
* Severity prediction display

---

# 🔍 How It Works

1. User uploads image or video frame.
2. YOLOv8 detects accident regions.
3. Detected region is cropped.
4. Cropped region is passed to CNN.
5. CNN predicts severity (Moderate / Severe).
6. Result displayed via Streamlit interface.

---

# 📈 Applications

* Smart City Traffic Monitoring
* Highway Surveillance Systems
* Emergency Response Automation
* AI-powered CCTV Monitoring
* Road Safety Research

---

# 🔮 Future Improvements

* Add independent test dataset evaluation
* Integrate SMS alert system
* Deploy on edge devices (Jetson Nano, Raspberry Pi)
* Integrate tracking (DeepSORT)
* Add multi-accident scene detection
* Expand to multi-class severity grading

---

# 📜 License

Dataset: Creative Commons Attribution 4.0
Code: MIT License (or your chosen license)

---
