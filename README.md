# Tomato Plant Disease Detection using CNN & Image Processing

## Overview
This project is an AI-driven web application designed to detect diseases in tomato plant leaves using **Convolutional Neural Networks (CNN)**. By simply uploading an image of a leaf, users can instantly receive predictions of the plant’s health condition along with confidence percentages.

This tool is particularly valuable for:
- 🌾 Farmers
- 🔬 Researchers
- 🌱 Agricultural students
- 🤖 AI/ML enthusiasts

The goal is to provide a **fast, accessible, and affordable solution** for early disease detection to help improve crop yield and reduce economic losses.

---

## Features
- ✅ Real-time disease prediction from leaf images
- ✅ Confidence score for predictions
- ✅ Identify both **healthy** and **diseased** leaves
- ✅ Simple and intuitive Streamlit user interface
- ✅ Powered by a custom-trained deep learning model (CNN)

---

## Model Information
- **Architecture**: Custom Convolutional Neural Network (CNN)
- **Framework**: TensorFlow / Keras
- **Input Image Size**: 128x128 pixels
- **Accuracy**: ~96% (on validation dataset)
- **Dataset**: Tomato leaf disease dataset (from PlantVillage & augmented offline)

### Diseases Detected:
| Category | Disease Name |
|----------|--------------|
| Fungal   | Early Blight, Late Blight, Leaf Mold, Target Spot, Septoria Leaf Spot |
| Bacterial| Bacterial Spot |
| Viral    | Mosaic Virus, Yellow Leaf Curl Virus |
| Pest     | Spider Mites |
| Healthy  | Tomato Healthy Leaf |

---

## Tech Stack
| Component       | Technology |
|----------------|------------|
| Programming    | Python     |
| Deep Learning  | TensorFlow, Keras |
| Web Framework  | Streamlit |
| Image Processing | Pillow, OpenCV |
| Deployment     | Streamlit Cloud / Local |

---

## Project Structure
```bash
Tomato-Disease-Detection/
│
├── main.py                          # Streamlit web app
├── trained_plant_disease_model.keras  # Trained CNN model
├── home_page.jpeg                  # Home page banner image
├── requirements.txt                # Dependencies
└── README.md                       # Documentation file
