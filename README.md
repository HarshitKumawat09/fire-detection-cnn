<div align="center">

# 🔥 Fire Detection System

**AI-Powered Real-Time Fire Detection in Videos**

*Detect fire instantly with deep learning — frame by frame, with confidence scores and visual alerts*

[![Python](https://img.shields.io/badge/Python-3.13+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20+-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-Educational-green?style=for-the-badge)](LICENSE)

[🎯 Live Demo](#-running-the-application) · [🧠 Model](#-model-details) · [🛠️ Install](#-installation) · [📖 Usage](#-usage-guide) · [📦 Dataset](https://www.kaggle.com/datasets/phylake1337/fire-dataset/data)

---

### ✅ Model Status: **Working & Accurate**

> Trained for **20 epochs** on **999 images** — achieving **~97–99% accuracy** on both fire and non-fire classification

</div>

<br>

## 🌟 Features

<table>
<tr>
<td width="50%">

📹 **Video Upload**
Drag & drop support for MP4, AVI, MOV, MKV

🤖 **AI-Powered Detection**
TensorFlow/Keras CNN with sigmoid output

⚡ **Frame-by-Frame Analysis**
Smart skip intervals for speed vs accuracy

</td>
<td width="50%">

📊 **Real-Time Progress**
Live progress bar & per-frame confidence

🎯 **Adjustable Threshold**
Fine-tune sensitivity from 0.1 to 0.9

📸 **Fire Snapshots**
Auto-captures the highest-confidence frame

</td>
</tr>
</table>

<br>

## 📸 Screenshots
 
### Upload Interface
![Fire Detection System - Upload Screen](screenshot_upload.png)
*Upload a video file and configure detection settings*
 
### Detection Results
![Fire Detection System - Results Screen](screenshot_result.png)
*Real-time fire detection snapshot with 99.8% confidence score*
 
<br>

## 🧠 Model Details

| Property | Value |
|---|---|
| **Architecture** | 4-layer CNN (32→64→128→128 filters) |
| **Input Shape** | `(224, 224, 3)` RGB |
| **Output** | Binary — `0 = No Fire`, `1 = Fire` |
| **Activation** | Sigmoid (probability output) |
| **Optimizer** | Adam |
| **Loss** | Binary Cross-Entropy |
| **Training Epochs** | 20 |
| **Dataset Size** | ~999 images (fire + non-fire) |
| **Validation Accuracy** | ~97–99% |
| **Label Fix** | `1 - prediction` flip (alphabetical TF labeling) |

### CNN Architecture

```
┌─────────────────────────────────────────────┐
│  Input (224×224×3)                           │
│  ┌───────────┐  ┌───────────┐               │
│  │ Conv2D 32 │→│ MaxPool   │               │
│  └───────────┘  └───────────┘               │
│  ┌───────────┐  ┌───────────┐               │
│  │ Conv2D 64 │→│ MaxPool   │               │
│  └───────────┘  └───────────┘               │
│  ┌───────────┐  ┌───────────┐               │
│  │Conv2D 128 │→│ MaxPool   │               │
│  └───────────┘  └───────────┘               │
│  ┌───────────┐  ┌───────────┐               │
│  │Conv2D 128 │→│ MaxPool   │               │
│  └───────────┘  └───────────┘               │
│  ┌─────────────────────────┐               │
│  │ Flatten → Dense 512     │               │
│  │ Dropout 0.5             │               │
│  │ Dense 256               │               │
│  │ Dropout 0.3             │               │
│  │ Dense 1 (Sigmoid) ✅🔥  │               │
│  └─────────────────────────┘               │
└─────────────────────────────────────────────┘
```

<br>

## 📂 Project Structure

```
fire-detection-cv/
├── 📄 app.py                              # Streamlit web application
├── 📄 SIMPLE_TRAINING_FIX.py              # Model training script (20 epochs)
├── 🧠 fire_detection_model_corrected.keras # Trained model weights
├── 📓 fire-detection-using-cnn-tenser-flow.ipynb  # Original notebook
├── 📄 requirements.txt                    # Python dependencies
└── 📄 README.md                           # You are here!
```

<br>

## 🛠️ Installation

### Prerequisites

| Requirement | Version |
|---|---|
| ![Python](https://img.shields.io/badge/Python-3.13+-blue) | 3.13 or higher |
| ![pip](https://img.shields.io/badge/pip-latest-orange) | Latest |

### Quick Start

```bash
# 1. Clone / download the project
git clone <repo-url> && cd fire-detection-cv

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify model file exists
ls fire_detection_model_corrected.keras

# 4. Launch the app 🚀
streamlit run app.py
```

> 🌐 App opens automatically at **http://localhost:8501**

<br>

## 🚀 Running the Application

| Method | Command |
|---|---|
| **Streamlit (Recommended)** | `streamlit run app.py` |
| **Python** | `python app.py` |

<br>

## 📖 Usage Guide

### Step 1 — Upload Video
Click the file uploader and select a video file (MP4, AVI, MOV, or MKV)

### Step 2 — Configure Settings
| Setting | Range | Recommendation |
|---|---|---|
| **Detection Threshold** | 0.1 – 0.9 | `0.3 – 0.7` for best balance |
| **Frame Skip Interval** | 1 – 20 | `1 – 5` for accuracy, `10 – 15` for speed |

### Step 3 — Analyze
Click **🔍 Analyze Video** and watch the real-time progress bar

### Step 4 — Review Results
| Result | Meaning |
|---|---|
| 🔴 **FIRE DETECTED** | At least one frame exceeds threshold |
| 🟢 **NO FIRE DETECTED** | All frames below threshold |
| 📸 **Fire Snapshot** | Frame with highest confidence |
| 📈 **Confidence Chart** | Prediction values across all frames |

<br>

## 📊 Algorithm Pipeline

```
Video File
    │
    ▼
┌──────────────┐
│ Frame Extract │  ← OpenCV reads every Nth frame
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Preprocess   │  BGR→RGB · Resize 224×224 · Normalize 0-1
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  CNN Predict  │  Sigmoid output → flip: 1 - prediction
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Threshold    │  confidence > threshold → FIRE 🔥
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   Results     │  Max confidence · Fire snapshot · Summary
└──────────────┘
```

<br>

## 🔧 Configuration

### Model Path
The app loads `fire_detection_model_corrected.keras` from the same directory by default. To change:

```python
# In app.py, change this line:
MODEL_PATH = "fire_detection_model_corrected.keras"  # Update to your path
```

### Retraining the Model

```bash
# 1. Update the dataset path in SIMPLE_TRAINING_FIX.py
# 2. Run training (20 epochs)
python SIMPLE_TRAINING_FIX.py

# 3. Model saves as fire_detection_model_corrected.keras
```

<br>

## ⚙️ Supported Formats & Browsers

| Video Formats | Browsers |
|---|---|
| MP4 · AVI · MOV · MKV | Chrome 80+ · Firefox 75+ · Safari 13+ · Edge 80+ |

<br>

## 🐛 Troubleshooting

| Issue | Solution |
|---|---|
| **Model file not found** | Ensure `fire_detection_model_corrected.keras` is in the same directory as `app.py` |
| **Unable to open video** | Verify format is supported and file is not corrupted |
| **App not starting** | Run `pip install -r requirements.txt` · Check Python ≥ 3.13 |
| **Slow processing** | Increase Frame Skip Interval to 10–15 · Use smaller video files |

### Performance Tips

| Tip | Setting |
|---|---|
| Faster processing | Frame skip `10–15` |
| Better accuracy | Frame skip `1–5` |
| Best threshold range | `0.3 – 0.7` |
| Optimal video size | Under 100 MB |

<br>

## 🤝 Contributing

1. 🍴 Fork the repository
2. 🌿 Create a feature branch
3. ✏️ Make your changes
4. ✅ Test thoroughly
5. 🔀 Submit a pull request

<br>

## 📄 License

This project is provided as-is for **educational and demonstration purposes**.

<br>

## 🆘 Support

1. Check the [Troubleshooting](#-troubleshooting) section above
2. Verify all files are present and dependencies installed
3. Test with a small video file first
4. Check system resources for large video processing

<br>

---

<div align="center">

**🔥 Fire Detection System**

*Built with ❤️ using TensorFlow & Streamlit*

[⬆ Back to Top](#-fire-detection-system)

</div>
