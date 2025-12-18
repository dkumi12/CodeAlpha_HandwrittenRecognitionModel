---
title: Handwritten Character Recognition
emoji: ✍️
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 7860
---

# ✍️ Handwritten Character Recognition AI

[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/dkumi12/handwritten-cnn)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)](https://docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A deep learning system for recognizing handwritten characters (A-Z, a-z, 0-9) using **MobileNetV2 transfer learning**, achieving **94.28% top-3 accuracy** across 62 character classes.

---

## 🎯 Model Performance

| Metric | Score |
|--------|-------|
| **Top-1 Accuracy** | 75.37% |
| **Top-3 Accuracy** | 94.28% |
| Total Classes | 62 |
| Dataset Size | 3,410 images |

> The 94%+ top-3 accuracy means the correct character is almost always in the model's top 3 predictions.

---

## 🎥 Demo

> **Note:** Demo GIF will be added soon. In the meantime, try the [live app](https://huggingface.co/spaces/YOUR_SPACE_URL)!

<!-- Uncomment when demo.gif is ready:
![Demo](assets/demo.gif)
-->

---

## 🏗️ Architecture

### Model Design

- **Base Model:** MobileNetV2 (ImageNet pre-trained weights)
- **Input Shape:** 128×128×3 (RGB)
- **Custom Classification Head:**
  - GlobalAveragePooling2D
  - Dropout (0.4)
  - Dense (512 units, ReLU)
  - Dense (62 units, Softmax)

### Training Strategy
1. **Phase 1 - Feature Extraction:** Frozen MobileNetV2 base, train custom head
2. **Phase 2 - Fine-tuning:** Unfreeze last 30 layers for domain adaptation

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Hugging Face Spaces                      │
│  ┌─────────────────────┐    ┌─────────────────────────────┐ │
│  │   Streamlit UI      │───▶│     FastAPI Backend         │ │
│  │   (Port 7860)       │    │     (Port 8000)             │ │
│  │  • Drawing Canvas   │    │    • /predict endpoint      │ │
│  │  • File Upload      │    │    • Image preprocessing    │ │
│  └─────────────────────┘    └─────────────────────────────┘ │
│                                        │                    │
│                              ┌─────────▼─────────┐          │
│                              │  MobileNetV2 CNN  │          │
│                              │  (62-class output)│          │
│                              └───────────────────┘          │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔧 Tech Stack

| Component | Technology |
|-----------|------------|
| ML Framework | TensorFlow / Keras |
| Model | MobileNetV2 (Transfer Learning) |
| Backend API | FastAPI + Uvicorn |
| Frontend UI | Streamlit + Drawable Canvas |
| Containerization | Docker |
| CI/CD | GitHub Actions → Hugging Face |
| Model Storage | Google Drive (via gdown) |

---

## 📊 Dataset Details

| Attribute | Value |
|-----------|-------|
| Total Images | 3,410 |
| Training Set | 2,046 (60%) |
| Validation Set | 682 (20%) |
| Test Set | 682 (20%) |
| Image Size | 128×128 pixels |
| Color Mode | Grayscale → RGB |

### Classes (62 total)
- **Digits:** 0-9 (10 classes)
- **Uppercase Letters:** A-Z (26 classes)
- **Lowercase Letters:** a-z (26 classes)

---

## 🔍 Known Confusion Patterns

The model shows expected confusion between visually similar characters:

| Predicted | Actual | Confusion Rate | Reason |
|-----------|--------|----------------|--------|
| 0 (zero) | O (letter) | 54.5% | Nearly identical shapes |
| 1 (one) | I (letter) | 45.5% | Vertical line similarity |
| 3 (three) | 5 (five) | 36.4% | Curved segments |
| 1 (one) | l (lowercase L) | 27.3% | Vertical line similarity |

> These are the same character pairs that humans often struggle to distinguish in handwriting!

### 📈 Training Visualizations

> **Note:** Visualizations will be added soon. See `assets/README.md` for instructions on generating them.

<!-- Uncomment when images are ready:
#### Training Curves
![Training Curves](assets/training_curves.png)

#### Confusion Matrix
![Confusion Matrix](assets/confusion_matrix.png)

#### Model Architecture
![Architecture](assets/architecture.png)
-->

---

## 🚀 Live Demo

**[Try the Demo on Hugging Face Spaces →](https://huggingface.co/spaces/dkumi12/handwritten-cnn)**

### How to Use:
1. **Draw** a character on the canvas (or upload an image)
2. Click **Identify Character**
3. View the prediction and confidence score

---

## 🏃 Run Locally

### Prerequisites
- Python 3.9+
- Docker (optional)

### Option 1: Docker (Recommended)
```bash
git clone https://github.com/dkumi12/CodeAlpha_HandwrittenRecognitionModel.git
cd CodeAlpha_HandwrittenRecognitionModel
docker build -t handwritten-cnn .
docker run -p 7860:7860 handwritten-cnn
```

### Option 2: Manual Setup
```bash
git clone https://github.com/dkumi12/CodeAlpha_HandwrittenRecognitionModel.git
cd CodeAlpha_HandwrittenRecognitionModel
pip install -r requirements.txt
./run_app.sh
```

Then open `http://localhost:7860` in your browser.

---

## 📁 Project Structure

```
CodeAlpha_HandwrittenRecognitionModel/
├── main.py                 # FastAPI backend + inference logic
├── ui.py                   # Streamlit frontend with canvas
├── requirements.txt        # Python dependencies
├── Dockerfile              # Container configuration
├── run_app.sh              # Startup script (runs both services)
├── test_api.py             # API testing utilities
├── training/               # Model training resources
│   └── train_mobilenet.ipynb
├── docs/                   # Documentation & visualizations
│   └── MODEL_CARD.md
└── .github/
    └── workflows/
        └── deploy.yml      # CI/CD to Hugging Face
```

---

## 📈 Future Improvements

- [ ] Complete Phase 2 fine-tuning (training interrupted at epoch 6)
- [ ] Implement attention mechanisms for confusable character pairs
- [ ] Add ensemble methods combining multiple model architectures
- [ ] Implement confidence threshold filtering
- [ ] Add batch prediction support
- [ ] Create model versioning system

---

## 🛠️ API Reference

### POST `/predict`

Accepts an image file and returns the predicted character.

**Request:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_image.png"
```

**Response:**
```json
{
  "prediction": "A",
  "confidence": 0.9823,
  "class_id": 10
}
```

---

## 👤 Author

**Daniel Kumi**  
ML Engineer | AWS Certified AI Practitioner

- 🔗 [LinkedIn](https://www.linkedin.com/in/daniel-kumi-9b5834205/)
- 🐙 [GitHub](https://github.com/dkumi12)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🙏 Acknowledgments

- [CodeAlpha](https://www.codealpha.tech/) - ML Internship Program
- [Hugging Face](https://huggingface.co/) - Model hosting platform
- [TensorFlow](https://tensorflow.org/) - ML framework
