# Model Card: Handwritten Character Recognition

## Model Details

| Attribute | Details |
|-----------|---------|
| **Model Name** | Handwritten Character Recognition CNN |
| **Model Type** | Convolutional Neural Network (Transfer Learning) |
| **Architecture** | MobileNetV2 + Custom Classification Head |
| **Developer** | Daniel Kumi |
| **Date** | December 2025 |
| **Version** | 1.0 |
| **Framework** | TensorFlow 2.x / Keras |
| **License** | MIT |

---

## Intended Use

### Primary Use Cases
- Educational demonstrations of deep learning
- Handwriting digitization prototypes
- Character recognition research and experimentation
- Interactive AI demos for portfolios

### Out-of-Scope Uses
- Production OCR systems (not optimized for scale)
- Connected/cursive handwriting recognition
- Multi-character or word recognition
- Real-time video stream processing

---

## Training Data

### Dataset Overview
| Attribute | Value |
|-----------|-------|
| Total Samples | 3,410 images |
| Training Set | 2,046 (60%) |
| Validation Set | 682 (20%) |
| Test Set | 682 (20%) |
| Image Dimensions | 128×128 pixels |
| Color Mode | Grayscale converted to RGB |

### Class Distribution
- **62 balanced classes:**
  - Digits: 0-9 (10 classes)
  - Uppercase: A-Z (26 classes)
  - Lowercase: a-z (26 classes)

### Data Augmentation
- Rotation: ±10°
- Width/Height Shift: 12%
- Zoom: 12%
- Shear: 5%

---

## Evaluation Results

### Performance Metrics

| Metric | Score |
|--------|-------|
| **Top-1 Accuracy** | 75.37% |
| **Top-3 Accuracy** | 94.28% |
| Loss Function | Categorical Cross-Entropy |

### Confusion Analysis

Top confusion pairs (visually similar characters):

| True Label | Predicted As | Confusion Rate |
|------------|--------------|----------------|
| O (letter) | 0 (zero) | 54.5% |
| I (letter) | 1 (one) | 45.5% |
| 5 (five) | 3 (three) | 36.4% |
| l (lowercase L) | 1 (one) | 27.3% |

---

## Model Architecture

### Base Model
- **MobileNetV2** pre-trained on ImageNet
- Lightweight architecture suitable for deployment
- ~3.4M parameters in base

### Custom Classification Head
```
GlobalAveragePooling2D
    ↓
Dropout (rate=0.4)
    ↓
Dense (512 units, ReLU activation)
    ↓
Dense (62 units, Softmax activation)
```

### Training Configuration
| Parameter | Phase 1 | Phase 2 |
|-----------|---------|---------|
| Base Layers | Frozen | Last 30 unfrozen |
| Learning Rate | 1e-3 | 1e-4 |
| Epochs | 12 planned | 18 planned |
| Batch Size | 32 | 32 |

---

## Limitations

### Technical Limitations
1. **Isolated Characters Only:** Model expects single, centered characters
2. **Clean Input Required:** Performs best on clear, high-contrast images
3. **Fixed Input Size:** Images resized to 128×128 may lose detail
4. **Similar Character Confusion:** Struggles with O/0, I/1/l pairs

### Ethical Considerations
- Model trained on limited dataset demographics
- May have biases toward certain handwriting styles
- Not suitable for high-stakes decisions

---

## Deployment

### Infrastructure
- **Hosting:** Hugging Face Spaces
- **Container:** Docker
- **Backend:** FastAPI (port 8000)
- **Frontend:** Streamlit (port 7860)

### API Endpoint
```
POST /predict
Content-Type: multipart/form-data
Body: file=<image_file>

Response: {
  "prediction": "A",
  "confidence": 0.9823,
  "class_id": 10
}
```

---

## Citation

If you use this model in your research or projects, please cite:

```bibtex
@misc{kumi2025handwritten,
  author = {Kumi, Daniel},
  title = {Handwritten Character Recognition using MobileNetV2},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/dkumi12/CodeAlpha_HandwrittenRecognitionModel}
}
```

---

## Contact

**Daniel Kumi**  
ML Engineer | AWS Certified AI Practitioner

- GitHub: [@dkumi12](https://github.com/dkumi12)
- LinkedIn: [Daniel Kumi](https://www.linkedin.com/in/daniel-kumi-9b5834205/)
- Demo: [Hugging Face Space](https://huggingface.co/spaces/dkumi12/handwritten-cnn)
