# ✍️ Handwritten Character Recognition Model

This project implements a **Convolutional Neural Network (CNN)** to recognize **handwritten characters**: digits (0–9), uppercase (A–Z), and lowercase letters (a–z). It includes a **Streamlit web app** for real-time predictions using uploaded image files.

This project was developed during my internship with **CodeAlpha**.

---

## 📊 Dataset

- **Size**: 37,995 grayscale images  
- **Image Dimensions**: 28x28 pixels  
- **Classes**: 62 (0–9, A–Z, a–z)  
- **Metadata**: `english.csv` maps image filenames to labels

---

## 🧠 Model

- **Architecture**: Convolutional Neural Network (CNN)  
- **Libraries**: TensorFlow, Keras  
- **Accuracy**: High performance on test data (details in training logs)

---

## 🛠️ Features

- Trains a CNN for 62-class classification  
- Preprocessing with grayscale normalization  
- Real-time predictions via a Streamlit web app  
- Confidence score displayed for each prediction  
- Clean UI for image uploads and results

---

## 🔧 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/dkumi12/CodeAlpha_HandwrittenRecognitionModel.git
cd CodeAlpha_HandwrittenRecognitionModel
```

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

### 3. Download Dataset

Download from [Google Drive](https://drive.google.com/drive/folders/1IeUA5T60VL9dx7b1QQCNKGGt3uFA1Ej9?usp=drive_link) and extract into the root folder as `Images/`.

---

## ▶️ Usage

### 🔁 Train the Model

```bash
python train.py
```

### 🌐 Run the Web App

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501` and upload an image to get a prediction!

---

## 📁 Project Structure

```
CodeAlpha_HandwrittenRecognitionModel/
├── app.py                    # Streamlit app
├── train.py                  # Model training
├── requirements.txt
├── english.csv               # Label metadata
├── model/                    # Trained model and label binarizer
└── Images/                   # Training data (external)
```

---

## 📫 Contact

**Name**: David Osei Kumi  
**Email**: [12dkumi@gmail.com](mailto:12dkumi@gmail.com)  
**GitHub**: [@dkumi12](https://github.com/dkumi12)

