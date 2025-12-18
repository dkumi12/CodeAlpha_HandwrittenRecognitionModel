# Assets

This folder contains images and visual resources for the README.

## 📁 Current Assets

| File | Description | Status |
|------|-------------|--------|
| `demo.gif` | Demo of the Gradio interface in action | ⏳ To be added |
| `confusion_matrix.png` | Confusion matrix from model evaluation | ⏳ To be added |
| `training_curves.png` | Training/validation accuracy and loss curves | ⏳ To be added |
| `architecture.png` | Model architecture visualization | ⏳ To be added |

## 🎨 How to Generate These Assets

### 1. Demo GIF
- Open your Hugging Face Space
- Use a screen recorder (e.g., [ScreenToGif](https://www.screentogif.com/))
- Record drawing a character and getting prediction
- Save as `demo.gif` (keep under 5MB for GitHub)

### 2. Confusion Matrix
Add this to your training notebook:

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# After model evaluation
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Create confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)

# Plot
plt.figure(figsize=(15, 15))
sns.heatmap(cm, annot=False, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - 62 Classes')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()
```

### 3. Training Curves
Add this to your training notebook:

```python
import matplotlib.pyplot as plt

# Plot training history
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy
axes[0].plot(history.history['accuracy'], label='Training Accuracy')
axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
axes[0].set_title('Model Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True)

# Loss
axes[1].plot(history.history['loss'], label='Training Loss')
axes[1].plot(history.history['val_loss'], label='Validation Loss')
axes[1].set_title('Model Loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
plt.show()
```

### 4. Architecture Visualization
Add this to your training notebook:

```python
from tensorflow.keras.utils import plot_model

# Visualize model architecture
plot_model(
    model,
    to_file='architecture.png',
    show_shapes=True,
    show_layer_names=True,
    rankdir='TB',  # Top to Bottom
    expand_nested=True,
    dpi=150
)
```

## 📤 After Creating Assets

Once you've generated these images:

1. Add them to this `assets/` folder
2. Update the README to include them:

```markdown
## 🎥 Demo

![Demo](assets/demo.gif)

## 📊 Model Performance

### Training Curves
![Training Curves](assets/training_curves.png)

### Confusion Matrix
![Confusion Matrix](assets/confusion_matrix.png)

### Architecture
![Architecture](assets/architecture.png)
```

3. Commit and push:
```bash
git add assets/
git commit -m "docs: Add training visualizations and demo"
git push origin main
```
