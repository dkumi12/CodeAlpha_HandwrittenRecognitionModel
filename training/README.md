# Training Resources

This folder contains the model training notebooks and resources.

## 📁 Contents

| File | Description |
|------|-------------|
| `train_mobilenet.ipynb` | Main training notebook with MobileNetV2 transfer learning |
| `model_evaluation.ipynb` | Confusion matrix analysis and metrics visualization |
| `data_exploration.ipynb` | Dataset analysis and preprocessing exploration |

## 🏋️ Training Summary

### Model Evolution

1. **Simple CNN** - Initial attempt, 40.91% accuracy (baseline)
2. **Improved CNN** - Added BatchNorm, Dropout, L2 regularization
3. **MobileNetV2** - Transfer learning, achieved 75.37% top-1 / 94.28% top-3

### Final Model Configuration

```python
# Base Model
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(128, 128, 3)
)

# Custom Head
x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.4)(x)
x = Dense(512, activation='relu')(x)
outputs = Dense(62, activation='softmax')(x)
```

### Training Parameters

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning Rate | 1e-3 (Phase 1), 1e-4 (Phase 2) |
| Batch Size | 32 |
| Image Size | 128×128 |
| Augmentation | Rotation, Shifts, Zoom, Shear |

### Data Augmentation

```python
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.12,
    height_shift_range=0.12,
    zoom_range=0.12,
    shear_range=0.05,
    fill_mode='nearest'
)
```

## 📊 Results

- **Top-1 Accuracy:** 75.37%
- **Top-3 Accuracy:** 94.28%
- **Training Status:** Phase 1 completed (6 epochs), Phase 2 planned

## 🔄 To Reproduce

1. Upload the training notebook to Google Colab
2. Mount Google Drive with the dataset
3. Run all cells sequentially
4. Model will be saved to `model/mobilenet_best.h5`

## 📝 Notes

- Training was performed on Google Colab with GPU runtime
- Dataset is balanced across all 62 classes
- Class weighting was applied during training
