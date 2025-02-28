import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import pickle

# Define paths
metadata_path = 'english.csv'
base_dir = r'C:\Users\abami\OneDrive\Desktop\CodeAlpha_HandwrittenRecognition\Images'  # Use raw string to handle backslashes

# Load metadata
metadata = pd.read_csv(metadata_path)
print(metadata.head())

# Function to load and preprocess images
def load_and_preprocess_images(metadata, base_dir):
    images = []
    labels = []
    for index, row in metadata.iterrows():
        filename = row['image']
        label = row['label']
        img_path = os.path.join(base_dir, filename)
        try:
            img = load_img(img_path, color_mode='grayscale', target_size=(28, 28))
            img_array = img_to_array(img) / 255.0
            images.append(img_array)
            labels.append(label)
        except FileNotFoundError:
            print(f"Image not found: {img_path}")
            continue
    return np.array(images), np.array(labels)

# Load and preprocess
X, y = load_and_preprocess_images(metadata, base_dir)
X = X.reshape((X.shape[0], 28, 28, 1))

# Encode labels
lb = LabelBinarizer()
y = lb.fit_transform(y)
num_classes = len(lb.classes_)
print(f"Number of classes: {num_classes}")  # Should be 62

# Save label binarizer
with open('label_binarizer.pkl', 'wb') as f:
    pickle.dump(lb, f)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define model
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# Compile and train
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, validation_split=0.2, epochs=10, batch_size=128)

# Evaluate
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_accuracy:.4f}')

# Save model
model.save('handwritten_character_recognition_model.h5')

# Optional: Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()