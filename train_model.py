import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Define dataset and model save paths
train_path = r'C:\Users\FINRISE\Desktop\mask_detection_app\dataset'
model_path = r'C:\Users\FINRISE\Desktop\mask_detection_app\model\mask_detector.h5'

# Data Preprocessing
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Training data generator
train_gen = datagen.flow_from_directory(
    train_path,
    target_size=(100, 100),
    batch_size=32,
    class_mode='binary',
    subset='training',
    shuffle=True
)

# Validation data generator
val_gen = datagen.flow_from_directory(
    train_path,
    target_size=(100, 100),
    batch_size=32,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

# Print class labels mapping
print("Class indices:", train_gen.class_indices)

# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary classification
])

# Compile the model
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Optional: Early stopping and model checkpoint
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
checkpoint = ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True)

# Train the model
model.fit(
    train_gen,
    epochs=10,  # Adjust as needed
    validation_data=val_gen,
    callbacks=[early_stop, checkpoint]
)

# Ensure model save directory exists (optional if using checkpoint)
os.makedirs(os.path.dirname(model_path), exist_ok=True)

# Save the model (optional if checkpoint is used)
model.save(model_path)
print(f"Model saved to: {model_path}")
