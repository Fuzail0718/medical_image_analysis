# simple_train.py
import tensorflow as tf
from tensorflow import keras
import os
import json

print("🔧 Training Simple Pneumonia Detection Model")

# Setup data
train_dir = './chest_xray/train'

# Simple data preparation
train_data = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values
    validation_split=0.2  # 20% for validation
)

# Load images
train_generator = train_data.flow_from_directory(
    train_dir,
    target_size=(150, 150),  # Small size for speed
    batch_size=32,
    class_mode='binary',  # Simple binary classification
    subset='training'
)

val_generator = train_data.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

print(f"✅ Loaded {train_generator.samples} training images")
print(f"✅ Classes: {train_generator.class_indices}")

# Very simple model
model = keras.Sequential([
    # Feature extraction
    keras.layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(150, 150, 3)),
    keras.layers.MaxPooling2D(2, 2),

    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(2, 2),

    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(2, 2),

    # Classification
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')  # Binary output
])

# Simple compilation
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("✅ Model created successfully!")

# Train model
print("🎯 Starting training...")
history = model.fit(
    train_generator,
    epochs=10,  # Few epochs for speed
    validation_data=val_generator,
    verbose=1
)

# Save everything
model.save('model.h5')
print("💾 Model saved as model.h5")

# Save class names
class_info = {
    'class_indices': train_generator.class_indices,
    'classes': ['NORMAL', 'PNEUMONIA']
}
with open('class_names.json', 'w') as f:
    json.dump(class_info, f, indent=2)
print("💾 Class info saved")

# Show final accuracy
final_acc = history.history['val_accuracy'][-1]
print(f"🎯 Final Validation Accuracy: {final_acc*100:.1f}%")
print("🚀 Training completed! Run: python simple_app.py")
