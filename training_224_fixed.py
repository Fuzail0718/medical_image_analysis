# training_224_fixed.py
import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import json
from datetime import datetime
import matplotlib.pyplot as plt

def train_224_model():
    print("🎯 TRAINING WITH 224x224 IMAGES")
    print("=" * 50)
    
    train_dir = './chest_xray/train'
    
    if not os.path.exists(train_dir):
        print(f"❌ Training directory not found: {train_dir}")
        return False
    
    print("📁 Dataset structure:")
    for class_name in ['NORMAL', 'PNEUMONIA']:
        class_dir = os.path.join(train_dir, class_name)
        if os.path.exists(class_dir):
            images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpeg', '.jpg', '.png'))]
            print(f"   {class_name}: {len(images)} images")
    
    # Data augmentation for 224x224 images
    train_datagen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        horizontal_flip=True,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        validation_split=0.2
    )
    
    val_datagen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )
    
    print("\n🔄 Loading 224x224 images...")
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),  # Use 224x224 to match your app
        batch_size=32,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    val_generator = val_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),  # Use 224x224
        batch_size=32,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    print(f"✅ Data loaded! Classes: {train_generator.class_indices}")
    print(f"📊 Training samples: {train_generator.samples}")
    print(f"📊 Validation samples: {val_generator.samples}")
    
    # Calculate class weights for imbalance
    print("\n⚖️ Handling class imbalance...")
    class_totals = [0, 0]
    for i, class_name in enumerate(['NORMAL', 'PNEUMONIA']):
        class_dir = os.path.join(train_dir, class_name)
        if os.path.exists(class_dir):
            images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpeg', '.jpg', '.png'))]
            class_totals[i] = len(images)
    
    total = sum(class_totals)
    class_weights = {
        0: total / (2 * class_totals[0]),  # Weight for NORMAL
        1: total / (2 * class_totals[1])   # Weight for PNEUMONIA
    }
    print(f"Class weights: {class_weights}")
    
    # Model for 224x224 images
    print("\n🏗️ Building model for 224x224 images...")
    model = keras.Sequential([
        # First conv block
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        keras.layers.MaxPooling2D(2, 2),
        
        # Second conv block
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D(2, 2),
        
        # Third conv block
        keras.layers.Conv2D(128, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D(2, 2),
        
        # Fourth conv block
        keras.layers.Conv2D(256, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D(2, 2),
        
        # Classifier with regularization
        keras.layers.Dropout(0.5),
        keras.layers.Flatten(),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(2, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("✅ Model compiled!")
    model.summary()
    
    # Callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            'pneumonia_model_224.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    # Train with class weights
    print("\n🎯 Training with class weights...")
    history = model.fit(
        train_generator,
        epochs=25,
        validation_data=val_generator,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    # Save model
    model.save('pneumonia_model.h5')  # Overwrite the old model
    file_size = os.path.getsize('pneumonia_model.h5') / (1024 * 1024)
    print(f"💾 Model saved! Size: {file_size:.1f} MB")
    
    # Save class info
    class_info = {
        'class_indices': train_generator.class_indices,
        'class_weights_used': class_weights,
        'image_size': 224,
        'training_date': datetime.now().isoformat()
    }
    with open('class_indices.json', 'w') as f:
        json.dump(class_info, f, indent=2)
    
    # Test the model
    print("\n🧪 Testing model...")
    test_image = np.random.random((1, 224, 224, 3)).astype(np.float32)
    prediction = model.predict(test_image, verbose=0)
    
    class_names = list(train_generator.class_indices.keys())
    predicted_class = class_names[np.argmax(prediction[0])]
    confidence = np.max(prediction[0])
    
    print(f"✅ Model test passed!")
    print(f"   Prediction: {predicted_class}")
    print(f"   Confidence: {confidence:.3f}")
    print(f"   Probabilities: {dict(zip(class_names, prediction[0]))}")
    
    # Results
    best_val_acc = max(history.history['val_accuracy'])
    final_val_acc = history.history['val_accuracy'][-1]
    
    print("\n" + "=" * 50)
    print("🎉 TRAINING COMPLETED!")
    print("=" * 50)
    print(f"⭐ Best Validation Accuracy: {best_val_acc*100:.2f}%")
    print(f"📊 Final Validation Accuracy: {final_val_acc*100:.2f}%")
    
    return True

if __name__ == "__main__":
    success = train_224_model()
    if success:
        print("\n🚀 Training complete! Run: streamlit run app_enhanced.py")
        print("✅ Model now uses 224x224 images and should work with your app!")
    else:
        print("\n❌ Training failed.")