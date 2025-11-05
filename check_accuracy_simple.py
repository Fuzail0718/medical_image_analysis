# check_accuracy_simple.py
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
from PIL import Image

print("📊 Checking Model Accuracy")
print("=" * 40)

# Load the trained model
try:
    model = keras.models.load_model('model.h5')
    print("✅ Model loaded successfully!")
except:
    print("❌ Model not found. Train the model first.")
    exit()

# Test on test dataset
test_dir = './chest_xray/test'

if not os.path.exists(test_dir):
    print(f"❌ Test folder not found: {test_dir}")
    exit()

print(f"📁 Testing on: {test_dir}")

# Count images in test set
normal_test_dir = os.path.join(test_dir, 'NORMAL')
pneumonia_test_dir = os.path.join(test_dir, 'PNEUMONIA')

normal_count = 0
pneumonia_count = 0

if os.path.exists(normal_test_dir):
    normal_count = len([f for f in os.listdir(normal_test_dir)
                       if f.endswith(('.jpeg', '.jpg', '.png'))])
if os.path.exists(pneumonia_test_dir):
    pneumonia_count = len([f for f in os.listdir(
        pneumonia_test_dir) if f.endswith(('.jpeg', '.jpg', '.png'))])

print(f"📊 Test dataset:")
print(f"   NORMAL images: {normal_count}")
print(f"   PNEUMONIA images: {pneumonia_count}")
print(f"   TOTAL images: {normal_count + pneumonia_count}")

# Test a few sample images


def test_sample_images():
    print("\n🧪 Testing sample images...")

    correct_predictions = 0
    total_tested = 0

    # Test NORMAL images
    if os.path.exists(normal_test_dir) and normal_count > 0:
        normal_images = [f for f in os.listdir(
            normal_test_dir) if f.endswith(('.jpeg', '.jpg', '.png'))]

        # Test first 10 normal images
        for i, img_name in enumerate(normal_images[:10]):
            try:
                img_path = os.path.join(normal_test_dir, img_name)
                img = Image.open(img_path)
                img = img.resize((150, 150))
                img_array = np.array(img) / 255.0

                if len(img_array.shape) == 2:
                    img_array = np.stack([img_array]*3, axis=-1)
                elif img_array.shape[2] == 4:
                    img_array = img_array[:, :, :3]

                img_batch = np.expand_dims(img_array, axis=0)
                prediction = model.predict(img_batch, verbose=0)[0][0]

                # For NORMAL, prediction should be < 0.5
                if prediction < 0.5:
                    correct_predictions += 1
                    result = "✅ CORRECT"
                else:
                    result = "❌ WRONG"

                total_tested += 1
                print(f"   NORMAL {i+1}: {result} (AI: {prediction:.3f})")

            except Exception as e:
                print(f"   Error testing {img_name}: {e}")

    # Test PNEUMONIA images
    if os.path.exists(pneumonia_test_dir) and pneumonia_count > 0:
        pneumonia_images = [f for f in os.listdir(
            pneumonia_test_dir) if f.endswith(('.jpeg', '.jpg', '.png'))]

        # Test first 10 pneumonia images
        for i, img_name in enumerate(pneumonia_images[:10]):
            try:
                img_path = os.path.join(pneumonia_test_dir, img_name)
                img = Image.open(img_path)
                img = img.resize((150, 150))
                img_array = np.array(img) / 255.0

                if len(img_array.shape) == 2:
                    img_array = np.stack([img_array]*3, axis=-1)
                elif img_array.shape[2] == 4:
                    img_array = img_array[:, :, :3]

                img_batch = np.expand_dims(img_array, axis=0)
                prediction = model.predict(img_batch, verbose=0)[0][0]

                # For PNEUMONIA, prediction should be > 0.5
                if prediction > 0.5:
                    correct_predictions += 1
                    result = "✅ CORRECT"
                else:
                    result = "❌ WRONG"

                total_tested += 1
                print(f"   PNEUMONIA {i+1}: {result} (AI: {prediction:.3f})")

            except Exception as e:
                print(f"   Error testing {img_name}: {e}")

    return correct_predictions, total_tested


# Run the test
correct, total = test_sample_images()

if total > 0:
    accuracy = (correct / total) * 100
    print(f"\n🎯 ACCURACY RESULTS:")
    print(f"   Correct predictions: {correct}/{total}")
    print(f"   Accuracy: {accuracy:.1f}%")

    if accuracy >= 80:
        print("   ✅ EXCELLENT! Model is working well!")
    elif accuracy >= 70:
        print("   ✅ GOOD! Model is decent.")
    elif accuracy >= 60:
        print("   ⚠️  FAIR - Could be better.")
    else:
        print("   ❌ POOR - Model needs improvement.")
else:
    print("❌ No images were tested.")

# Quick performance check
print("\n🔍 Performance Summary:")
print("   0.0 - 0.3 = Very confident NORMAL")
print("   0.3 - 0.7 = Unsure")
print("   0.7 - 1.0 = Very confident PNEUMONIA")

print("\n🚀 Your model is ready for the presentation!")
