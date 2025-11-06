# check_accuracy.py
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
from PIL import Image


def check_accuracy():
    print("🧪 CHECKING MODEL ACCURACY")
    print("=" * 50)

    # Load model
    try:
        model = keras.models.load_model('pneumonia_model.h5')
        print("✅ Model loaded successfully!")
    except:
        print("❌ Model not found. Train the model first.")
        return

    # Test directory
    test_dir = './chest_xray/test'

    if not os.path.exists(test_dir):
        print(f"❌ Test directory not found: {test_dir}")
        return

    # Count test images
    normal_dir = os.path.join(test_dir, 'NORMAL')
    pneumonia_dir = os.path.join(test_dir, 'PNEUMONIA')

    normal_count = len([f for f in os.listdir(normal_dir) if f.endswith(
        ('.jpeg', '.jpg', '.png'))]) if os.path.exists(normal_dir) else 0
    pneumonia_count = len([f for f in os.listdir(pneumonia_dir) if f.endswith(
        ('.jpeg', '.jpg', '.png'))]) if os.path.exists(pneumonia_dir) else 0

    print(f"📊 Test Dataset:")
    print(f"   NORMAL images: {normal_count}")
    print(f"   PNEUMONIA images: {pneumonia_count}")
    print(f"   TOTAL images: {normal_count + pneumonia_count}")

    # Test samples from each class
    print("\n🔍 Testing on sample images...")

    correct_predictions = 0
    total_tested = 0
    results = []

    # Test NORMAL images
    if os.path.exists(normal_dir) and normal_count > 0:
        normal_images = [f for f in os.listdir(normal_dir) if f.endswith(
            ('.jpeg', '.jpg', '.png'))][:5]  # Test 5 images

        for img_name in normal_images:
            try:
                img_path = os.path.join(normal_dir, img_name)
                img = Image.open(img_path)
                img = img.resize((224, 224))  # Match your training size
                img_array = np.array(img) / 255.0

                # Handle image formats
                if len(img_array.shape) == 2:  # Grayscale
                    img_array = np.stack([img_array]*3, axis=-1)
                elif img_array.shape[2] == 4:  # RGBA
                    img_array = img_array[:, :, :3]

                img_batch = np.expand_dims(img_array, axis=0)
                prediction = model.predict(img_batch, verbose=0)[0]

                predicted_class = np.argmax(prediction)
                confidence = np.max(prediction)

                # For NORMAL, class index should be 0
                is_correct = (predicted_class == 0)
                if is_correct:
                    correct_predictions += 1
                    result = "✅ CORRECT"
                else:
                    result = "❌ WRONG"

                total_tested += 1
                results.append(
                    f"NORMAL: {result} (Confidence: {confidence:.3f})")

            except Exception as e:
                print(f"   Error testing {img_name}: {e}")

    # Test PNEUMONIA images
    if os.path.exists(pneumonia_dir) and pneumonia_count > 0:
        pneumonia_images = [f for f in os.listdir(pneumonia_dir) if f.endswith(
            ('.jpeg', '.jpg', '.png'))][:5]  # Test 5 images

        for img_name in pneumonia_images:
            try:
                img_path = os.path.join(pneumonia_dir, img_name)
                img = Image.open(img_path)
                img = img.resize((224, 224))
                img_array = np.array(img) / 255.0

                if len(img_array.shape) == 2:
                    img_array = np.stack([img_array]*3, axis=-1)
                elif img_array.shape[2] == 4:
                    img_array = img_array[:, :, :3]

                img_batch = np.expand_dims(img_array, axis=0)
                prediction = model.predict(img_batch, verbose=0)[0]

                predicted_class = np.argmax(prediction)
                confidence = np.max(prediction)

                # For PNEUMONIA, class index should be 1
                is_correct = (predicted_class == 1)
                if is_correct:
                    correct_predictions += 1
                    result = "✅ CORRECT"
                else:
                    result = "❌ WRONG"

                total_tested += 1
                results.append(
                    f"PNEUMONIA: {result} (Confidence: {confidence:.3f})")

            except Exception as e:
                print(f"   Error testing {img_name}: {e}")

    # Display individual results
    print("\n📈 Individual Test Results:")
    for result in results:
        print(f"   {result}")

    # Calculate accuracy
    if total_tested > 0:
        accuracy = (correct_predictions / total_tested) * 100

        print(f"\n🎯 ACCURACY SUMMARY:")
        print(f"   Correct predictions: {correct_predictions}/{total_tested}")
        print(f"   Accuracy: {accuracy:.1f}%")

        # Performance rating
        if accuracy >= 90:
            rating = "🎉 EXCELLENT"
        elif accuracy >= 80:
            rating = "✅ VERY GOOD"
        elif accuracy >= 70:
            rating = "👍 GOOD"
        elif accuracy >= 60:
            rating = "⚠️  FAIR"
        else:
            rating = "❌ NEEDS IMPROVEMENT"

        print(f"   Rating: {rating}")

        # Confidence interpretation
        print(f"\n💡 Confidence Guide:")
        print(f"   0.0-0.3 = Very confident NORMAL")
        print(f"   0.3-0.7 = Unsure")
        print(f"   0.7-1.0 = Very confident PNEUMONIA")

    else:
        print("❌ No images were successfully tested.")

    print(f"\n🚀 Model is ready for demonstration!")


if __name__ == "__main__":
    check_accuracy()
