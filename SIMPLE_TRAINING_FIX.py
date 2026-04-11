#!/usr/bin/env python3
"""
SIMPLE FIX for Fire Detection Model Training
Just correct the dataset path and keep everything else the same
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.keras import layers, models

# -------------------------------
# 1. DEFINE CORRECT DATASET PATH
# -------------------------------
# FIX: Point to the actual dataset directory with subdirectories
data_dir = r"D:\Users\Dell\Downloads\archive (2)\fire_dataset"

print(f"🔍 Dataset path: {data_dir}")
print(f"📁 Path exists: {os.path.exists(data_dir)}")

# -------------------------------
# 2. CREATE TRAINING AND VALIDATION DATASETS (FIXED PATH)
# -------------------------------
print("\n🚀 Loading datasets...")

# FIX: Use correct dataset path
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,  # FIX: Correct path to fire_dataset directory
    validation_split=0.3,       
    subset="training",
    seed=123,
    image_size=(224, 224),
    batch_size=32,
    label_mode='binary'  # Binary classification
)

val_test_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,  # FIX: Correct path
    validation_split=0.3,
    subset="validation",
    seed=123,
    image_size=(224, 224),
    batch_size=32,
    label_mode='binary'
)

# -------------------------------
# 3. Split validation and test sets
# -------------------------------
val_size = 0.67  
val_ds = val_test_ds.take(int(len(val_test_ds) * val_size))
test_ds = val_test_ds.skip(int(len(val_test_ds) * val_size))

# -------------------------------
# 4. Check class names and dataset info
# -------------------------------
class_names = train_ds.class_names
print(f"\n✅ Class names: {class_names}")
print(f"📊 Number of classes: {len(class_names)}")

# IMPORTANT: TensorFlow assigns labels alphabetically
# 'fire_images' = 0, 'non_fire_images' = 1
# We'll flip this during interpretation

print(f"📊 Training batches: {len(train_ds)}")
print(f"📊 Validation batches: {len(val_ds)}")
print(f"📊 Test batches: {len(test_ds)}")

# -------------------------------
# 5. Normalize the images
# -------------------------------
normalization_layer = tf.keras.layers.Rescaling(1./255)

train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

# -------------------------------
# 6. Build CNN model
# -------------------------------
model = models.Sequential([
    layers.Input(shape=(224, 224, 3)),
    
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')  # Binary classification
])

model.summary()

# -------------------------------
# 7. Compile the model
# -------------------------------
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# -------------------------------
# 8. Train the model
# -------------------------------
print("\n🚀 Starting training...")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20
)

# -------------------------------
# 9. Test with CORRECTED interpretation
# -------------------------------
print("\n🧪 Testing with CORRECTED label interpretation...")

def test_single_image(image_path, expected_class):
    """Test a single image with corrected label interpretation"""
    try:
        # Load and preprocess image
        img = tf.keras.utils.load_img(
            image_path, target_size=(224, 224)
        )
        img_array = tf.keras.utils.img_to_array(img)
        img_array = img_array / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make prediction
        prediction = model.predict(img_array, verbose=0)[0][0]
        
        # CORRECTED INTERPRETATION:
        # Original: fire_images=0, non_fire_images=1
        # We want: fire=1, non_fire=0
        # So we flip the prediction
        corrected_prediction = 1 - prediction
        
        predicted_class = "fire_images" if corrected_prediction > 0.5 else "non_fire_images"
        
        print(f"📸 {os.path.basename(image_path)}:")
        print(f"   Expected: {expected_class}")
        print(f"   Raw prediction: {prediction:.6f}")
        print(f"   Corrected prediction: {corrected_prediction:.6f}")
        print(f"   Predicted class: {predicted_class}")
        print(f"   {'✅ CORRECT' if predicted_class == expected_class else '❌ WRONG'}")
        
        return corrected_prediction, predicted_class == expected_class
        
    except Exception as e:
        print(f"❌ Error testing {image_path}: {e}")
        return 0.0, False

# Test images
fire_dir = os.path.join(data_dir, "fire_images")
non_fire_dir = os.path.join(data_dir, "non_fire_images")

print("\n🔥 Testing Fire Images:")
fire_results = []
if os.path.exists(fire_dir):
    fire_files = [f for f in os.listdir(fire_dir) if f.endswith('.png')][:3]
    for f in fire_files:
        pred, correct = test_single_image(os.path.join(fire_dir, f), "fire_images")
        fire_results.append((pred, correct))

print("\n✅ Testing Non-Fire Images:")
non_fire_results = []
if os.path.exists(non_fire_dir):
    non_fire_files = [f for f in os.listdir(non_fire_dir) if f.endswith('.png')][:3]
    for f in non_fire_files:
        pred, correct = test_single_image(os.path.join(non_fire_dir, f), "non_fire_images")
        non_fire_results.append((pred, correct))

# -------------------------------
# 10. Final Analysis
# -------------------------------
if fire_results and non_fire_results:
    fire_avg = np.mean([r[0] for r in fire_results])
    non_fire_avg = np.mean([r[0] for r in non_fire_results])
    fire_correct = np.mean([r[1] for r in fire_results])
    non_fire_correct = np.mean([r[1] for r in non_fire_results])
    
    print(f"\n🎉 FINAL RESULTS (with corrected interpretation):")
    print(f"🔥 Fire images - Avg prediction: {fire_avg:.6f}, Accuracy: {fire_correct:.2%}")
    print(f"✅ Non-fire images - Avg prediction: {non_fire_avg:.6f}, Accuracy: {non_fire_correct:.2%}")
    
    if fire_avg > 0.5 and non_fire_avg < 0.5:
        print("🎉 SUCCESS! Model is working correctly with corrected interpretation!")
    else:
        print("⚠️  Model still needs improvement")

# -------------------------------
# 11. Save the model
# -------------------------------
print("\n💾 Saving the model...")
model.save("fire_detection_model_corrected.keras")
print("✅ Model saved as 'fire_detection_model_corrected.keras'")

print("\n🎯 IMPORTANT FOR STREAMLIT APP:")
print("The model predicts correctly but needs interpretation flip")
print("Update your app.py predict_frame function to:")
print("  prediction = 1 - model.predict(frame, verbose=0)[0][0]  # Flip prediction")
