import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Load the trained model
model_path = 'crop.h5'
if os.path.exists(model_path):
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully.")
else:
    print(f"Model file {model_path} not found!")
    exit()

# Class labels mapping
class_labels = {0: 'jute', 1: 'maize', 2: 'rice', 3: 'sugarcane', 4: 'wheat'}

# Directory containing crop images
base_dir = r"C:\Users\PMLS\Desktop\Projects\data science\dj-dl-fertilizer\Crop_Dataset\crop_images"

# Function to preprocess and predict an image
def predict_image(img_path):
    print(f"Processing image: {img_path}")  # Debug print
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Expand dims to match input shape
    img_array = img_array / 255.0  # Normalize

    predictions = model.predict(img_array)
    print(f"Raw predictions: {predictions}")  # Debug print
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions)  # Get confidence score
    return class_labels[predicted_class], confidence

# Check five random images from each class
for class_index, crop_class in class_labels.items():
    crop_class_path = os.path.join(base_dir, crop_class)
    
    if not os.path.exists(crop_class_path):
        print(f"Skipping {crop_class}, directory not found: {crop_class_path}")
        continue

    image_files = os.listdir(crop_class_path)[:5]  # Get first five images
    print(f"\nChecking {crop_class} ({len(image_files)} images found)...")

    if not image_files:
        print(f"No images found in {crop_class_path}")
        continue

    for img_file in image_files:
        img_path = os.path.join(crop_class_path, img_file)
        if not os.path.isfile(img_path):
            print(f"Skipping non-file: {img_path}")
            continue  # Skip non-file paths

        prediction, confidence = predict_image(img_path)
        print(f"{img_file} → Predicted: {prediction} (Confidence: {confidence:.2f})")
