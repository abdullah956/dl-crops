import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Load the trained model
model = tf.keras.models.load_model('crop.h5')

# Class labels
class_labels = {0: 'jute', 1: 'maize', 2: 'rice', 3: 'sugarcane', 4: 'wheat'}

# Load and preprocess the image
img_path = r"C:\Users\PMLS\Desktop\Projects\data science\dj-dl-fertilizer\Crop_Dataset\crop_images\wheat\wheat0001a.jpeg"  # Replace with your image path
img = image.load_img(img_path, target_size=(150, 150))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # Expand to match model input shape
img_array = img_array / 255.0  # Normalize

# Predict
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions)

# Output result
print(f'Predicted Crop: {class_labels[predicted_class]}')

