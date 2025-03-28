import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations (optional)

# Suppress warnings globally
tf.get_logger().setLevel('ERROR')

from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image

# Load model
model = load_model(r"C:\Users\Acer\Desktop\Smart-Mathematics-Tutor-main\Smart-Mathematics-Tutor-main\Model Building\shapes.h5")  

# Optional: Compile the model if needed (e.g., for evaluation or retraining)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

def predict(InputImg):
    img = image.load_img(InputImg, target_size=(64, 64))  # Load and reshape the image
    x = image.img_to_array(img)  # Convert image to array
    x = np.expand_dims(x, axis=0)  # Expand dimensions to match input shape
    pred = np.argmax(model.predict(x), axis=-1)  # Get the class index
    index = ['circle', 'square', 'triangle']  # Class names
    result = str(index[pred[0]])  # Map index to class name
    return result
