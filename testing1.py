import tensorflow as tf
import cv2
import numpy as np

# Load the trained model from the .keras file
model = tf.keras.models.load_model('my_model1.keras')

# Path to the new image
new_image_path = r'C:\Bangtan\sitting.jpeg'
new_image = cv2.imread(new_image_path)
new_image = cv2.resize(new_image, (64, 64))
new_image = new_image / 255.0
new_image = np.expand_dims(new_image, axis=0)

# Predict the action
predicted_class = model.predict(new_image)

# Define the action labels (adjust this list based on your actual classes)
action_labels = ['walking', 'running', 'sitting', 'dancing']
predicted_action = action_labels[np.argmax(predicted_class)]

print(f'The predicted action is: {predicted_action}')
