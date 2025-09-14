import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

# Paths to the dataset folders
IMAGE_PATH = 'C:/archive/weizmann/images'
LABEL_PATH = 'C:/archive/weizmann/labels'

# Function to load images and labels
def load_images_and_labels(image_path, label_path, img_size=(64, 64)):
    image_files = [f for f in os.listdir(image_path) if f.endswith('.jpg')]
    X = []
    y = []

    for img_file in image_files:
        # Load and preprocess image
        img_path = os.path.join(image_path, img_file)
        image = cv2.imread(img_path)
        image = cv2.resize(image, img_size)
        X.append(image)

        # Load label from the corresponding text file
        label_file = img_file.replace('.jpg', '.txt')
        label_path_full = os.path.join(label_path, label_file)
        with open(label_path_full, 'r') as file:
            # Read the first value in the line (assuming it's the label)
            label_line = file.read().strip().split()  # Split the line into parts
            label = float(label_line[0])  # Use the first value as label (or modify as needed)
        
        # Append the label to the list
        y.append(label)

    X = np.array(X) / 255.0  # Normalize images
    y = np.array(y)

    # If needed, categorize labels (e.g., based on action types or labels range)
    y = to_categorical(y, num_classes=len(set(y)))  # One-hot encode labels

    return X, y

# Load images and labels
X, y = load_images_and_labels(IMAGE_PATH, LABEL_PATH)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an instance of ImageDataGenerator for data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Fit the generator on the training data
datagen.fit(X_train)

# Build a simple Convolutional Neural Network (CNN)
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(y.shape[1], activation='softmax')  # Use the number of classes from one-hot encoding
])

# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model using the generator
model.fit(datagen.flow(X_train, y_train, batch_size=32), 
          validation_data=(X_test, y_test), 
          epochs=10)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {accuracy:.2f}')

# Make predictions
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Print classification report and confusion matrix
print("Classification Report:")
print(classification_report(y_true_classes, y_pred_classes))

print("Confusion Matrix:")
print(confusion_matrix(y_true_classes, y_pred_classes))

# Save the model
model.save('my_model1.keras', save_format='keras')