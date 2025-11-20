import cv2
import numpy as np

def preprocess_image(image_path, target_size=(128, 128)):
    """
    Reads an image, converts it to grayscale, resizes it, 
    normalizes it, and reshapes it for the model.
    """
    try:
        # Read image in grayscale
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not read image at {image_path}")

        # Resize
        img = cv2.resize(img, target_size)

        # Normalize to 0-1
        img = img.astype('float32') / 255.0

        # Reshape for CNN-LSTM input
        # The model likely expects (batch_size, time_steps, features) or (batch_size, height, width, channels)
        # For CNN-LSTM, we often treat the image as a sequence of rows or patches.
        # Let's assume a standard CNN input for now (128, 128, 1) and let the model handle the reshaping/time-distributed parts
        # OR if it's purely CNN-LSTM on raw signal, but here we have images.
        # Let's stick to a 4D tensor for Keras: (1, 128, 128, 1)
        img = np.expand_dims(img, axis=-1) # (128, 128, 1)
        img = np.expand_dims(img, axis=0)  # (1, 128, 128, 1)
        
        return img
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return None
