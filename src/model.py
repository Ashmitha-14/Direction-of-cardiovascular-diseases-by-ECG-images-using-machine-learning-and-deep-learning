def build_model(input_shape=(128, 128, 1), num_classes=4):
    """
    Builds a CNN-LSTM model for ECG image classification.
    """
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, LSTM, TimeDistributed, Reshape, Dropout

    model = Sequential()

    # CNN layers wrapped in TimeDistributed are typical for video, 
    # but for single images, we usually use CNN to extract features then reshape for LSTM 
    # OR we slice the image. 
    # Given the user description "CNN-LSTM model to detect... from ECG images", 
    # a common approach for static images is to treat rows as time steps.
    
    # Let's assume we treat the image height as time steps and width as features (or vice versa).
    # Input: (128, 128, 1)
    
    # Option 1: CNN feature extractor -> Reshape -> LSTM
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    
    # Current shape: (Batch, 14, 14, 128) approx depending on padding
    # We need to reshape this to (Batch, TimeSteps, Features) for LSTM
    # Let's flatten the spatial dimensions into a sequence
    
    # Calculate shape after pooling: 128 -> 64 -> 32 -> 16 (approx)
    # Let's use a Reshape layer to prepare for LSTM.
    # We can treat one spatial dimension as time.
    
    # Actually, a simpler "CNN-LSTM" often implies TimeDistributed(CNN) -> LSTM
    # But that requires the input to be a sequence of frames. 
    # Since we have single images, we will interpret the "CNN-LSTM" requirement as:
    # Extract features with CNN, then sequence modeling on the feature map.
    
    model.add(Reshape((-1, 128))) # Reshape to (Batch, SequenceLength, Features)
    
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(0.5))
    
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

if __name__ == "__main__":
    model = build_model()
    model.summary()
