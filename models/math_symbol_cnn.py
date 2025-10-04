from tensorflow.keras import layers, models
from tensorflow.keras.regularizers import l2

def build_cnn_model(input_shape=(28, 28, 1), num_classes=16):
    """
    Enhanced CNN architecture optimized for handwritten math symbol recognition
    """
    model = models.Sequential([
        # First convolutional block - extract basic features
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', 
                      input_shape=input_shape, kernel_regularizer=l2(0.0005)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                      kernel_regularizer=l2(0.0005)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        
        # Second convolutional block - extract complex features
        layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                      kernel_regularizer=l2(0.0005)),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                      kernel_regularizer=l2(0.0005)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
        # Third convolutional block - high-level features
        layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                      kernel_regularizer=l2(0.0005)),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                      kernel_regularizer=l2(0.0005)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.4),
        
        # Flatten and fully connected layers
        layers.Flatten(),
        
        # First dense block
        layers.Dense(512, activation='relu', kernel_regularizer=l2(0.0005)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        # Second dense block
        layers.Dense(256, activation='relu', kernel_regularizer=l2(0.0005)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model