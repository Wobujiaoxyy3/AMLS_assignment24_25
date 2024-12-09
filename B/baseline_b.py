from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Define baseline model of task B
def baseline_model_b(input_shape):
    model = Sequential([
        # First layer: Conv + Maxpooling
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),

        # Second layer: Conv + Maxpooling
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),

        # Flatten + FullyConnection
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),  # Dropout

        # Output
        Dense(8, activation='softmax')  
    ])
    return model