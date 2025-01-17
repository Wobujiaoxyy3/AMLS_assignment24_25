import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Define baseline model A
def baseline_model_a(input_shape):
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
        Dense(1, activation='sigmoid')
    ])

    return model

# Train and Evaluate baseline model A
def train_and_evaluate_baseline_a(model, x_train, y_train, x_val, y_val, x_test, y_test, batch_size=16, epochs=50):
    #compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    # set early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # train the model
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping]
    )

    # get the test result
    test_loss, test_accuracy = model.evaluate(x_test, y_test)

    return history, test_loss, test_accuracy


def use_baseline_a(x_train, y_train, x_val, y_val, x_test, y_test):
    input_shape = x_train.shape[1:]
    model = baseline_model_a(input_shape)

    history, test_loss, test_accuracy = train_and_evaluate_baseline_a(model, x_train, y_train, x_val, y_val, x_test, y_test, batch_size=128, epochs=100)

    return history, test_loss, test_accuracy
