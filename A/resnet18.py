import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, MaxPooling2D, Flatten, Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import LearningRateScheduler

from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, ReLU, Add, GlobalAveragePooling2D, Dense, MaxPooling2D
)
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

# Define the ResNet block
def resnet_block(x, filters, stride=1):
    shortcut = x

    # First convolution layer
    x = Conv2D(filters, (3, 3), strides=stride, padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Second convolution layer
    x = Conv2D(filters, (3, 3), strides=1, padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)

    # Add shortcut connection
    if stride != 1 or K.int_shape(x)[-1] != K.int_shape(shortcut)[-1]:
        shortcut = Conv2D(filters, (1, 1), strides=stride, padding='same', kernel_initializer='he_normal')(shortcut)
        shortcut = BatchNormalization()(shortcut)

    x = Add()([x, shortcut])
    x = ReLU()(x)

    return x

# Define ResNet-18 model
def resnet18(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    # Initial Conv Layer
    x = Conv2D(64, (3, 3), strides=1, padding='same', kernel_initializer='he_normal')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Residual blocks
    x = resnet_block(x, 64)
    x = resnet_block(x, 64)

    x = resnet_block(x, 128, stride=2)
    x = resnet_block(x, 128)

    x = resnet_block(x, 256, stride=2)
    x = resnet_block(x, 256)

    x = resnet_block(x, 512, stride=2)
    x = resnet_block(x, 512)

    # Global Average Pooling + Dense Output
    x = GlobalAveragePooling2D()(x)

    if num_classes == 1:
        outputs = Dense(1, activation='sigmoid', kernel_initializer='he_normal')(x)
    else:
        outputs = Dense(num_classes, activation='softmax', kernel_initializer='he_normal')(x)

    model = Model(inputs, outputs)
    return model

# Define learning rate schedule
def lr_schedule(epoch, lr):
    if epoch < 50:
        return 0.001
    elif epoch < 75:
        return 0.0001
    else:
        return 0.00001

# Train and Evaluate ResNet-18
def train_and_evaluate_resnet18(model, is_binary, x_train, y_train, x_val, y_val, x_test, y_test, batch_size=16, epochs=100):
    
    if is_binary == True:
        loss = 'binary_crossentropy'
    else:
        loss = 'categorical_crossentropy'
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=loss,
        metrics=['accuracy']
    )

    lr_scheduler = LearningRateScheduler(lr_schedule)


    # Train the model
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[lr_scheduler]
    )

    # Get the test result
    test_loss, test_accuracy = model.evaluate(x_test, y_test)

    return history, test_loss, test_accuracy

# Use ResNet-18
def use_resnet18(x_train, y_train, x_val, y_val, x_test, y_test):
    input_shape = x_train.shape[1:]
    num_classes = y_train.shape[-1]
    
    print(f"Number of classes: {num_classes}")

    model = resnet18(input_shape, num_classes)

    is_binary = False
    if num_classes == 1:
        is_binary = True

    history, test_loss, test_accuracy = train_and_evaluate_resnet18(model, is_binary, x_train, y_train, x_val, y_val, x_test, y_test, batch_size=16, epochs=100)

    return history, test_loss, test_accuracy