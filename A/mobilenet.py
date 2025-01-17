import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, DepthwiseConv2D, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras import backend as K

# Define a depthwise separable convolution block
def depthwise_separable_conv(x, filters, stride=1):
    # Depthwise convolution
    x = DepthwiseConv2D((3, 3), strides=stride, padding='same', depthwise_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Pointwise convolution
    x = Conv2D(filters, (1, 1), strides=1, padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    return x

# Define MobileNet model
def mobilenet(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    # Initial Conv Layer
    x = Conv2D(32, (3, 3), strides=2, padding='same', kernel_initializer='he_normal')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # MobileNet blocks
    x = depthwise_separable_conv(x, 64, stride=1)

    x = depthwise_separable_conv(x, 128, stride=2)
    x = depthwise_separable_conv(x, 128, stride=1)

    x = depthwise_separable_conv(x, 256, stride=2)
    x = depthwise_separable_conv(x, 256, stride=1)

    x = depthwise_separable_conv(x, 512, stride=2)
    for _ in range(5):
        x = depthwise_separable_conv(x, 512, stride=1)

    x = depthwise_separable_conv(x, 1024, stride=2)
    x = depthwise_separable_conv(x, 1024, stride=1)

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

# Train and Evaluate MobileNet
def train_and_evaluate_mobilenet(model, is_binary, x_train, y_train, x_val, y_val, x_test, y_test, batch_size=128, epochs=100):
    if is_binary:
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

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(x_test, y_test)

    return history, test_loss, test_accuracy

# Use MobileNet
def use_mobilenet(x_train, y_train, x_val, y_val, x_test, y_test):
    input_shape = x_train.shape[1:]
    num_classes = y_train.shape[-1]

    print(f"Number of classes: {num_classes}")

    model = mobilenet(input_shape, num_classes)

    is_binary = False
    if num_classes == 1:
        is_binary = True

    history, test_loss, test_accuracy = train_and_evaluate_mobilenet(model, is_binary, x_train, y_train, x_val, y_val, x_test, y_test, batch_size=128, epochs=100)

    return history, test_loss, test_accuracy