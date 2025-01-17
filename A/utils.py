import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


def load_and_preprocess_data(dataset_path, normalization=True, add_channel=True, if_categorize=False):
    data = np.load(dataset_path)

    x_train = data['train_images']
    y_train = data['train_labels']
    x_val = data['val_images']
    y_val = data['val_labels']
    x_test = data['test_images']
    y_test = data['test_labels']
    
    # normalization
    if normalization == True:
        x_train = x_train / 255.0
        x_val = x_val / 255.0
        x_test = x_test / 255.0
    
    # add channel for gray images
    if add_channel == True:
        x_train = x_train[..., np.newaxis]
        x_val = x_val[..., np.newaxis]
        x_test = x_test[..., np.newaxis]
    
    if if_categorize == True:
        y_train = to_categorical(y_train, num_classes=8)
        y_val = to_categorical(y_val, num_classes=8)
        y_test = to_categorical(y_test, num_classes=8)
    
    return x_train, y_train, x_val, y_val, x_test, y_test

def data_augmentation(x_train, y_train, num_augmented):

    # datagen params
    datagen = ImageDataGenerator(
    rotation_range=50,
    width_shift_range=0.2,
    height_shift_range=0.2,
    vertical_flip=True
    )

    # store the augmented data
    augmented_images = []
    augmented_labels = []

    # augmented data generation
    for x, y in datagen.flow(x_train, y_train, batch_size=1):
        augmented_images.append(x[0])  
        augmented_labels.append(y[0])  

        if len(augmented_images) >= len(x_train) * num_augmented:
            break

    x_augmented = np.concatenate((x_train, np.array(augmented_images)))
    y_augmented = np.concatenate((y_train, np.array(augmented_labels)))

    return x_augmented, y_augmented


def plot_metrics(history, metrics, titles):
    # creater a big frame
    plt.figure(figsize=(12, 5)) 

    for i, metric in enumerate(metrics):
        plt.subplot(1, len(metrics), i + 1)
        plt.plot(history.history[metric], label=f'Train {metric.capitalize()}')
        plt.plot(history.history[f'val_{metric}'], label=f'Validation {metric.capitalize()}')
        plt.xlabel('Epochs')
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.title(titles[i])

    plt.show()

