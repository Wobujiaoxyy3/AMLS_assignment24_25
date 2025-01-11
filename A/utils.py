import numpy as np
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Loads and preprocesses the dataset.
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
    
    # add channel for graystyle images
    if add_channel == True:
        x_train = x_train[..., np.newaxis]
        x_val = x_val[..., np.newaxis]
        x_test = x_test[..., np.newaxis]
    
    # Convert labels to one-hot encoding
    if if_categorize == True:
        y_train = to_categorical(y_train, num_classes=8)
        y_val = to_categorical(y_val, num_classes=8)
        y_test = to_categorical(y_test, num_classes=8)
    
    return x_train, y_train, x_val, y_val, x_test, y_test

# Plots training and validation curves over epochs.
def plot_metrics(history, metric, title):
    plt.plot(history.history[metric], label=f'Train {metric.capitalize()}')
    plt.plot(history.history[f'val_{metric}'], label=f'Validation {metric.capitalize()}')
    plt.xlabel('Epochs')
    plt.ylabel(metric.capitalize())
    plt.legend()
    plt.title(title)
    plt.show()

