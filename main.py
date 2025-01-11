import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from A.utils import load_and_preprocess_data
from A.visualization import 
from A.baseline_a import use_baseline_a
from B.baseline_b import use_baseline_b


# implementation of task A
# load breastmnist dataset
breastmnist_dataset_path = 'Datasets/breastmnist.npz'

# load and preprocess data
x_train, y_train, x_val, y_val, x_test, y_test = load_and_preprocess_data(breastmnist_dataset_path)

# call the baseline model A to solve the problem
history, test_loss, test_accuracy = use_baseline_a(x_train, y_train, x_val, y_val, x_test, y_test)

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# plot the result
plot_metrics(history, 'accuracy', 'Training and Validation Accuracy of A')
plot_metrics(history, 'loss', 'Training and Validation Loss of A')



# implementation of task B
# load bloodmnist dataset
bloodmnist_dataset_path = 'Datasets/bloodmnist.npz'

x_train, y_train, x_val, y_val, x_test, y_test = load_and_preprocess_data(bloodmnist_dataset_path, add_channel=False, if_categorize=True)

# call the baseline model B to solve the problem
history, test_loss, test_accuracy = use_baseline_b(x_train, y_train, x_val, y_val, x_test, y_test)

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# plot the result
plot_metrics(history, 'accuracy', 'Training and Validation Accuracy of B')
plot_metrics(history, 'loss', 'Training and Validation Loss of B')

