import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from A.utils import load_and_preprocess_data, plot_metrics
from A.baseline_a import use_baseline_a
from B.baseline_b import use_baseline_b
from A.resnet18 import use_resnet18
from A.mobilenet import mobilenet

# Setting up argument parser
parser = argparse.ArgumentParser(description="Run specific models on specific tasks.")
parser.add_argument("--task", choices=['A', 'B'], required=True, help="Task to run (A or B)")
parser.add_argument("--model", choices=['baseline_a', 'resnet18', 'mobilenet', 'baseline_b'], required=True, help="Model to use (baseline_a, resnet18, mobilenet for Task A; baseline_b, resnet18, mobilenet for Task B)")
args = parser.parse_args()

# Choose dataset based on the task
if args.task == 'A':
    dataset_path = 'Datasets/breastmnist.npz'
elif args.task == 'B':
    dataset_path = 'Datasets/bloodmnist.npz'

# Load and preprocess data
x_train, y_train, x_val, y_val, x_test, y_test = load_and_preprocess_data(dataset_path, add_channel=(args.task == 'A'), if_categorize=(args.task == 'B'))

# Select model and run the task
if args.model == 'baseline_a' and args.task == 'A':
    history, test_loss, test_accuracy = use_baseline_a(x_train, y_train, x_val, y_val, x_test, y_test)
elif args.model == 'baseline_b' and args.task == 'B':
    history, test_loss, test_accuracy = use_baseline_b(x_train, y_train, x_val, y_val, x_test, y_test)
elif args.model == 'resnet18':
    history, test_loss, test_accuracy = use_resnet18(x_train, y_train, x_val, y_val, x_test, y_test)
elif args.model == 'mobilenet':
    history, test_loss, test_accuracy = use_mobilenet(x_train, y_train, x_val, y_val, x_test, y_test)
else:
    raise ValueError("Invalid model/task combination.")

# Output results
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Plot the result
metrics = ['accuracy', 'loss']
titles = [f'Training and Validation Accuracy of Task {args.task}', f'Training and Validation Loss of Task {args.task}']
plot_metrics(history, metrics, titles)
