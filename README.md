This project is designed to perform medical image classification tasks on BreastMNIST and BloodMNIST datasets using different machine learning models. The project is structured into specific directories with scripts that build, train, and evaluate various models.

## Project Structure
Dataset folder: Contains the datasets used in the project.
A/: Contains scripts for models and utilities specific to Task A.
B/: Contains scripts for models specific to Task B.
main.py: Script to run the models on specified tasks.

## Dataset Folder
Before running the models, download the required datasets:

BreastMNIST.npz
BloodMNIST.npz
from the MedMNIST official website available at [MedMNIST Datasets](https://zenodo.org/records/10519652). Please place these files in the Dataset folder.

## Directory A
baseline_a.py: This script includes the construction, training, and evaluation of the baseline model for Task A.
resnet18.py: Contains the construction, training, and evaluation of the ResNet18 model for Task A.
mobilenet.py: Implements the construction, training, and evaluation of the MobileNet model for Task A.
utils.py: Includes utility functions such as load_and_preprocess_data which are essential for data handling and preprocessing.

## Directory B
baseline_b.py: Contains the construction, training, and evaluation of the baseline model for Task B.

## main.py
This script allows you to run specific models on specific tasks. It accepts command-line arguments to specify the task and the model.

## How to Run
Ensure that Python 3 and required packages (numpy, tensorflow) are installed.
Navigate to the project directory in the command line.
Use the following command format to run the models:
``` bash
python main.py --task [A/B] --model [model_name]
```

Where:

[A/B] should be replaced with the task you want to run (A for BreastMNIST, B for BloodMNIST).
[model_name] should be one of baseline_a, resnet18, mobilenet for Task A and baseline_b, resnet18, mobilenet for Task B.
Example Command
To run the ResNet18 model on Task A, use the following command:

``` bash
python main.py --task A --model resnet18
```
This command initializes the training and evaluation of the ResNet18 model on the BreastMNIST dataset.

Requirements:
Python 3.9 or higher
Matplotlib
Numpy
Tensorflow

**Important**: Ensure compatibility among these packages to avoid conflicts during installation or runtime.
