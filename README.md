Multi-Class Weather Classification Project
This project involves training and evaluating machine learning models for multi-class weather classification using the Multi-class Weather Dataset.

Table of Contents
Overview
Project Structure
Prerequisites
Installation
Usage
Results
Contributing
License
Overview
This project aims to classify weather conditions into four categories: Cloudy, Rain, Shine, and Sunrise, using different machine learning models. The dataset consists of images captured under different weather conditions.

Three models have been implemented and evaluated:

Fully Connected Network (FCNet): A simple feedforward neural network.
Convolutional Neural Network (CNN): A more complex neural network architecture designed for image classification.
Transfer Learning with ResNet-18: Utilizing a pre-trained ResNet-18 model as a feature extractor for image classification.
Project Structure
The project structure is organized as follows:

data/: Contains the Multi-class Weather Dataset.
models/: Includes the implementation of the machine learning models.
utils/: Helper functions and utilities.
train.py: The main script for training the models.
evaluate.py: Evaluation script to assess model performance.
plot_results.py: Script for plotting training and validation results.
Prerequisites
Before running the code, make sure you have the following prerequisites installed:

Python 3.x
PyTorch
NumPy
PIL (Python Imaging Library)
Matplotlib
scikit-learn
tqdm
Jupyter Notebook (for running Jupyter notebooks, if used)
You can install the required packages using the provided requirements.txt file.

Installation
Clone this repository to your local machine:

bash
Copy code
git clone https://github.com/yourusername/multi-class-weather-classification.git
cd multi-class-weather-classification
Create a virtual environment (optional but recommended):

bash
Copy code
python -m venv venv
source venv/bin/activate  # On Windows, use venv\Scripts\activate
Install the required packages:

bash
Copy code
pip install -r requirements.txt
Usage
Training the Models: Run the following command to train the models:

bash
Copy code
python train.py
You can adjust hyperparameters and settings within the train.py script.

Evaluation: To evaluate model performance, run:

bash
Copy code
python evaluate.py
This will provide confusion matrices and various metrics (accuracy, precision, recall, F1-score) for each model.

Plotting Results: To visualize training and validation results, run:

bash
Copy code
python plot_results.py
Results
The project achieved the following results:

FCNet:
Training Accuracy: 96.06%
Validation Accuracy: 94.69%
Test Accuracy: 93.85%
CNN:
Training Accuracy: 95.00%
Validation Accuracy: 92.04%
Test Accuracy: 91.67%
Transfer Learning with ResNet-18:
Training Accuracy: 97.33%
Validation Accuracy: 95.00%
Test Accuracy: 96.46%
Contributing
Contributions to this project are welcome. If you find any issues or have suggestions for improvements, please create a GitHub issue or submit a pull request.

License
This project is licensed under the MIT License - see the LICENSE file for details.

