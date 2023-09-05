```markdown
# Weather Classification with Deep Learning

This repository contains code and instructions for training a deep learning model to classify different weather conditions: Cloudy, Rain, Shine, and Sunrise using PyTorch.

## Table of Contents
- [Requirements](#requirements)
- [Dataset](#dataset)
- [Usage](#usage)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)

## Requirements
- Python 3.x
- PyTorch
- torchvision
- scikit-learn
- PIL (Python Imaging Library)
- matplotlib
- Google Colab (optional, for GPU support)

## Dataset
We use the "Multi-class Weather Dataset" for training and evaluation. You can download it from [this link](https://your-dataset-link-here) and unzip it in the root directory.

The dataset structure should look like this:
```
- Multi-class Weather Dataset/
    - Cloudy/
        - cloudy1.jpg
        - cloudy2.jpg
        - ...
    - Rain/
        - rain1.jpg
        - rain2.jpg
        - ...
    - Shine/
        - shine1.jpg
        - shine2.jpg
        - ...
    - Sunrise/
        - sunrise1.jpg
        - sunrise2.jpg
        - ...
```

## Usage
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/weather-classification.git
   cd weather-classification
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

## Training
You can choose between three different models for training:
- Fully Connected Neural Network (FCNet)
- Convolutional Neural Network (CNN)
- Transfer Learning with ResNet-18

To train the FCNet model, run the following command:
```bash
python train_fcnet.py
```

To train the CNN model, use:
```bash
python train_cnn.py
```

To train the ResNet-18 transfer learning model:
```bash
python train_resnet.py
```

You can adjust hyperparameters inside the respective training scripts.

## Evaluation
To evaluate a trained model, run the following command:
```bash
python evaluate.py --model_path path/to/your/model.pth
```

Replace `path/to/your/model.pth` with the path to the trained model you want to evaluate.

## Results
The evaluation script will display the accuracy, precision, recall, and F1-score for each weather class, as well as a confusion matrix.

Enjoy classifying weather conditions with deep learning!
```

In this simplified README, everything is contained in one file, and each section provides clear instructions on how to set up and use the project. You can add more details and explanations as needed for your specific project.
