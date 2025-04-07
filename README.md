# Sound Classification on ESC-50 Dataset

This repository contains code and experiments for sound classification on the ESC-50 dataset. The goal is to classify audio recordings into one of 50 environmental sound classes.

## Dataset

The **ESC-50** dataset consists of 50 environmental sound classes, with each class containing 40 recordings, resulting in a total of 2,000 sound recordings. These classes include sounds such as dog barks, gunshots, and various natural sounds.

- **Dataset link:** [ESC-50 Dataset](https://github.com/karoldvl/ESC-50)

### Random Guessing Accuracy

With 50 classes, the expected random guessing accuracy is approximately **2%** (1/50), assuming each class is equally likely to occur.

## Model Overview

The network used for this sound classification task is a **Multilayer Perceptron (MLP)**, which processes **spectrograms** extracted from the audio files. The model is evaluated using **5-fold cross-validation** based on the predefined splits from the ESC-50 dataset.

### Training Process
- The data is split into 5 folds, ensuring that the training and testing data are split evenly across the different classes.
- For each fold, an MLP model is trained on the training data and tested on the hold-out validation data.
- The model architecture is designed to take in spectrograms as input features.

## Installation

It is recommended to create a new environment:

```bash
conda create -n challenge2 python=3.10
conda activate challenge2
```

Clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/yourusername/Challenge2_2025.git
cd Challenge2_2025
pip install torch==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

# Usage
To run the sound classification experiments, use the following command:

```bash
python train_crossval.py
```
This will start the training process using the MLP model and 5-fold cross-validation.

## Results
The model will output the classification results for each fold, including metrics such as accuracy and loss.

To test all cross-validation folds use the following command:

```bash
python test_crossval.py results/EXPERIMENT_DIR
```
