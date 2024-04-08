# Domain Adaptation from SVHN to MNIST

This project implements domain adaptation from the Street View House Numbers (SVHN) dataset to the Modified National Institute of Standards and Technology (MNIST) dataset using deep learning techniques. Domain adaptation aims to adapt a model trained on a source domain (SVHN) to perform well on a target domain (MNIST) with limited labeled data in the target domain.

## Overview

Domain adaptation is a subfield of transfer learning where the knowledge learned from a source domain is transferred to a related but different target domain. In this project, we use the SVHN dataset, which consists of real-world images of house numbers, and the MNIST dataset, which contains handwritten digit images. The goal is to train a model on the SVHN dataset and adapt it to perform well on the MNIST dataset.

## Components

### Feature Extractor (Generator)

The feature extractor is responsible for extracting features from input images. We use a convolutional neural network (CNN) architecture as the feature extractor, which learns hierarchical representations of the input images.

### Classifiers (Predictors)

We employ two classifiers to predict the digit labels: one for the SVHN domain and another for the MNIST domain. Both classifiers share the same feature extractor but have separate output layers.

### Training Procedure

The training procedure involves training the feature extractor and classifiers on the SVHN dataset. Then, we fine-tune the model on the MNIST dataset using adversarial domain adaptation techniques to align the feature distributions between the source and target domains.

## Usage

To run the code:

1. Clone or download the repository.

2. Install the required dependencies listed in the `requirements.txt` file:
```pip install -r requirements.txt```

3. Run the `main.py` script:
python main.py

4. Monitor the training progress and evaluate the model's performance on the SVHN and MNIST datasets.

## Results

After training the model, you can visualize the accuracy of the model on the SVHN and MNIST datasets over epochs using the provided visualization scripts.

## Citation

If you use this code for your research, please consider citing the following paper:
https://arxiv.org/abs/1712.02560

## License

This project is licensed under the [MIT License](LICENSE.txt).
