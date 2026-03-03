# cifar10-architecture-experiments
build classification model using Convolutional Neural Network from Scratch

## Overview

This project implements a custom Convolutional Neural Network (CNN) trained on the CIFAR-10 dataset. The model was built and optimized through multiple experiments, including architecture tuning, learning rate scheduling, early stopping, and regularization.

## The final model achieved:
- 92% test accuracy
- Balanced precision, recall, and F1-score across all classes
- Stable training with minimal overfitting

## Dataset

CIFAR-10 consists of 60,000 32x32 color images across 10 classes:
- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck
- The dataset is balanced, with 6,000 images per class.

## Model Architecture

The final architecture includes:
- Multiple convolutional blocks
- Double convolution layers
- ReLU activation
- Max pooling
- Fully connected classifier
- Learning rate scheduler
- Early stopping
The model was trained from scratch using PyTorch.

## Results

Test Performance:
- Accuracy: 92%
- Macro F1-score: 0.92
- Consistent performance across all classes
- Mild overfitting (train-test gap ~2–3%)
The confusion matrix shows strong diagonal dominance, indicating correct classification across most categories.

## Training Strategy

Key techniques used:
- Architecture scaling (32 → 64 base filters)
- Learning rate scheduling
- Early stopping
- Hyperparameter tuning
- Performance evaluation with classification report

## Future Work

- Transfer learning comparison (e.g., ResNet18)
- Fine-tuning pretrained models
- Data augmentation improvements
- Model compression for deployment

## Tech Stack

- Python
- PyTorch
- Torchvision
- Gradio
- NumPy
