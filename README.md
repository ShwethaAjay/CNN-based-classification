## Overview
This project explores different Convolutional Neural Network (CNN) architectures for image classification. The model has been modified through three iterations to improve performance and robustness. These changes include increasing the number of filters, adjusting kernel sizes, and integrating batch normalization and dropout to enhance training stability.

## Architecture Evolution
### Initial Model
The original model is a simple CNN with two convolutional layers, followed by fully connected layers.

- **Conv2D (conv1)**: 6 filters, 5x5 kernel, ReLU activation
- **MaxPool2D (pool1)**: 2x2 pool size, stride 2
- **Conv2D (conv2)**: 16 filters, 5x5 kernel, ReLU activation
- **MaxPool2D (pool2)**: 2x2 pool size, stride 2
- **Flatten layer**
- **Fully Connected Layers (fc1, fc2, fc3)**: 120, 84, and 10 neurons respectively
- **Output Layer**: Softmax activation

### Setup 1: Increasing Filters and Adding Layers
In this version, the number of filters is increased, and an additional convolutional layer is introduced.

- Increased filters: conv1 (6 → 32), conv2 (16 → 64), conv3 (new, 128 filters)
- Kernel sizes: 5x5 for conv1 and conv2, 3x3 for conv3
- **Global Average Pooling** added to reduce dimensions
- Expanded fully connected layers: 512 and 256 neurons

### Setup 2: Modifying Kernel Sizes
This setup reduces kernel sizes to improve feature extraction efficiency.

- Kernel sizes reduced from 5x5 → 3x3 for conv1 and conv2
- **Conv3 keeps a 3x3 kernel**
- **Global Average Pooling** retained
- Fully connected layers remain unchanged

### Setup 3: Adding Batch Normalization & Dropout
This version enhances stability and prevents overfitting.

- **Batch Normalization** added after each convolutional layer
- **Dropout layers** (0.5 probability) added before fully connected layers
- Filters and neuron counts remain unchanged

## Implementation
