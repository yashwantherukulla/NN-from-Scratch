# Deep Neural Network (DNN) from Scratch

## Overview

This project implements a fully connected deep neural network (DNN) from scratch using NumPy. The model is trained and evaluated on the MNIST dataset to classify handwritten digits (0-9). The implementation includes forward propagation, backpropagation, weight optimization, and model evaluation metrics.

## Features

- Fully connected deep neural network (DNN) implemented using NumPy
- Customizable architecture with user-defined layer sizes
- Choice of activation functions (ReLU, Sigmoid, Softmax)
- He initialization for weight parameters
- Forward and backward propagation with gradient descent optimization
- Cross-entropy loss function
- Performance evaluation using accuracy, confusion matrix, and classification report
- Training visualization with loss and accuracy graphs

## File Structure

```
model_code/
│── DNN.py            # Implementation of the DNN model
│── main.py           # Main script to train and evaluate the model
│── utils.py          # Helper functions for data fetching, preprocessing, and visualization
│── results/          # Directory to store evaluation outputs (graphs, images, reports)
```

## Dependencies

Ensure you have the following dependencies installed before running the project:

```bash
pip install numpy matplotlib seaborn scikit-learn
```

## Dataset

This model is trained on the MNIST dataset, which consists of 70,000 grayscale images of handwritten digits (28x28 pixels). The dataset is fetched using `fetch_openml` from `scikit-learn`.

## Usage

### 1. Fetch and Preprocess Data

The dataset is loaded and preprocessed using the `fetch_mnist()` and `preprocess()` functions in `utils.py`.

```python
from utils import fetch_mnist, preprocess
X, y = fetch_mnist()
X, y = preprocess(X, y)
```

### 2. Initialize and Train the Model

```python
from DNN import DNN
model = DNN(sizes=[784, 64, 10], activation='relu') # here, the number of hidden layers and their dimensions can be changed
stats = model.train(X_train, y_train, X_test, y_test, epochs=10, batch_size=128, lr=0.01)
```

### 3. Test a Sample Image

```python
idx = np.random.randint(0, len(X_test))
img = X_test[idx]
actual_label = np.argmax(y_test[idx])
model.test(img, actual_label)
```

### 4. Evaluate Model Performance

```python
eval_results = model.eval(X_train, y_train, X_test, y_test, stats=stats)
print(eval_results)
```

### 5. Run the Full Pipeline

You can simply run `main.py` to execute the entire training and evaluation pipeline:

```bash
python model_code/main.py
```

## Results

During training, accuracy and loss metrics are recorded. The final evaluation includes:

- Accuracy and loss plots for training and testing
- Confusion matrix visualization
- Precision, recall, and F1-score
- Sample predictions

## Model Architecture

The model consists of:

- An input layer (784 neurons for 28x28 pixel images)
- One hidden layer with 64 neurons (ReLU activation)
- An output layer with 10 neurons (Softmax activation for classification)

## Future Improvements

- Implement batch normalization for faster convergence
- Add support for different weight initialization strategies
- Experiment with additional hidden layers for deeper architectures
- Extend the model to support different datasets
