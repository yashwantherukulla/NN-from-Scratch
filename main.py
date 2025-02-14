from sklearn.datasets import fetch_openml
import numpy as np
import matplotlib.pyplot as plt
from DNN import DNN
import time

def fetch_mnist():
    print("Fetching MNIST dataset...")
    start_time = time.time()
    mnist = fetch_openml('mnist_784', version=1, cache=True, as_frame=False)
    X, y = mnist.data, mnist.target
    print(f"Dataset fetched in {time.time() - start_time:.2f} seconds")
    return X, y

def preprocess(X: np.ndarray, y: np.ndarray):
    print("Preprocessing data...")
    start_time = time.time()
    
    # Normalizing the data from (0,255) to (0,1)
    X = X / 255
    print("Data normalized")
    
    # Converting the labels from str -> int
    y = y.astype(np.int32)
    
    y_onehot = np.zeros((len(y), 10))
    y_onehot[np.arange(len(y)), y] = 1
    y = y_onehot
    
    print(f"Preprocessing completed in {time.time() - start_time:.2f} seconds")
    return X, y

def show_graphs(stats: dict):
    train_losses = stats["train_loss"]
    train_accs = stats["train_acc"]
    test_losses = stats["test_loss"]
    test_accs = stats["test_acc"]
    
    epochs = range(1, len(train_losses) + 1)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training and Testing Metrics', fontsize=16)
    
    # Plot 1: Training metrics
    ax1.plot(epochs, train_losses, 'b-', label='Loss')
    ax1.set_ylabel('Loss', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    
    ax1_twin = ax1.twinx()
    ax1_twin.plot(epochs, train_accs, 'r-', label='Accuracy')
    ax1_twin.set_ylabel('Accuracy', color='r')
    ax1_twin.tick_params(axis='y', labelcolor='r')
    ax1.set_title('Training Metrics per Epoch')
    ax1.set_xlabel('Epoch')
    
    # Plot 2: Testing metrics
    ax2.plot(epochs, test_losses, 'b-', label='Loss')
    ax2.set_ylabel('Loss', color='b')
    ax2.tick_params(axis='y', labelcolor='b')
    
    ax2_twin = ax2.twinx()
    ax2_twin.plot(epochs, test_accs, 'r-', label='Accuracy')
    ax2_twin.set_ylabel('Accuracy', color='r')
    ax2_twin.tick_params(axis='y', labelcolor='r')
    ax2.set_title('Testing Metrics per Epoch')
    ax2.set_xlabel('Epoch')
    
    # Plot 3: Loss comparison
    ax3.plot(epochs, train_losses, 'b-', label='Train Loss')
    ax3.plot(epochs, test_losses, 'r-', label='Test Loss')
    ax3.set_title('Training vs Testing Loss')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss')
    ax3.legend()
    
    # Plot 4: Accuracy comparison
    ax4.plot(epochs, train_accs, 'b-', label='Train Accuracy')
    ax4.plot(epochs, test_accs, 'r-', label='Test Accuracy')
    ax4.set_title('Training vs Testing Accuracy')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Accuracy')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('results.png')
    plt.close()


def main():
    print("\nStarting Neural Network training process")
    
    X, y = fetch_mnist()
    X, y = preprocess(X, y)

    train_size = 60000  # total is 70000
    test_size = y.shape[0] - train_size
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    print(f"Train set size: {train_size}, Test set size: {test_size}")

    print("Shuffling training data...")
    shuffle_index = np.random.permutation(np.arange(train_size))
    X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

    print("Initializing Neural Network...")
    dnn = DNN(sizes=[784, 64, 10], activation='sigmoid')
    
    print("Starting training...")
    training_start = time.time()
    stats = dnn.train(X_train, y_train, X_test, y_test, epochs=10, batch_size=128, lr=0.01)
    training_time = time.time() - training_start
    show_graphs(stats)
    
    print(f"Training completed in {training_time:.2f} seconds")

if __name__ == "__main__":
    main()