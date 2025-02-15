import numpy as np
import time
import matplotlib.pyplot as plt
import os
import seaborn as sns

class DNN():
    def __init__(self, sizes: list, activation='sigmoid'):
        self.sizes = sizes
        self.num_layers = len(sizes)

        if activation == 'sigmoid':
            self.activation = self.sigmoid
        elif activation == 'relu':
            self.activation = self.relu
        else:
            raise ValueError("Activation can either be 'sigmoid' or 'relu'. Given neither")
        
        self.params = self.initialize()
        self.cache = {} # for saving the activations

    def relu(self, x, grad=False):
        if grad:
            x = np.where(x<0, 0, x)
            x = np.where(x>=0, 1, x)
            return x
        return np.maximum(0,x)

    def sigmoid(self, x, grad=False):
        if grad:
            return (np.exp(-x))/((np.exp(-x)+1)**2)
        return 1/(1 + np.exp(-x))
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
        return exp_x / np.sum(exp_x, axis=0, keepdims=True)
    
    def initialize(self):
        params = {}
        for i in range(1, self.num_layers):
            # He init
            scale = np.sqrt(1./self.sizes[i-1])
            params[f"W{i}"] = np.random.randn(self.sizes[i], self.sizes[i-1]) * scale #Return a sample (or samples) from the “standard normal” distribution.
            params[f"b{i}"] = np.zeros((self.sizes[i], 1))

        return params
    
    def forward(self, x):
        self.cache["X"] = x
        self.cache["A0"] = x.T

        for i in range(1, self.num_layers):
            self.cache[f"Z{i}"] = self.params[f"W{i}"] @ self.cache[f"A{i-1}"] + self.params[f"b{i}"]
            if (i < self.num_layers-1):
                self.cache[f"A{i}"] = self.activation(self.cache[f"Z{i}"])
            else:
                self.cache[f"A{i}"] = self.softmax(self.cache[f"Z{i}"])
        
        return self.cache[f"A{self.num_layers-1}"]
    
    def backprop(self, y, y_hat):
        batch_size = y.shape[0]
        self.grads = {}

        l = self.num_layers - 1
        dZ = y_hat - y.T
        self.grads[f"W{l}"] = (1./batch_size) * (dZ @ self.cache[f"A{l-1}"].T)
        self.grads[f"b{l}"] = (1./batch_size) * np.sum(dZ, axis=1, keepdims=True)
        dA = self.params[f"W{l}"].T @ dZ

        for i in range(l-1, 0, -1):
            dZ = dA * self.activation(self.cache[f"Z{i}"], grad=True)
            self.grads[f"W{i}"] = (1./batch_size) * (dZ @ self.cache[f"A{i-1}"].T)
            self.grads[f"b{i}"] = (1./batch_size) * np.sum(dZ, axis=1, keepdims=True)
            if i>1:
                dA = self.params[f"W{i}"].T @ dZ
        
        return self.grads
    
    def optimize(self, lr=0.001):
        for key in self.params:
            self.params[key] = self.params[key] - lr*self.grads[key]
    
    def cross_entropy_loss(self, y, y_hat):
        epsilon = 1e-15  # Small constant to prevent log(0)
        y_hat = np.clip(y_hat, epsilon, 1 - epsilon)
        return -(1./y.shape[0])*(np.sum(y.T*np.log(y_hat)))

    def acc(self, y, y_hat):
        return np.mean(np.argmax(y, axis=-1) == np.argmax(y_hat.T, axis=-1))
    
    def train(self, X_train, y_train, X_test, y_test, epochs = 10, batch_size=64, lr=0.01):
        num_batches = -(-X_train.shape[0]//batch_size)

        start_time = time.time()
        template = "Epoch {}: {:.2f}s, train acc={:.2f}, train loss={:.2f}, test acc={:.2f}, test loss={:.2f}"
        stats = {"train_acc" : [],
                 "train_loss" : [],
                 "test_acc" : [],
                 "test_loss" : []}

        for epoch in range(epochs):
            for i in range(num_batches):
                begin = i * batch_size
                end = min(begin + batch_size, X_train.shape[0]-1)
                X = X_train[begin:end]
                y = y_train[begin:end]

                y_hat = self.forward(X)
                self.backprop(y, y_hat)
                self.optimize(lr)

            y_hat = self.forward(X_train)
            train_acc = self.acc(y_train, y_hat)
            train_loss = self.cross_entropy_loss(y_train, y_hat)

            y_hat = self.forward(X_test)
            test_acc = self.acc(y_test, y_hat)
            test_loss = self.cross_entropy_loss(y_test, y_hat)

            print(template.format(epoch+1, time.time()-start_time, train_acc, train_loss, test_acc, test_loss))

            stats["train_acc"].append(train_acc)
            stats["train_loss"].append(train_loss)
            stats["test_acc"].append(test_acc)
            stats["test_loss"].append(test_loss)
        
        return stats
    
    def test(self, img, y=None, save_path='model_code/results/test.png'):
        img_reshaped = img.reshape(1, -1)
        pred_prob = self.forward(img_reshaped)
        y_hat = np.argmax(pred_prob)
        print(pred_prob)
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(img.reshape(28, 28), cmap='gray')
        plt.axis('off')
        plt.title('Input Image')

        plt.subplot(1, 2, 2)
        probabilities = pred_prob.flatten()
        colors = ['lightcoral'] * 10
        colors[y_hat] = 'lightgreen'
        
        plt.bar(range(10), probabilities, color=colors)
        plt.xlabel('Digit')
        plt.ylabel('Probability')
        plt.title(f'Prediction: {y_hat}' + (f'\nActual: {y}' if y is not None else ''))
        plt.xticks(range(10))
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
        
        return y_hat
    
    def eval(self, X_train, y_train, X_test, y_test, stats=None, save_path='model_code/results/evals.png'):
        sns.set_style("darkgrid")
        fig = plt.figure(figsize=(20, 15))

        # 1. Training History Plot
        if stats:
            plt.subplot(3, 2, 1)
            plt.plot(stats['train_acc'], label='Train Accuracy')
            plt.plot(stats['test_acc'], label='Test Accuracy')
            plt.title('Training History - Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            
            plt.subplot(3, 2, 2)
            plt.plot(stats['train_loss'], label='Train Loss')
            plt.plot(stats['test_loss'], label='Test Loss')
            plt.title('Training History - Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()

        # 2. Confusion Matrix
        y_pred_test = np.argmax(self.forward(X_test), axis=0)
        y_true_test = np.argmax(y_test, axis=1)
        
        conf_matrix = np.zeros((10, 10))
        for i in range(len(y_true_test)):
            conf_matrix[y_true_test[i]][y_pred_test[i]] += 1

        plt.subplot(3, 2, 3)
        sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        
        # 3. Sample Predictions (now limited to 3 samples)
        n_samples = 3  # Changed from 5 to 3
        random_indices = np.random.randint(0, len(X_test), n_samples)
        
        for idx, i in enumerate(random_indices):
            plt.subplot(3, 2, 4 + idx)
            img = X_test[i].reshape(28, 28)
            pred_prob = self.forward(X_test[i].reshape(1, -1))
            pred = np.argmax(pred_prob)
            true = np.argmax(y_test[i])
            
            plt.imshow(img, cmap='gray')
            plt.axis('off')
            plt.title(f'Pred: {pred}, True: {true}')
        
        # 4. Add text information
        test_acc = self.acc(y_test, self.forward(X_test))
        train_acc = self.acc(y_train, self.forward(X_train))
        test_loss = self.cross_entropy_loss(y_test, self.forward(X_test))
        
        info_text = (
            f'Model Architecture: {self.sizes}\n'
            f'Final Train Accuracy: {train_acc:.4f}\n'
            f'Final Test Accuracy: {test_acc:.4f}\n'
            f'Final Test Loss: {test_loss:.4f}\n'
        )
        
        fig.text(0.02, 0.02, info_text, fontsize=10, family='monospace')
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        return {
            'test_accuracy': test_acc,
            'train_accuracy': train_acc,
            'test_loss': test_loss,
            'test_acc': test_acc
        }