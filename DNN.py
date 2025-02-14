import numpy as np
import time

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



