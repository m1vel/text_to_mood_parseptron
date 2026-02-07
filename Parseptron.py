import numpy as np
from typing import List, Tuple

class EmotionPerceptron:
    def __init__(self, input_size: int, lr: float = 0.01, momentum: float = 0.9):
        self.input_size = input_size
        self.lr = lr
        self.momentum = momentum
        limit = np.sqrt(6 / (input_size + 1))
        self.weights = np.random.uniform(-limit, limit, input_size)
        self.bias = np.random.uniform(-limit, limit)
        self.weights_velocity = np.zeros(input_size)
        self.bias_velocity = 0.0
        self.loss_history = []
        self.accuracy_history = []
    
    def sigmoid(self, x: float) -> float:
        if x >= 0:
            return 1 / (1 + np.exp(-x))
        else:
            exp_x = np.exp(x)
            return exp_x / (1 + exp_x)
    
    def activate(self, x: np.ndarray) -> float:
        x_array = np.array(x, dtype=np.float32)
        weighted_sum = np.dot(self.weights, x_array) + self.bias
        return self.sigmoid(weighted_sum)
    
    def predict(self, x: np.ndarray, threshold: float = 0.5) -> int:
        return 1 if self.activate(x) >= threshold else 0
    
    def train(self, x: np.ndarray, y: float) -> float:
        x_array = np.array(x, dtype=np.float32)
        pred = self.activate(x_array)
        error = y - pred
        delta = error * pred * (1 - pred)
        weights_grad = delta * x_array
        bias_grad = delta
        self.weights_velocity = (self.momentum * self.weights_velocity + 
                                self.lr * weights_grad)
        self.bias_velocity = (self.momentum * self.bias_velocity + 
                             self.lr * bias_grad)
        self.weights += self.weights_velocity
        self.bias += self.bias_velocity
        loss = (error ** 2) / 2
        self.loss_history.append(loss)
        return abs(error)
    
    def get_important_features(self, dictionary: List[str], top_n: int = 10) -> List[Tuple[str, float]]:
        features = list(zip(dictionary, self.weights))
        sorted_features = sorted(features, key=lambda x: abs(x[1]), reverse=True)
        return sorted_features[:top_n]
    
    def save_weights(self, filepath: str):
        np.savez(filepath, weights=self.weights, bias=self.bias)
    
    def load_weights(self, filepath: str):
        data = np.load(filepath)
        self.weights = data['weights']
        self.bias = data['bias']