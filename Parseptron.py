class Parseptron:
    def __init__(self, input_size, lr=0.1):
        self.weights = [0.0] * input_size
        self.bias = 0.0
        self.lr = lr
    def activate(self, x):
        weighted_sum = sum(w * xi for w, xi in zip(self.weights, x)) + self.bias
        return weighted_sum
    def predict(self, x):
        return 1 if self.activate(x) >= 0 else 0
    def train(self, x, y):
        pred = self.predict(x)
        error = y - pred

        for i in range(len(self.weights)):
            self.weights[i] += self.lr * error * x[i]
        self.bias += self.lr * error
        return abs(error)