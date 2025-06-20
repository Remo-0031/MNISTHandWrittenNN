import numpy as np


class layer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size) * np.sqrt(2.0/input_size)
        self.bias = np.zeros((output_size, 1))

    def forward(self, inputs):
        self.input = inputs
        self.output = np.dot(self.weights, inputs) + self.bias
        return self.output

    def updateWeights(self):
        learning_Rate = 0.005
        self.weights -= learning_Rate * self.dW
        self.bias -= learning_Rate * self.db

    def backward(self,output_Gradient):
        N = self.input.shape[1]

        self.dW = np.dot(output_Gradient, self.input.T) / N
        self.db = np.sum(output_Gradient, axis=1, keepdims=True) / N
        self.dA = np.dot(self.weights.T, output_Gradient)
        return self.dA
