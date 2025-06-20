from Layer import layer
from ReLu import relu
from SoftMax import softmax
import numpy as np
import pandas as pd

dataset = pd.read_csv('train.csv').to_numpy()

X = dataset[:, 1:].T
X = X/255.0
y = dataset[:, 0]

network = [
    layer(784, 128),
    relu(),
    layer(128, 10),
    softmax()
]

Weights = np.load("BestModelWeights.npz")
network[0].weights = Weights["W1"]
network[0].bias = Weights["b1"]
network[2].weights = Weights["W2"]
network[2].bias = Weights["b2"]

output = X
for n in network:
    output = n.forward(output)

loss = network[3].crossEnt(output, y)
predictions = np.argmax(output, axis=0)
acc = np.mean(predictions == y)

print("The current loss: ", loss)
print(f"Accuracy: {acc:.4f}")

#next time we will use this model to do something fun :)