from Layer import layer
from ReLu import relu
from SoftMax import softmax
import numpy as np
import pandas as pd
from convertToPixel import preProcessScreenShot

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


def inferenceModel(image):
    output = image
    for n in network:
        output = n.forward(output)

    predictions = np.argmax(output, axis=0)
    print(predictions)


def inferScreenShot():
    image = preProcessScreenShot()
    output = image
    for n in network:
        output = n.forward(output)

    predictions = np.argmax(output, axis=0)
    print("The screenshotted Image contains the Number: ", predictions)

# next time we will use this model to do something fun :)
