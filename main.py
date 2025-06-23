from Layer import layer
from ReLu import relu
from SoftMax import softmax
import numpy as np
import pandas as pd

dataset = pd.read_csv('train.csv').to_numpy()

X = dataset[:, 1:].T
X = X / 255.0
y = dataset[:, 0]

network = [
    layer(784, 128),
    relu(),
    layer(128, 10),
    softmax()
]

Epoch_arr = []
acc_arr = []
Num_Epochs = 2000

# Main training code ish
for i in range(Num_Epochs):
    output = X
    for n in network:
        output = n.forward(output)

    loss = network[3].crossEnt(output, y)
    predictions = np.argmax(output, axis=0)
    acc = np.mean(predictions == y)

    if i % 5 == 0:
        Epoch_arr.append(i)
        acc_arr.append(acc)
        print("This is epoch number: ", i)
        print("The current loss: ", loss)
        print(f"Accuracy: {acc:.4f}")

    grad = output

    for n_rev in reversed(network):
        grad = n_rev.backward(grad)

    network[0].updateWeights()
    network[2].updateWeights()

# Create a dataframe using the epochs as X and Accuracy as the Y that will be used for the visualization Later
accuracy_per_5_epoch = pd.DataFrame({'x': Epoch_arr, 'y': acc_arr})
print(accuracy_per_5_epoch)
accuracy_per_5_epoch.to_csv("Visualization_Of_Training")

# Save the weights acquired from training the model
np.savez("BestModelWeights.npz", W1=network[0].weights,
         b1=network[0].bias,
         W2=network[2].weights,
         b2=network[2].bias)

"""
# making the first layer (input 784, output 128)
inputLayer = layer(784,128)
z1 = inputLayer.forward(X)

# making the activation RELU
rLayer = relu()
a1 = rLayer.forward(z1)

# making the hidden layer (input 128 output 10)
hiddenLayer = layer(128,10)
z2 = hiddenLayer.forward(a1)

# making the softmax activation
sLayer = softmax()
a2 = sLayer.forward(z2)

# calculate the loss
loss = sLayer.crossEnt(a2,y)


# backward output layer
dz = sLayer.backward(a2,y)

hz = hiddenLayer.backward(dz)
rz = rLayer.backward(hz)
iz = inputLayer.backward(rz)

hiddenLayer.updateWeights()
inputLayer.updateWeights()
"""
