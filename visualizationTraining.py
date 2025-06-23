import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

training = pd.read_csv("Visualization_Of_Training")
print(training.columns)
plt.plot(training['x'],training['y'])
plt.xlabel("Num of Epochs")
plt.ylabel("Accuracy")
plt.title("Visualization of Epochs vs Accuracy")
plt.show()