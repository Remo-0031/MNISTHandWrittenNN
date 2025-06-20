import numpy as np


class relu:
    def forward(self,inputs):
        self.inputs = inputs
        self.output = np.maximum(0,inputs)
        return self.output

    def backward(self,output_gradient):
        return output_gradient * (self.inputs > 0)