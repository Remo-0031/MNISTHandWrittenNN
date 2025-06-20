import numpy as np


class softmax():
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.exp(inputs - np.max(inputs, axis=0, keepdims=True))
        self.prob = self.output / np.sum(self.output, axis=0, keepdims=True)
        return self.prob

    def crossEnt(self, preds, targets):
        self.preds = preds
        self.targets = targets
        N = preds.shape[1]

        correct_prob = preds[targets, np.arange(N)]

        loss = np.mean(-np.log(correct_prob + 1e-9))
        return loss

    def backward(self, gradient_desc):
        N = self.preds.shape[1]

        oneshot = np.zeros_like(self.preds)
        oneshot[self.targets, np.arange(N)] = 1

        dz = self.preds - oneshot

        return dz