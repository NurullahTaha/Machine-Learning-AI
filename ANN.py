import numpy as np

class BaseLayer():
    def forward():
        pass

    def backward():
        pass

    def update():
        pass

class HiddenLayer(BaseLayer):
    def __init__(self, num_neurons, num_inputs):
        self.weights = 0.01 * np.random.randn(num_inputs, num_neurons)

class Network(BaseLayer):
    def __init__(self, num_inputs):
        self.layers = []
        self.inputs = num_inputs
        self.last_output = num_inputs

    def addHiddenLayer(self, num_neurons):
        self.layers.append(HiddenLayer(self.last_output, num_neurons))

