import numpy as np
import mnist
import scipy.misc
import matplotlib.pyplot as plt
import pickle
import os

'''
class BaseLayer:
    def forward(self, input_data):
        pass

    def getOutSize(self):
        pass

class HiddenLayer(BaseLayer):
    def __init__(self, n_inputs, n_neurons):
        self.out_size = n_neurons
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, input_data):
        self.output = np.dot(input_data, self.weights) + self.biases
        return self.output   

    def getOutSize(self):
        return self.out_size   
    

class Activation(BaseLayer):
    def __init__(self, n_inputs):
        self.n_inputs = n_inputs

    def forward(self, n_inputs):
        return 1 / (1 + np.exp(-n_inputs))
    
    def getOutSize(self):
        return self.n_inputs 
    

class Network:
    def __init__(self, num_inputs):
        self.layers = []
        self.num_inputs = num_inputs
        self.last_outsize = num_inputs

    def addHiddenLayer(self, n_neurons):
        if len(self.layers) == 0:
            self.layer.append(HiddenLayer(self.layers.getOutSize(), n_neurons))
        else:
            self.layer.append(HiddenLayer(self.layers[-1].getOutSize(), n_neurons))
        self.last_outsize = n_neurons
                          
    def addActivation(self, n_inputs):
        self.layer.append(Activation(self.last_outsize, n_inputs))

    def forward(self, input_data):
        current_data = input_data

        for layer in self.layers:
            current_data = layer.forward(current_data)

        return current_data 


net = Network(5)

net.addHiddenLayer(3)


'''



class BaseLayer():
    def forward(self, n_inputs):
        pass
    
    def backward(self, out_error):
        pass

    def update(self, learning_rate):
        pass


class Layer_Dense(BaseLayer):
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(2. / n_inputs)
        self.biases = np.zeros((1, n_neurons))
        pass

    def forward(self, n_inputs):
        self.input = n_inputs
        self.output = np.dot(self.input, self.weights) + self.biases
        return self.output
    
    def backward(self, out_error):
        # dW. check dims and multiplication
        self.dW = np.dot(self.input.transpose(), out_error)

        # dB check dims
        self.dB = np.sum(out_error, axis=0, keepdims=True)

        # dE. -- we need to figure this out
        dE = np.dot(out_error, self.weights.transpose())


        return dE
    
    def update(self, learning_rate):
        self.weights = self.weights - learning_rate * self.dW
        self.biases = self.biases - learning_rate * self.dB


    
class Activation(BaseLayer):
    def forward(self, n_inputs):
        self.n_inputs = n_inputs
        return 1 / (1 + np.exp(-self.n_inputs))
    
    def backward(self, out_error):
        # Backpropagte the error
        self.out_error = out_error

        self.init_sig = 1 / (1 + np.exp(-self.n_inputs))
        self.der_sig = 1 - (1 / (1 + np.exp(-self.n_inputs)))
        self.final = self.init_sig * self.der_sig

        return self.final * out_error
        pass

    def update(self, learning_rate):
        pass

class Activation_ReLu(BaseLayer):
    def forward(self, n_inputs):
        self.n_inputs = n_inputs
        return np.maximum(0, self.n_inputs)

    def backward(self, out_error):
        c_input = out_error.copy()

        c_input[self.n_inputs <= 0] = 0

        return c_input
    
    def update(self, learning_rate):
        pass


    
class Network:
    def __init__(self):
        self.layers = []
    
    def add(self, layer):
        self.layers.append(layer)

    def forward(self, input_data):
        current_data = input_data

        for layer in self.layers:
            current_data = layer.forward(current_data)

        self.net_out = current_data

        return current_data 
    

    def backward(self, targets):
        
        loss = self.net_out - targets

        # Reverse iterate layers
        for layer in self.layers[::-1]:
            loss = layer.backward(loss) 

    def update(self):
        for layer in self.layers:
            layer.update(0.01)

    

print("Loading Data...")
mnist.datasets_url = 'https://storage.googleapis.com/cvdf-datasets/mnist/'

train_images = mnist.train_images() 
train_labels = mnist.train_labels()

test_images = mnist.test_images()
test_labels = mnist.test_labels()

X_train = train_images.reshape((-1, 784)) / 255.0
X_test = test_images.reshape((-1, 784)) / 255.0


def one_hot(labels):
    output = np.zeros((len(labels), 10))
    for i in range(len(labels)):
        output[i, labels[i]] = 1
    return output

y_test = one_hot(test_labels)

y_train = one_hot(train_labels)

# LIMIT DATA (2000 images for speed)
LIMIT = 60000
split = int(LIMIT * 0.8)
#val_end = int(LIMIT * 0.9)

X_temp = X_train[ : LIMIT]
y_temp = y_train[ : LIMIT]

X_val = X_temp[split : ]
y_val = y_temp[split : ]

#X_final = X_temp[val_end : LIMIT]
#y_final = y_temp[val_end : LIMIT]

X_train = X_temp[ : split]
y_train = y_temp[ : split]


print(f"Data Split: Train={len(X_train)}, Val={len(X_val)}, Final={len(X_test)}")



# 3. SMART NETWORK SETUP (Load if exists, else create) 

if os.path.exists('my_brain.pkl'):
    print("FOUND SAVED BRAIN! Resuming training...")
    
    with open('my_brain.pkl', 'rb') as f:
        net = pickle.load(f)

else:
    print("✨ NO SAVE FOUND. Creating new random brain...")
    
    # Create from scratch
    net = Network()
    net.add(Layer_Dense(784, 64))   
    net.add(Activation_ReLu())
    net.add(Layer_Dense(64, 64))
    net.add(Activation()) 
    net.add(Layer_Dense(64, 64))
    net.add(Activation_ReLu())
    net.add(Layer_Dense(64, 10))
    net.add(Activation())

# 4. TRAIN

print("Training started...")

for epoch in range(50): 
    error_sum = 0
    for i in range(len(X_train)):
        img = X_train[i:i+1] 
        target = y_train[i:i+1]
    
        net.forward(img)
        net.backward(target)
        net.update()
        
        error_sum += np.mean(np.square(target - net.net_out))

    #Validation

    validation = X_val[ : ]

    val_out = net.forward(validation)
    
    prediction = np.argmax(val_out, axis=1)
    label = np.argmax(y_val, axis=1)

    accuracy = np.mean(prediction == label)
    
    
    print(f"Epoch {epoch}, Loss: {error_sum/len(X_train):.4f}, Val Acc: {accuracy*100:.2f}%")

#Test

print(f"Study Material: {len(X_train)} images")
print(f"Final Exam Size: {len(test_images)} images")

final_test = net.forward(X_test)

prediction = np.argmax(final_test, axis=1)
label = np.argmax(y_test, axis=1)

accuracy = np.mean(prediction == label)

print(f"\n-----------------------------------------------------")
print(f"FINAL REPORT (Official MNIST Benchmark)")
print(f"-----------------------------------------------------")
print(f"Official Test Accuracy: {accuracy*100:.2f}%")


print("\n--- SAVING MODEL ---")

with open('my_brain.pkl', 'wb') as f:
    pickle.dump(net, f)

print("Brain saved to 'my_brain.pkl'! You can now load this later.")




# -----------------------------------------------------
# 5. VISUAL TEST (Exact Match to Screenshot) 🖼️
# -----------------------------------------------------
print("\n--- VISUAL TEST ---")

# 1. Pick a random test image
idx = np.random.randint(0, len(X_test))
sample = X_test[idx:idx+1]
true_label = test_labels[idx]

# 2. Get Prediction
probs = net.forward(sample)
prediction = np.argmax(probs)

# 3. Print Output (Matching your screenshot style)
# We use np.set_printoptions to force the scientific notation look
np.set_printoptions(suppress=False, precision=8, linewidth=200)

# 3. Print Output (The new Percentage Loop)
print(f"\nTrue Label: {true_label}")
print("Network's Confidence:")

for i in range(10):
    # Convert scientific number to percentage
    p = probs[0][i] * 100
    
    # Print with 8 decimal places
    print(f"  Digit {i}: {p:.8f}%")

print(f"\n> Final Prediction: {prediction}")

# 4. Show the Image
image_grid = sample.reshape(28, 28)
plt.imshow(image_grid, cmap='gray')
plt.title(f"True: {true_label} | Pred: {prediction}")
plt.show()







'''
class network:
    def __init__(self):
        self.layers = []
        self.current_input_size = 0

    def setInputSize(self, n):
        self.current_input_size = n
        pass

    def addHiddenFullyConnected(self, n_neurons):
        rows = self.current_input_size
        cols = n_neurons

        weights = np.random.randn(rows, cols)

        biases = np.zeros((1, n_neurons))

        self.layers.append((weights, biases))
        self.current_input_size = n_neurons

        pass
        

    def setOutputSize(self, n):
        self.addHiddenFullyConnected(n)

    def forward(self, input_data):
        final_data = input_data
        for weights, biases in self.layers:
            z = np.dot(final_data, weights) + biases

            final_data = self.sigmoid(z)

        return final_data
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

my_net = network()

my_net.setInputSize(3)
my_net.addHiddenFullyConnected(6)
my_net.addHiddenFullyConnected(7)
my_net.setOutputSize(10)

fake_data = np.array([[0.3, 1.2, 4.5]])

prediction = my_net.forward(fake_data)
print(prediction)





def linear_forward(x,w,b):
    z = np.dot(x, w) + b 
    return z

#input
x = np.array([
    [2.0, 1.5],
    [1.3, 5.1] ])

#parameters
w_input_hidden1 = np.array([
    [0.5, -0.3, 0.2],
    [1.2, -1.4, -0.7]
    
])
b_hidden1 = 0.1

w_hidden1_hidden2 = np.array([
    [2.1, -1.2, 1.2, 4.3],
    [1.2, -1.4, -0.7, 0.3],
    [4.3, 0.3, 0.7, -1.2]

])
b_hidden2 = 0.5

w_hidden2_output = np.array([
    [2.3],
    [2.3],
    [2.3],
    [2.3]
])
b_output = 0.2


hidden1 = linear_forward(x, w_input_hidden1, b_hidden1) 
hidden2 = linear_forward(hidden1, w_hidden1_hidden2,b_hidden2)
output = linear_forward(hidden2, w_hidden2_output, b_output)
print(output)

'''


