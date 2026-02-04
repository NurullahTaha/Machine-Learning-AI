import numpy as np
import mnist
# import scipy.misc
import matplotlib.pyplot as plt
import pickle
import os

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

class Convolution_Layer(BaseLayer):
    def __init__(self, filter_size, num_filters, stride, input_depth):
        self.depth = input_depth
        self.filter_size = filter_size
        self.filters = np.random.randn(num_filters, input_depth, filter_size, filter_size) * 0.1
        self.num_filters = num_filters
        self.stride = stride
        self.input_padding = int((self.filter_size - 1) / 2)
        self.biases = np.zeros((num_filters, 1))
        pass

    def forward(self, n_inputs):
        self.n_inputs = n_inputs

        self.height = n_inputs.shape[2]
        self.width = n_inputs.shape[3]

        self.output_w = int(((self.width - self.filter_size) / self.stride) + 1)
        self.output_h = int(((self.height - self.filter_size) / self.stride) + 1)

        self.output_size = (n_inputs.shape[0], self.num_filters, self.output_h, self.output_w)

        self.output = np.zeros(self.output_size)

        for image in range(n_inputs.shape[0]):

            for filter in range(self.num_filters):

                for height in range(self.output_h):
                    
                    for width in range(self.output_w):

                        height_start = height * self.stride
                        width_start = width * self.stride

                        height_end = height_start + self.filter_size
                        width_end = width_start + self.filter_size

                        img_slice = self.n_inputs[image, : , height_start : height_end, width_start : width_end]

                        self.output[image, filter, height, width] = np.sum(img_slice * self.filters[filter]) + self.biases[filter, 0]

        return self.output
                        




    def backward(self, out_error):
        batch_size = out_error.shape[0]

        

        self.d_filter = np.zeros_like(self.filters)
        self.d_biases = np.zeros_like(self.biases)
        self.d_inputs = np.zeros_like(self.n_inputs)

        limit = out_error.shape[0] 

        for image in range(limit):

            for filter in range(self.num_filters):

                for height in range(self.output_h):
                    
                    for width in range(self.output_w):

                        height_start = height * self.stride
                        width_start = width * self.stride

                        height_end = height_start + self.filter_size
                        width_end = width_start + self.filter_size

                        current_error = out_error[image, filter, height, width]

                        img_slice = self.n_inputs[image, : , height_start : height_end, width_start : width_end]

                        self.d_filter[filter] += img_slice * current_error
                        self.d_biases[filter] += current_error
                        self.d_inputs[image, : , height_start : height_end, width_start : width_end] += self.filters[filter] * current_error

        return self.d_inputs


    def update(self, learning_rate):

        self.filters = self.filters - learning_rate * self.d_filter
        self.biases = self.biases - learning_rate * self.d_biases

        pass

class Pooling_Layer(BaseLayer):
    def __init__(self, p_size, stride):
        self.p_size = p_size
        self.stride = stride

    def forward(self, n_inputs):
        self.n_inputs = n_inputs

        self.batch_size = n_inputs.shape[0]
        self.n_channels = n_inputs.shape[1]
        self.height = n_inputs.shape[2]
        self.width = n_inputs.shape[3]

        self.output_h = int((((self.height - self.p_size) / self.stride) + 1))
        self.output_w = int((((self.width - self.p_size) / self.stride) + 1))

        self.output = np.zeros(self.batch_size, self.n_channels, self.output_h, self.output_w)

        for image in range(self.batch_size):
            for channel in range(self.n_channels):
                for height in range(self.output_h):
                    for width in range(self.output_w):

                        height_start = height * self.stride
                        width_start = width * self.stride

                        height_end = height_start + self.p_size
                        width_end = width_start + self.p_size

                        patch = self.n_inputs[image, channel, height_start : height_end, width_start : width_end]
                        self.output[image, channel, height, width] = np.mean(patch)


        
        return self.output


    def backward(self, out_error):

        self.input_gradient = np.zeros(self.batch_size, self.n_channels, self.height, self.width)

        self.patch_area = self.p_size * self.p_size

        for image in range(self.batch_size):
            for channel in range(self.n_channels):
                for height in range(self.output_h):
                    for width in range(self.output_w):

                        height_start = height * self.stride
                        width_start = width * self.stride

                        height_end = height_start + self.p_size
                        width_end = width_start + self.p_size

                        self.current_error = out_error[image, channel, height, width]

                        self.input_gradient[image, channel, height_start : height_end, width_start : width_end] += self.current_error / self.patch_area
        
        return self.input_gradient


    def update(self, learning_rate):
        pass


class Conversion(BaseLayer):
    def forward(self, n_inputs):
        self.n_inputs = n_inputs
        self.n_inputs_shape = n_inputs.shape
        self.flatten = self.n_inputs.reshape(n_inputs.shape[0], (n_inputs.shape[1] * n_inputs.shape[2] * n_inputs.shape[3]))
        return self.flatten
    
    def backward(self, out_error):
        return out_error.reshape(self.n_inputs_shape[0], self.n_inputs_shape[1], self.n_inputs_shape[2], self.n_inputs_shape[3])
    
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
            layer.update(0.1)




print("Loading Data...")

#function open and load the batches
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

#initalise the arrays for the data
train_images = []
train_labels = []

#store the data into the arrays
for i in range(1, 6):
    data_dict = unpickle(os.path.join('cifar-10-batches-py', f'data_batch_{i}'))
    train_images.append(data_dict[b'data'])
    train_labels.append(data_dict[b'labels'])

#merge the data in the arrays to form one big array
raw_train_images = np.vstack(train_images)
raw_train_labels = np.hstack(train_labels)

#load the test images into the array
test_dict = unpickle(os.path.join('cifar-10-batches-py', 'test_batch'))
raw_test_images = test_dict[b'data']
raw_test_labels = test_dict[b'labels']

train_images = raw_train_images.reshape(-1, 3, 32, 32)
test_images = raw_test_images.reshape(-1, 3, 32, 32)


X_train = train_images / 255
X_test = test_images / 255


def one_hot(labels):
    output = np.zeros((len(labels), 10))
    for i in range(len(labels)):
        output[i, labels[i]] = 1
    return output

y_test = one_hot(raw_test_labels)
y_train = one_hot(raw_train_labels)

LIMIT = 1000
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

if os.path.exists('cifar10_brain.pkl'):
    print("FOUND SAVED BRAIN! Resuming training...")
    
    with open('cifar10_brain.pkl', 'rb') as f:
        net = pickle.load(f)

else:
    print("✨ NO SAVE FOUND. Creating new random brain...")
    
    # Create from scratch
    net = Network()
    net.add(Convolution_Layer(3, 16, 2, 3))
    net.add(Activation_ReLu())
    net.add(Conversion())

    net.add(Layer_Dense(3600, 128))   
    net.add(Activation_ReLu())

    # net.add(Conversion())
    # net.add(Layer_Dense(1024*3, 128))   
    # net.add(Activation_ReLu())

    net.add(Layer_Dense(128, 64))
    net.add(Activation_ReLu())

    net.add(Layer_Dense(64, 64))
    net.add(Activation_ReLu())

    net.add(Layer_Dense(64, 10))
    net.add(Activation())

# 4. TRAIN

print("Training started...")

batch_size = 1

for epoch in range(10): 
    error_sum = 0
    for i in range(0, len(X_train), batch_size):
        img = X_train[i: i + batch_size] 
        target = y_train[i: i + batch_size]
    
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
    
    
    num_batches = len(X_train) // batch_size
    print(f"Epoch {epoch}, Loss: {error_sum/num_batches:.4f}, Val Acc: {accuracy*100:.2f}%")

#Test

print(f"Study Material: {len(X_train)} images")
print(f"Final Exam Size: {len(test_images)} images")

final_test = net.forward(X_test)

prediction = np.argmax(final_test, axis=1)
label = np.argmax(y_test, axis=1)

accuracy = np.mean(prediction == label)

# ==========================================
# 6. FINAL REPORT & SAVE
# ==========================================
print("\n--- SAVING MODEL ---")
with open('cifar10_brain.pkl', 'wb') as f:
    pickle.dump(net, f)
print("Brain saved to 'cifar10_brain.pkl'!")

# Final Test
final_test = net.forward(X_test)
prediction = np.argmax(final_test, axis=1)
label = np.argmax(y_test, axis=1)
accuracy = np.mean(prediction == label)
print(f"Official Test Accuracy: {accuracy*100:.2f}%")




# ==========================================
# 7. VISUAL TEST 🖼️
# ==========================================
print("\n--- VISUAL TEST ---")

# Names for CIFAR-10
class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

idx = np.random.randint(0, len(X_test))
sample = X_test[idx:idx+1]
true_label = np.argmax(y_test[idx]) # Decode one-hot back to number

probs = net.forward(sample)
pred_idx = np.argmax(probs)

print(f"True Label: {class_names[true_label]}")
print("Network's Confidence:")
for i in range(10):
    p = probs[0][i] * 100
    print(f"  {class_names[i]}: {p:.2f}%")

print(f"\n> Final Prediction: {class_names[pred_idx]}")

# Show Image (32x32 now!)
image_grid = sample.reshape(3, 32, 32).transpose(1, 2, 0)
plt.imshow(image_grid, cmap='gray')
plt.title(f"True: {class_names[true_label]} | Pred: {class_names[pred_idx]}")
plt.show()