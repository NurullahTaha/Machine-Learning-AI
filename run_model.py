import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import pickle

# =========================================================
# 1. YOUR CLASS DEFINITIONS 🏗️
# (These must match exactly what you trained with)
# =========================================================

class BaseLayer():
    def forward(self, n_inputs):
        pass
    
    def backward(self, out_error):
        pass

    def update(self, learning_rate):
        pass


class Layer_Dense(BaseLayer):
    def __init__(self, n_inputs, n_neurons):
        # Initializing weights and biases
        self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(2. / n_inputs)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, n_inputs):
        self.input = n_inputs
        self.output = np.dot(self.input, self.weights) + self.biases
        return self.output
    
    def backward(self, out_error):
        # Calculate gradients
        self.dW = np.dot(self.input.transpose(), out_error)
        self.dB = np.sum(out_error, axis=0, keepdims=True)
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
        self.out_error = out_error
        self.init_sig = 1 / (1 + np.exp(-self.n_inputs))
        self.der_sig = 1 - (1 / (1 + np.exp(-self.n_inputs)))
        self.final = self.init_sig * self.der_sig
        return self.final * out_error

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
        for layer in self.layers[::-1]:
            loss = layer.backward(loss) 

    def update(self):
        for layer in self.layers:
            layer.update(0.07)

# =========================================================
# 2. LOAD THE SAVED BRAIN 🧠
# =========================================================
print("Loading saved brain...")

try:
    with open('my_brain.pkl', 'rb') as f:
        net = pickle.load(f)
    print("Success! Brain loaded.")
except FileNotFoundError:
    print("\n❌ ERROR: 'my_brain.pkl' not found!")
    print("Please run your training script once to generate the file.")
    exit()

# =========================================================
# 3. THE DRAWING PAD 🎨
# =========================================================
print("\n--- INSTRUCTIONS ---")
print("1. Draw a number (0-9).")
print("2. Click PREDICT.")
print("3. Click CLEAR to try again.")

# Setup Canvas
canvas_grid = np.zeros((28, 28))
fig, ax = plt.subplots(figsize=(7, 7))
plt.subplots_adjust(bottom=0.2)
ax.set_title("Draw a Digit Here!", fontsize=16)
ax.axis('off') 
img_display = ax.imshow(canvas_grid, cmap='gray', vmin=0, vmax=1)

drawing = False

def on_mouse_down(event):
    global drawing
    drawing = True

def on_mouse_up(event):
    global drawing
    drawing = False

def on_mouse_move(event):
    global canvas_grid
    # Only draw if the mouse is pressed AND inside the canvas
    if drawing and event.xdata is not None and event.ydata is not None:
        x, y = int(event.xdata), int(event.ydata)
        
        # --- NEW SOFT BRUSH LOGIC ---
        
        # 1. Paint the Center (Solid White)
        if 0 <= x < 28 and 0 <= y < 28:
            canvas_grid[y, x] = 1.0
            
        # 2. Paint the Neighbors (Feathered/Soft)
        # We use a Cross (+) shape instead of a Square (box).
        # We set them to 0.5 (Grey) so edges look soft.
        feather_val = 0.5
        
        # Offsets for Up, Down, Left, Right (No corners!)
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        for dx, dy in offsets:
            nx, ny = x + dx, y + dy
            
            # Bounds check to keep it inside the box
            if 0 <= nx < 28 and 0 <= ny < 28:
                # We use max() so we don't accidentally turn a white pixel 
                # back into a grey one if we cross over it again.
                canvas_grid[ny, nx] = max(canvas_grid[ny, nx], feather_val)

        # Update the screen
        img_display.set_data(canvas_grid)
        fig.canvas.draw_idle()

# Buttons
ax_pred = plt.axes([0.55, 0.05, 0.2, 0.075])
ax_clear = plt.axes([0.25, 0.05, 0.2, 0.075])
btn_pred = Button(ax_pred, 'PREDICT')
btn_clear = Button(ax_clear, 'CLEAR')

def predict(event):
    input_vec = canvas_grid.reshape(1, 784)
    probs = net.forward(input_vec)
    pred = np.argmax(probs)
    conf = np.max(probs) * 100
    
    print(f"AI Guessed: {pred} ({conf:.2f}%)")
    ax.set_title(f"I think it's a {pred}! ({conf:.1f}%)", fontsize=16, color='blue')
    fig.canvas.draw_idle()

def clear(event):
    global canvas_grid
    canvas_grid = np.zeros((28, 28))
    img_display.set_data(canvas_grid)
    ax.set_title("Draw a Digit Here!", fontsize=16, color='black')
    fig.canvas.draw_idle()

fig.canvas.mpl_connect('button_press_event', on_mouse_down)
fig.canvas.mpl_connect('button_release_event', on_mouse_up)
fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)
btn_pred.on_clicked(predict)
btn_clear.on_clicked(clear)

plt.show()