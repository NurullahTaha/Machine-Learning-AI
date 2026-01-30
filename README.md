# Machine Learning AI (From Scratch)

This project is a deep learning library and application built entirely from scratch using **Python** and **NumPy**. It demonstrates the fundamental mathematics and logic behind neural networks without relying on high-level frameworks like PyTorch or TensorFlow.

## 🚀 Features

*   **No Frameworks:** All layers, activation functions, and backpropagation logic are implemented manually using matrix operations.
*   **Custom Layers:**
    *   **Dense (Fully Connected) Layers:** Standard neural network layers.
    *   **Convolutional Layers:** Implements 2D convolution for image processing.
*   **Activation Functions:**
    *   **ReLU:** Rectified Linear Unit for hidden layers.
    *   **Sigmoid:** For output/binary classification.
*   **Interactive Demo:** Includes a GUI tool (`run_model.py`) where you can draw digits (0-9) and have the AI predict them in real-time.

## 📂 Code Structure & Implementation Details

The core philosophy of this project is to implement every component of a neural network manually. Here is how the codebase is organized:

### 1. The Building Blocks (`cifar10.py` & `ANN.py`)
*   **`BaseLayer`**: An abstract parent class that defines the blueprint for all layers (`forward`, `backward`, `update`).
*   **`Network` Class**:
    *   Acts as the container for the model.
    *   **`add(layer)`**: Stacks layers sequentially (similar to `Sequential()` in Keras).
    *   **`forward(input)`**: Propagates data through the network.
    *   **`backward(targets)`**: Calculates the error and backpropagates it through all layers in reverse order.
    *   **`update()`**: Updates weights and biases based on the calculated gradients.

### 2. Convolutional Layer (`Convolution_Layer`)
This is the most complex component, designed for image recognition tasks (like CIFAR-10).
*   **Initialization**: defined by `filter_size` (e.g., 3x3), `num_filters` (depth of output), `stride` (step size), and `input_depth`.
*   **Forward Pass**:
    *   Instead of using optimized library calls, this implementation uses **nested loops** to slide the filters over the input image.
    *   It extracts slices of the input image and performs a dot product with the filter weights + bias.
    *   **Output Dimensions**: Automatically calculated based on input size, filter size, and stride:
        $$ Output_{dim} = \frac{Input - Filter}{Stride} + 1 $$
*   **Backward Pass**:
    *   Calculates gradients for **Filters** (`d_filter`), **Biases** (`d_biases`), and the **Input** (`d_inputs`) to pass gradients to the previous layer.

### 3. Dense & Utility Layers
*   **`Layer_Dense`**: A standard fully connected layer.
    *   **Forward**: $Y = X \cdot W + B$
    *   **Backward**: Computes gradients $dW$, $dB$, and $dE$ (error w.r.t input).
*   **`Conversion` Layer**:
    *   Acts as a bridge between 3D Convolutional layers and 1D Dense layers.
    *   **Flattening**: Reshapes the $(Batch, Depth, Height, Width)$ output of a Conv layer into a $(Batch, N)$ vector.

### 4. Data Handling
*   **CIFAR-10 Loader**: The script manually unpickles the official CIFAR-10 data batches.
*   **Preprocessing**: Images are normalized (divided by 255) to the range [0, 1]. Labels are one-hot encoded.

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/NurullahTaha/Machine-Learning-AI.git
    cd Machine-Learning-AI
    ```

2.  **Install dependencies:**
    This project mainly relies on `numpy` for math and `matplotlib` for the GUI.
    ```bash
    pip install numpy matplotlib mnist
    ```

3.  **Dataset Setup (Required for Training):**
    To train the model using `cifar10.py`, you need the **CIFAR-10 python version** dataset.
    
    *   **Download:** [CIFAR-10 Python Version](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)
    *   **Extract:** Unzip the downloaded file.
    *   **Place:** Move the extracted folder named `cifar-10-batches-py` into the root directory of this project.
    
    Your folder structure should look like this:
    ```text
    Machine-Learning-AI/
    ├── cifar-10-batches-py/  <-- The dataset folder
    ├── cifar10.py
    ├── run_model.py
    └── ...
    ```

## 🎮 Usage

### Interactive Digit Recognizer
Run the interactive drawing pad to test the pre-trained model:

```bash
python run_model.py
```
*   **Draw:** Use your mouse to draw a digit (0-9) on the canvas.
*   **Predict:** Click the **PREDICT** button to see what the AI thinks you drew.
*   **Clear:** Reset the canvas to try again.

*Note: This script requires a trained model file named `my_brain.pkl` in the same directory.*

### Training the CIFAR-10 Model
To train the Convolutional Network on the CIFAR-10 dataset:
```bash
python cifar10.py
```
*   This will look for the `cifar-10-batches-py` folder.
*   It trains for a set number of epochs and prints the Loss/Accuracy.
*   Saves the trained model to `cifar10_brain.pkl`.

## 🧠 Theory

This project implements the core components of Deep Learning:
*   **Forward Propagation**: Computing the output by passing inputs through layers.
*   **Backpropagation**: Calculating gradients of the loss function with respect to weights using the Chain Rule.
*   **Gradient Descent**: Updating weights to minimize the error.