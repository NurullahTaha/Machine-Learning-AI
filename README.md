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
    *(Note: `mnist` package might be required if you plan to retrain the digit recognizer)*

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

### Training & Experimentation
You can explore the neural network implementation in `cifar10.py` and `ANN.py`. These files contain the class definitions for the network architecture, layers, and training loops.

## 📂 File Structure

*   `run_model.py`: The interactive GUI application for digit recognition.
*   `cifar10.py`: Contains the core `Network`, `Layer_Dense`, `Convolution_Layer`, and activation class definitions.
*   `ANN.py`: Initial/Alternative implementation of the neural network components.
*   `my_brain.pkl`: A serialized (pickled) trained model used by the demo.

## 🧠 Theory

This project implements the core components of Deep Learning:
*   **Forward Propagation:** Computing the output by passing inputs through layers.
*   **Backpropagation:** Calculating gradients of the loss function with respect to weights using the Chain Rule.
*   **Gradient Descent:** Updating weights to minimize the error.