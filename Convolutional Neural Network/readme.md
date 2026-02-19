# CNN for MNIST Digit Classification (PyTorch)

This project implements a Convolutional Neural Network (CNN) using PyTorch to classify handwritten digits from the MNIST dataset.

The model is trained and evaluated on the MNIST dataset and demonstrates how convolutional layers, pooling, and fully connected layers work together for image classification.

---

## Project Overview

- Dataset: MNIST (70,000 grayscale handwritten digit images, 28x28 pixels)
- Framework: PyTorch
- Task: Multi-class classification (Digits 0–9)
- Model Type: Convolutional Neural Network (CNN)

---

## Model Architecture

The CNN architecture consists of:

1. **Convolutional Layer 1**
   - Input Channels: 1 (grayscale)
   - Output Channels: 6
   - Kernel Size: 3x3

2. **Convolutional Layer 2**
   - Input Channels: 6
   - Output Channels: 36
   - Kernel Size: 3x3

3. **Fully Connected Layers**
   - Flattened convolution output
   - Dense layers for classification
   - Output layer with 10 neurons (digits 0–9)

4. **Activation Functions**
   - ReLU

5. **Loss Function**
   - CrossEntropyLoss

6. **Optimizer**
   - (Specify what you used, e.g., Adam or SGD)

---

## Dataset

The MNIST dataset is automatically downloaded using:

```python
datasets.MNIST(root='/CNNdata', train=True, download=True, transform=transform)
```

- 60,000 training images
- 10,000 testing images
- Images converted to tensors using `transforms.ToTensor()`

---

## Installation & Requirements

Make sure you have Python 3.8+ installed.

Install required libraries:

```bash
pip install torch torchvision numpy pandas scikit-learn matplotlib
```

---

## How to Run

1. Clone the repository:

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

2. Open the notebook:

```bash
jupyter notebook CNN.ipynb
```

3. Run all cells to:
   - Load the dataset
   - Train the CNN
   - Evaluate performance
   - Generate predictions and confusion matrix

---

## Evaluation

The model is evaluated using:

- Test dataset accuracy
- Confusion matrix
- Loss tracking during training

Example metric used:

```python
from sklearn.metrics import confusion_matrix
```

---

## Sample Workflow

- Load MNIST dataset
- Create DataLoader with batch size = 10
- Define CNN model
- Train model
- Evaluate on test set
- Visualize predictions

---

## Project Structure

```
├── CNN.ipynb
├── README.md
```

---

## Future Improvements

- Add dropout layers for regularization
- Implement learning rate scheduling
- Try deeper CNN architectures
- Add model saving/loading
- Deploy using Flask or FastAPI

---

## License

This project is open-source and available under the MIT License.

---

## Author

Miksang Tamang  
GitHub: https://github.com/miksang0

---

If you found this project helpful, consider giving it a ⭐ on GitHub!

