**Deep Learning with PyTorch**

This repository is a structured, hands-on journey through PyTorch fundamentals to applied Deep Learning projects. It begins with core tensor operations, progresses to a fully connected neural network for structured data, and culminates in a Convolutional Neural Network (CNN) for image classification.

Whether you're transitioning from NumPy to PyTorch or building real neural networks, this repo provides practical, step-by-step implementations.

**Repository Overview**

This project is divided into three major sections:

PyTorch Fundamentals – Tensors & Operations

Iris Species Classification (Fully Connected Neural Network)

MNIST Digit Classification (Convolutional Neural Network)

Each section builds upon the previous one, reinforcing both conceptual understanding and implementation skills.

1️⃣ PyTorch Fundamentals: Tensors & Operations

A practical introduction to PyTorch tensors and the operations required for Deep Learning.

🔹 Key Concepts Covered
Data Conversion

Python Lists → torch.tensor

NumPy Arrays → torch.from_numpy()

Random initialization with:

torch.randn()

torch.zeros()

Tensor Manipulation

.reshape() vs .view()

Dynamic reshaping using -1

Understanding memory sharing between tensors

Slicing & Indexing

Accessing rows, columns, and elements

Feature extraction using : slicing

Tensor Mathematics

Arithmetic operators: + - * / %

Functional API: torch.add(), torch.mul(), etc.

In-place operations: .add_(), .mul_()

This section builds the foundation needed to understand neural network computations.

2️⃣ Iris Species Classification 🌸

Fully Connected Neural Network (Multilayer Perceptron)

A complete end-to-end machine learning workflow using PyTorch.

🎯 Objective

Classify iris flowers into:

Setosa

Versicolor

Virginica

Using 4 input features:

Sepal length

Sepal width

Petal length

Petal width

🧠 Model Architecture
Input Layer (4 features)
→ Hidden Layer (8 neurons, ReLU)
→ Hidden Layer (10 neurons, ReLU)
→ Output Layer (3 classes)
⚙️ Training Setup

Loss Function: nn.CrossEntropyLoss

Optimizer: Adam (lr=0.01)

Epochs: 200

Standard PyTorch Training Loop:

optimizer.zero_grad()
loss.backward()
optimizer.step()
📊 Evaluation

torch.no_grad() used during testing

Achieved 100% test accuracy (30/30 samples) in final run

Loss curve visualization shows smooth convergence

💾 Model Persistence

Model weights saved as:

iris_model.pt

Load with:

model = Model()
model.load_state_dict(torch.load('iris_model.pt'))
model.eval()
3️⃣ CNN for MNIST Digit Classification 🖊️

A Convolutional Neural Network implementation for image classification.

📌 Dataset

MNIST (70,000 handwritten digit images)

Image size: 28×28

10 classes (digits 0–9)

🧠 CNN Architecture

Convolutional Layers

Conv1: 1 → 6 channels, 3×3 kernel

Conv2: 6 → 36 channels, 3×3 kernel

Fully Connected Layers

Flatten

Dense layers

Output layer (10 neurons)

Activation: ReLU
Loss: CrossEntropyLoss
Optimizer: (Adam / SGD — specify in notebook)

📊 Evaluation Metrics

Test Accuracy

Confusion Matrix

Training Loss Tracking

🛠 Installation

Ensure Python 3.8+ is installed.

pip install torch torchvision numpy pandas scikit-learn matplotlib

Clone the repository:

git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

Run the notebooks:

jupyter notebook
📁 Project Structure
├── PyTorch_Fundamentals.ipynb
├── Iris_Classification.ipynb
├── CNN_MNIST.ipynb
├── iris_model.pt
├── README.md
🚀 Learning Outcomes

By completing this repository, you will:

✔ Understand tensor operations and memory behavior
✔ Build and train neural networks from scratch
✔ Implement backpropagation in PyTorch
✔ Work with structured and image datasets
✔ Save and load trained models
✔ Evaluate models using proper metrics

🔮 Future Improvements

Add dropout for regularization

Implement learning rate scheduling

Experiment with deeper CNN architectures

Add model deployment (Flask / FastAPI)

Convert notebooks into modular Python scripts

👨‍💻 Author

Miksang Tamang
GitHub: https://github.com/miksang0

If you found this repository helpful, consider giving it a ⭐ on GitHub!
