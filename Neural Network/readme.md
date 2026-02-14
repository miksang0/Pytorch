# Iris Species Classification using PyTorch

This project implements a Fully Connected Neural Network (Multilayer Perceptron) to classify Iris flowers into three species: Setosa, Versicolor, and Virginica. It covers the entire workflow from data ingestion to model persistence.

## Project Overview
The goal is to predict the species of an iris flower based on four physical measurements: sepal length, sepal width, petal length, and petal width.

## Model Architecture
The model is built using `torch.nn.Module` with the following structure:
* **Input Layer:** 4 features (sepal/petal measurements).
* **Hidden Layer 1:** 8 neurons with ReLU activation.
* **Hidden Layer 2:** 10 neurons with ReLU activation.
* **Output Layer:** 3 neurons (representing the 3 flower classes).



---

## Workflow

### 1. Data Preprocessing
* **Loading:** Data is fetched directly from a remote CSV.
* **Encoding:** Categorical labels (`species`) are manually encoded to numerical values (0, 1, 2) for compatibility with the loss function.
* **Splitting:** The dataset is split into **80% Training** and **20% Testing** using Scikit-Learn.
* **Tensor Conversion:** Data is converted to `torch.FloatTensor` for features and `torch.LongTensor` for labels.

### 2. Training Phase
* **Loss Function:** `nn.CrossEntropyLoss` (ideal for multi-class classification).
* **Optimizer:** `Adam` optimizer with a learning rate of `0.01`.
* **Epochs:** 200 iterations through the training set.
* **Backpropagation:** The standard PyTorch "Big Three":
    1. `optimizer.zero_grad()`
    2. `loss.backward()`
    3. `optimizer.step()`



### 3. Evaluation
* Uses `torch.no_grad()` to disable gradient calculation during testing (saving memory and computation).
* Achieved high accuracy on the test set, identifying 30/30 samples correctly in the final run.
* Visualization of the loss curve shows steady convergence over 200 epochs.

---

## Model Persistence
The trained model weights are saved to a file named `iris_model.pt`. 

### How to Load the Model:
```python
# Initialize a new model instance
model = Model()

# Load the saved state dictionary
model.load_state_dict(torch.load('iris_model.pt'))

# Set to evaluation mode
model.eval()
