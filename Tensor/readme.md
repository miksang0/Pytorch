# PyTorch Fundamentals: Tensors and Operations

This repository contains a comprehensive introduction to **PyTorch**, focusing on the transition from standard Python data structures to Tensors, and exploring the fundamental operations required for Deep Learning.

## Overview
The notebook serves as a practical guide for:
* Converting between Lists, Numpy Arrays, and PyTorch Tensors.
* Understanding Tensor dimensionality (1D, 2D, and 3D).
* Mastering Tensor reshaping, viewing, and slicing techniques.
* Performing element-wise mathematical operations.

---

## Key Topics Covered

### 1. Data Conversion
Demonstrates how to initialize data and bridge the gap between scientific computing libraries.
* **Lists to Tensors:** Standard Python lists to `torch.tensor`.
* **Numpy Integration:** Creating tensors from `numpy.ndarray` using `torch.from_numpy()` or `torch.tensor()`.
* **Random Initialization:** Generating noise/weights using `torch.randn` and `torch.zeros`.

### 2. Manipulation & Reshaping
Crucial for preparing data to enter Neural Network layers.
* **`.reshape()` vs. `.view()`:** Understanding how to change dimensions while maintaining data integrity.
* **Dynamic Reshaping:** Using `-1` to let PyTorch automatically calculate the necessary dimension size.
* **Memory Linking:** Observations on how changing an original tensor affects its reshaped "view."



### 3. Slicing and Indexing
Techniques to grab specific data points, rows, or columns.
* **Basic Slicing:** Accessing specific indices.
* **Column Extraction:** Using `:` syntax to isolate features (vital for separating labels from data).

### 4. Tensor Mathematics
Arithmetic operations optimized for GPU/CPU performance.
* **Standard Operators:** `+`, `-`, `*`, `/`, `%`.
* **Functional API:** `torch.add()`, `torch.sub()`, `torch.mul()`, etc.
* **In-place Operations:** Using the underscore suffix (e.g., `.add_()`) to modify tensors in memory without creating new copies.

---

## Getting Started

### Prerequisites
Ensure you have the following installed:
* Python 3.x
* PyTorch
* NumPy

### Usage
Run the cells sequentially to observe how Tensors behave under different operations. Pay close attention to the `dtype` outputs, as PyTorch often defaults to `float32` or `float64` depending on the input source.

```python
import torch
import numpy as np

# Quick Example: Creating a 2x3 Tensor
tensor = torch.randn(2, 3)
print(tensor)
