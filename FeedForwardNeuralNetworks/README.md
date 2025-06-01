
# Exploring Feedforward Neural Networks: Analysis on Regression and Image Classification Tasks

## Author
- [Jenny Van](https://github.com/tuongv-1736461)

## Abstract

This project investigates the performance of feedforward neural networks (FNNs) on two distinct tasks:

1. **Regression** using a synthetic dataset from an [Optimization Project](https://github.com/tuongv-1736461/Machine-Learning-projects/tree/main/Optimization), where a three-layer FNN is trained and compared against polynomial models.
2. **Classification** using the MNIST dataset, where dimensionality is reduced using Principal Component Analysis (PCA) before training FNN, LSTM, SVM, and decision tree models.

This work compares different models' effectiveness, explores the role of training/test set splits, and evaluates how dimensionality reduction impacts classification accuracy.

---

## Table of Contents
1. [Introduction](#1-introduction)  
2. [Theoretical Background](#2-theoretical-background)  
3. [Methodology](#3-methodology)  
   - [3.1 Regression with FNN](#31-regression-with-fnn)  
   - [3.2 MNIST Classification with PCA](#32-mnist-classification-with-pca)  
4. [Results](#4-results)  
5. [Conclusion](#5-conclusion)  

---

## 1. Introduction

Feedforward neural networks (FNNs) are foundational models in machine learning. This project evaluates their performance in:

- **Function fitting tasks**, using synthetic X and Y data.
- **Image classification tasks**, using the MNIST dataset.

In the regression task, we assess generalization across various train/test splits and compare the FNN with linear and high-degree polynomial models. In the classification task, PCA is applied to reduce input dimensionality before training a three-layer FNN, LSTM, and classical ML models. We then evaluate each model's accuracy and robustness.

---

## 2. Theoretical Background

### Feedforward Neural Networks (FNNs)
FNNs consist of input, hidden, and output layers where information flows in one direction. Each node in a hidden layer applies a non-linear activation function to its input, enabling the network to learn complex patterns. We use ReLU activation in this study.

### Least Squares Error
This is used as a loss function in regression, measuring the sum of squared differences between predicted and actual values. Minimizing this error via gradient descent adjusts the model’s parameters.

### Principal Component Analysis (PCA)
PCA reduces high-dimensional data by projecting it onto orthogonal components that capture the most variance. It improves computational efficiency and often enhances generalization for classical models.

### LSTM
Long Short-Term Memory networks are a type of Recurrent Neural Network (RNN) that maintain memory over sequences. Although typically used in time-series or text, here we evaluate its performance on PCA-reduced MNIST data for comparison.

---

## 3. Methodology

### 3.1 Regression with FNN

#### Dataset
- Input: `X = [0, 1, 2, ..., 30]`
- Target: `Y = noisy observations from a real-world optimization problem`

```
X = np.arange(0, 31)
Y = np.array([30, 35, 33, 32, 34, 37, 39, 38, 36, 36, 37, 39, 42, 45, 45, 41, 40, 39, 42, 44, 47, 49, 50, 49, 46, 48, 50, 53, 55, 54, 53])
```
This is a simple set of numbers where we try to predict `Y` values from `X` using different types of models.

#### Models Compared
- Linear regression
- Quadratic and 19th-degree polynomial regression
- Three-layer FNN

#### Architecture
```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
```

#### Training Details
- Loss function: Mean Squared Error (MSE)
- Optimizer: SGD with learning rate = 0.01
- Epochs: 10 (model sees the data 10 times)
- Evaluation: Comparison of training and test error for two splits:
  1. Train = first 20, Test = last 10
  2. Train = first and last 10, Test = middle 10

---

### 3.2 MNIST Classification with PCA

#### Data Processing
- MNIST images (28×28) flattened to 784-dimensional vectors. 
- PCA reduced dimensionality to 20 components. 
- Same PCA transformation applied to training and testing data. 

#### FNN Architecture
```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(20, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
```

#### Models Compared
- Three-layer FNN (PyTorch)
- LSTM with reshaped PCA input
- SVM (scikit-learn)
- Decision Tree (scikit-learn)

#### Training Details
- Optimizer: SGD or Adam
- Loss: CrossEntropyLoss
- Epochs: 5–10
- Batch size: 64
- Accuracy evaluated on MNIST 10,000 test images

---

## 4. Results

### 4.1 Regression Task – Least Squares Errors

#### Case 1: Train = First 20 | Test = Last 10
| Model                        | Train Error | Test Error        |
|-----------------------------|-------------|-------------------|
| Linear Regression           | 5.03        | 11.31             |
| Quadratic Polynomial        | 4.52        | 75.93             |
| 19th-degree Polynomial      | 3.52        | 7.5e+14 (overfit) |
| **FNN (3-layer)**           | 16.37       | 87.49             |

#### Case 2: Train = First & Last 10 | Test = Middle 10
| Model                        | Train Error | Test Error |
|-----------------------------|-------------|------------|
| Linear Regression           | 3.43        | 8.65       |
| Quadratic Polynomial        | 3.43        | 8.44       |
| 19th-degree Polynomial      | 3.65        | 25.42      |
| **FNN (3-layer)**           | 67.03       | 8.65       |

####  Analysis
- **Linear and quadratic models** performed consistently well in both cases, balancing simplicity with generalization.
- The **19th-degree polynomial** had very low train error but drastically higher test error, especially in Case 1. This is a classic case of **overfitting**, where the model memorizes the training data and fails to generalize.
- The **feedforward neural network (FNN)** had inconsistent results. In Case 1, its high train and test error suggest **underfitting**, possibly due to insufficient training or model capacity. In Case 2, it matched the test performance of simpler models but still showed a much higher train error.

> These results highlight the trade-off between model complexity and generalization. A simpler model may outperform a complex one when data is limited.

---

### 4.2 MNIST Classification Accuracy

| Model                        | Accuracy    |
|-----------------------------|-------------|
| **SVM**                     | **99.56%**   |
| Decision Tree               | 96.87%       |
| Feedforward NN              | 95.66%       |
| LSTM                        | 94.98%       |

#### Analysis
- The **Support Vector Machine (SVM)** performed best, benefiting from PCA-reduced inputs and its ability to find clean decision boundaries.
- The **Decision Tree** was also strong and offers high interpretability.
- The **FNN** performed competitively but slightly under the SVM, possibly due to information loss from PCA or limited training.
- The **LSTM**, which is designed for sequential data, did not perform as well in this non-sequential task.

> These results show that classic machine learning models like SVMs can still outperform neural networks in certain settings, especially when the dataset is structured and dimensionality is reduced effectively.

---

## 5. Conclusion

This project explored how feedforward neural networks (FNNs) perform in two common machine learning tasks: regression and classification. By comparing FNNs to both traditional models (like linear regression and decision trees) and more advanced models (like LSTMs and SVMs), we gained insights into when neural networks are effective and when simpler models may be more appropriate.

### Key Insights:
- **For regression**, FNNs did not outperform simpler models. In some cases, they underfit the data, highlighting the need for careful model tuning and appropriate training. High-degree polynomial models, while powerful, suffered from overfitting — performing well on training data but poorly on new data.
- **For classification**, FNNs performed strongly but were slightly outperformed by SVMs. This suggests that with well-structured, low-dimensional data (after PCA), traditional ML models can still be very competitive.
- **PCA** proved useful in reducing input complexity without significantly harming model accuracy, especially for classical ML algorithms.

### Final Takeaway:
Neural networks are powerful tools, but they’re not always the best solution out of the box. Simpler models can offer excellent performance with less computational cost, especially when data is clean and well-preprocessed. Understanding the trade-offs between complexity, interpretability, and generalization is key to choosing the right model for any problem.


