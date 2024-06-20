
# Exploring Feed-Forward Neural Networks: Analysis on Multiple Datasets

## Author
- [Jenny Van](https://github.com/tuongv-1736461)
## Abstract

This report investigates the application of feed-forward neural networks on two datasets. Firstly, a three-layer neural network is trained and evaluated on data points from [Optimization project](https://github.com/tuongv-1736461/Machine-Learning-projects/tree/main/Optimization), with the least-square errors computed and results compared on different training and testing sets. Secondly, a feed-forward neural network is trained on the MNIST dataset, and its performance is compared to LSTM, SVM, and decision tree classifiers after computing the first 20 principal component analysis (PCA) modes. The report explores the effectiveness of feed-forward neural networks and provides insights into their performance relative to other classifiers across different datasets.

## Introduction 

This report explores feed-forward neural networks' application on two datasets. Part one focuses on reevaluating [Optimization project](https://github.com/tuongv-1736461/Machine-Learning-projects/tree/main/Optimization)'s X and Y variables, fitting them to a three-layer neural network, and evaluating its performance using the least-square error on a test set. A comparative analysis is conducted by repeating the process with alternative training sets. The second part of the study involves training a feed-forward neural network on the MNIST dataset, consisting of handwritten digit images. PCA is used to compute the first 20 modes of the digit images. A feed-forward neural network is then built to classify the digits, and its performance is compared to LSTM, SVM, and decision tree classifiers. This report aims to explore the application of feed-forward neural networks in different datasets and provide insights into their performance compared to alternative classifiers. By conducting the analysis outlined above, valuable conclusions can be drawn regarding the effectiveness of neural networks and their comparative performance with other popular classification algorithms.

## Theoretical Background

Feed-forward Neural Networks:

Feed-forward neural networks are a type of artificial neural network that consists of interconnected layers of neurons. In these networks, information flows in a forward direction, from the input layer through one or more hidden layers to the output layer. Each neuron in a layer receives inputs from the previous layer and applies a non-linear activation function to produce an output. The weights connecting the neurons determine the strength of the connections and are adjusted during the training process to optimize the network's performance.

Least-Square Error:

Least-square error is a commonly used loss function in regression tasks, which measures the discrepancy between predicted and actual values. It calculates the squared difference between the predicted output and the true output and sums these squared differences over all data points. The goal is to minimize this error, typically achieved through optimization algorithms such as gradient descent.

LSTM (Long Short-Term Memory):

LSTM is a type of recurrent neural network (RNN) architecture that is particularly effective in modeling sequences and time-dependent data. It overcomes the limitation of traditional RNNs by incorporating memory cells, which allow the network to retain and access information over longer time intervals. LSTM networks are widely used in tasks such as natural language processing, speech recognition, and time series analysis.

In this report, we investigate the application of feed-forward neural networks in two different datasets. Firstly, we fit the data from [Optimization project](https://github.com/tuongv-1736461/Machine-Learning-projects/tree/main/Optimization) to a three-layer feed-forward neural network and evaluate its performance using least-square error. We compare the results obtained by training the network with different sets of training and test data. Secondly, we shift our focus to the MNIST dataset, consisting of handwritten digit images. We compute the first 20 principal component analysis (PCA) modes of the digit images and construct a feed-forward neural network for digit classification. We compare the performance of the neural network with other classifiers, including LSTM, SVM, and decision tree classifiers.

## Algorithm Implementation and Development

### Training and Evaluating a Neural Network for Regression on Two Different Datasets

The code starts by importing the necessary libraries, including numpy for data generation and torch for neural network-related operations.

```
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
```
Data points are generated using numpy, with X representing the input values and Y representing the corresponding target values.
```
X = np.arange(0, 31)
Y = np.array([30, 35, 33, 32, 34, 37, 39, 38, 36, 36, 37, 39, 42, 45, 45, 41, 40, 39, 42, 44, 47, 49, 50, 49, 46, 48, 50, 53, 55, 54, 53])
```
The neural network architecture is defined using the nn.Module class from PyTorch. It consists of three fully connected layers with ReLU activation, and the forward method defines the forward pass of the network.
```
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```
The data is split into two sets, X_train_1 and Y_train_1 for the first training set, and X_test_1 and Y_test_1 for the first test set. Similarly, a second set of training and test data (X_train_2, Y_train_2, X_test_2, Y_test_2) is created.
```
X_train_1, Y_train_1 = X[:20], Y[:20]
X_test_1, Y_test_1 = X[10:], Y[10:]

X_train_2 = np.concatenate((X[:10], X[-10:]))
Y_train_2 = np.concatenate((Y[:10], Y[-10:]))
X_test_2 = X[10:-10]
Y_test_2 = Y[10:-10]
```
A function named "train_and_evaluate" is defined to train and evaluate a model. It takes the training and test data as input.

```
def train_and_evaluate(X_train, Y_train, X_test, Y_test):
```
Within the function, the training and test data are converted to tensors using torch.Tensor and unsqueeze operations to match the expected input shape of the neural network.
```
  # Convert the training and test data to tensors
  X_train_tensor = torch.Tensor(X_train).unsqueeze(1)
  Y_train_tensor = torch.Tensor(Y_train).unsqueeze(1)
  X_test_tensor = torch.Tensor(X_test).unsqueeze(1)
  Y_test_tensor = torch.Tensor(Y_test).unsqueeze(1)
```
A new instance of the neural network is created, defining its architecture and layers.
```
  # Create a new neural network
  net = Net()
```
For training the network, the loss function (MSELoss) and optimizer (SGD) are defined. The mean squared error (MSE) is used as the loss function to measure the difference between predicted and actual values, and stochastic gradient descent (SGD) is employed as the optimizer to update the model's parameters.
```
  # Define the loss function
  criterion = nn.MSELoss()

  # Define the optimizer
  optimizer = optim.SGD(net.parameters(), lr=0.01)
```
The number of epochs, which determines the number of complete iterations over the training data, is set to 10.
```
  # Set the number of epochs
  num_epochs = 10
```
The code executes a training loop, where the optimizer's gradients are zeroed at the beginning of each epoch. The model's outputs are then computed for the training data, and the loss is calculated using the defined loss function. The gradients are subsequently backpropagated through the network, and the optimizer updates the model's parameters based on these gradients. During training, the loss is printed at every 100th epoch, providing an indication of the model's progress.
```
  # Training loop
  for epoch in range(num_epochs):
      optimizer.zero_grad()
      outputs = net(X_train_tensor)
      loss = criterion(outputs, Y_train_tensor)
      loss.backward()
      optimizer.step()

      # Print the loss at every 100th epoch
      if (epoch+1) % 100 == 0:
          print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')
```
After training, the model is evaluated on both the training and test data. Predictions are generated for both sets, and the loss is computed using the same loss function. This evaluation allows assessing the model's performance on both the data it was trained on and unseen test data.
```
  # Evaluate the model on training and test data
  with torch.no_grad():
      train_predictions = net(X_train_tensor)
      train_loss = criterion(train_predictions, Y_train_tensor)
      test_predictions = net(X_test_tensor)
      test_loss = criterion(test_predictions, Y_test_tensor)
```
Finally, the training and test errors are printed, providing insights into the accuracy and performance of the trained neural network model.
```
  # Print the training and test errors
  print(f'Train Error: {train_loss.item()}')
  print(f'Test Error: {test_loss.item()}')
```
Finally, the "train_and_evaluate" function is called twice, each with a different set of training and test data, to train and evaluate the model for both sets.
```
train_and_evaluate(X_train_1, Y_train_1, X_test_1, Y_test_1)
train_and_evaluate(X_train_2, Y_train_2, X_test_2, Y_test_2)
```
Overall, this code demonstrates the implementation of a neural network using PyTorch to perform regression on the given data. It includes steps such as data preprocessing, defining the model architecture, training the model, and evaluating its performance on training and test data.

### Neural Network Training and Evaluation for Image Classification using PCA and MNIST Dataset

This code performs the training and evaluation of a neural network for image classification using the MNIST dataset. It begins by importing the necessary libraries: 
```
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from sklearn.decomposition import PCA
```
The neural network architecture is defined with three fully connected (linear) layers. The input size is adjusted to accommodate the PCA-transformed data, and ReLU activation is applied after the first two linear layers.
```
# Define the neural network architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(20, 128)   # Adjusting input size to 20
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x.float()))  # Convert input to float
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```
The MNIST dataset is loaded and preprocessed. The training and test datasets are loaded using datasets.MNIST, and transforms.ToTensor() is used to convert the image data to tensors. The training images are flattened and converted to float. PCA is then performed on the flattened training images using sklearn.decomposition.PCA, with 20 PCA components computed.
```
# Load the MNIST dataset and apply transformations
train_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

# Get the data tensors from the datasets
X_train = train_dataset.data.float()  # Convert image data to float
Y_train = train_dataset.targets

# Reshape the input images for PCA (flatten each image)
X_train_flattened = X_train.view(X_train.size(0), -1)

# Compute PCA on the flattened images
pca = PCA(n_components=20)  # Number of PCA modes to compute (20 in this case)
X_train_pca = pca.fit_transform(X_train_flattened)

```
Data loaders are created to handle the training data. The PCA-transformed training data and the corresponding labels are combined into a TensorDataset, which is then loaded into a DataLoader with a specified batch size and shuffling.
```
# Create data loaders for training and testing
train_data = torch.utils.data.TensorDataset(torch.from_numpy(X_train_pca), Y_train)
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=64, shuffle=True)
```
The network is initialized, and the loss function is defined as nn.CrossEntropyLoss(). Stochastic Gradient Descent (SGD) optimizer with a learning rate of 0.01 is used.
```
# Initialize the network and define the loss function and optimizer
net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
```
The network is trained for the specified number of epochs. In each epoch, the training data is iterated over in batches. The optimizer is zeroed, a forward pass is performed, the loss is computed, and gradients are backpropagated. The network parameters are updated using the optimizer, and the loss is printed at regular intervals.
```
# Train the network
num_epochs = 10
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, len(train_loader), loss.item()))
```
The trained network is then tested on the test data. The test data is preprocessed similarly to the training data, with the PCA transformation applied to the flattened test images using the already computed PCA object. The test data and labels are combined into a TensorDataset, which is loaded into a DataLoader. The trained network is used to make predictions on the test data, and accuracy is calculated by comparing the predicted labels with the true labels. The accuracy of the network on the test images is printed.
```
# Test the network
test_data = test_dataset.data.float()  # Convert image data to float
X_test_flattened = test_data.view(test_data.size(0), -1)
X_test_pca = pca.transform(X_test_flattened)  # Apply PCA transformation on test data
Y_test = test_dataset.targets

test_data = torch.utils.data.TensorDataset(torch.from_numpy(X_test_pca), Y_test)
test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=64, shuffle=False)

with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.float()  # Convert images to float
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))
```
Overall, this code showcases the process of training a neural network for image classification using the MNIST dataset. It incorporates PCA for dimensionality reduction before training the network and evaluates the network's accuracy on the test set.

### LSTM-based Image Classification using PCA and MNIST Dataset
The code starts by importing the necessary modules and packages.
```
from torch.utils.data import TensorDataset
```
The device is set to 'cuda' if a GPU is available; otherwise, it is set to 'cpu'. This allows for GPU acceleration if possible.
```
# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```
The MNIST dataset is loaded and transformed using torchvision's transforms.Compose. The transformations convert the images to tensors and normalize them.
```
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
```
Two datasets, train_dataset and test_dataset, are created with the transformed MNIST data.
```
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)
```
Principal Component Analysis (PCA) is performed on the dataset. The training and test images are reshaped into vectors and transformed using PCA. The number of components for PCA is set to 20.
```
# Perform PCA on the dataset
train_images = train_dataset.data.reshape(-1, 28 * 28).numpy()
pca = PCA(n_components=20)
train_pca = pca.fit_transform(train_images)

test_images = test_dataset.data.reshape(-1, 28 * 28).numpy()
test_pca = pca.transform(test_images)
```
New datasets, train_pca_dataset and test_pca_dataset, are created using the PCA-transformed data along with the corresponding targets.
```
# Create new datasets with PCA-transformed data
train_pca_dataset = TensorDataset(torch.from_numpy(train_pca).float(), train_dataset.targets)
test_pca_dataset = TensorDataset(torch.from_numpy(test_pca).float(), test_dataset.targets)
```
Model parameters such as input size, hidden size, number of layers, number of classes, batch size, and the number of epochs are defined.
```
# Model parameters
input_size = 20  # PCA component size
hidden_size = 128
num_layers = 2
num_classes = 10
batch_size = 64  # Define the batch size here
num_epochs = 5
```
Data loaders, train_loader and test_loader, are created using the PCA datasets. These data loaders handle batching and shuffling of the data during training and evaluation.
```
# Define data loaders with PCA datasets
train_loader = torch.utils.data.DataLoader(dataset=train_pca_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_pca_dataset, batch_size=batch_size, shuffle=False)
```
The LSTM model architecture is defined by creating a class, LSTMNet, that extends the nn.Module class. It consists of an LSTM layer and a fully connected layer. The forward method defines the forward pass of the model.
```
# Define LSTM model architecture with modified input size
class LSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMNet, self).__init__()
        self.input_size = input_size  # Assign input_size to an attribute
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1, self.input_size)  # Reshape the input tensor
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
```
An instance of the LSTMNet model is created with the specified parameters, and it is moved to the designated device (GPU or CPU) for computation.
```
# Initialize the LSTM model with modified input size
model = LSTMNet(input_size, hidden_size, num_layers, num_classes).to(device)
```
The loss function (nn.CrossEntropyLoss) and optimizer (torch.optim.Adam) are defined for training the model. The optimizer uses the model parameters for optimization during the training process.
```
# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```
The code enters a training loop, where it iterates over the training data in batches. For each batch, it performs a forward pass, computes the loss, performs a backward pass to compute gradients, and updates the model parameters using the optimizer. The loss is printed at regular intervals during training.
```
# Training loop
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')
```
After training, the model is switched to evaluation mode using model.eval(). Then, a test loop is executed, where the model makes predictions on the test data. The predicted labels are compared with the true labels to calculate the accuracy of the model on the test set. The accuracy of the model on the test images is then printed, providing an evaluation of its performance on unseen data.
```
# Evaluation
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy on the test set: {accuracy:.2f}%')
```
In summary, this code performs PCA on the MNIST dataset to reduce its dimensionality, trains an LSTM model using the PCA-transformed data, and evaluates the model's accuracy on the test set.
## Computational Results
### Training and Evaluating a Neural Network for Regression on Two Different Datasets

The least square errors of the first 20 data points as training data and the last 10 points as testing data:
Model                                | Training Set | Test Set
-------------------------------------|--------------|---------
Linear regression                    | 5.03         | 11.31
Quadratic polynomial                 | 4.52         | 75.93
19th degree polynomial               | 3.52         | 750123339825593.60
Three-layer feed forward neural network | 16.37     | 87.49


The least square errors of the first 10 and last 10 data points as training data, and the middle 10 data points as testing data:

Model                                | Training Set | Test Set
-------------------------------------|--------------|---------
Linear model                         | 3.43         | 8.65
Quadratic model                      | 3.43         | 8.44
19th degree polynomial               | 3.65         | 25.42
Three-layer feed forward neural network | 67.03      | 8.65


In the first set of data points, where the first 20 points were used for training and the last 10 points for testing, the Three-layer feed forward neural network performed reasonably well with a training set error of 16.37 and a test set error of 87.49. However, compared to other models such as Linear regression and Quadratic polynomial, the neural network had higher errors on both the training and test sets.

In the second set of data points, where the first 10 and last 10 points were used for training and the middle 10 points for testing, the neural network showed mixed performance. While it had a relatively low test set error of 8.65, which was comparable to the Linear model and Quadratic model, the training set error of 67.03 was noticeably higher. This suggests that the neural network might have overfit the training data to some extent.

Overall, the performance of the Three-layer feed forward neural network varied depending on the data set. It showed relatively higher errors compared to other models in the first data set and had a trade-off between training and test set errors in the second data set. Further analysis and tuning of the neural network's architecture and hyperparameters may be needed to improve its performance and make it more competitive with other models.

### Comparison of Four Models Trained on MNIST Dataset with PCA Dimensionality Reduction
| Model                                 | Accuracy |
|---------------------------------------|----------|
| SVM                                   | 99.56%   |
| Decision Tree                         | 96.87%   |
| Three-layer feed forward neural network | 95.66%   |
| LSTM                                  | 94.98%   |


Comparing the performance of the four different models trained on the MNIST dataset with PCA dimensionality reduction, we can observe the following:

SVM:

The Support Vector Machine (SVM) model stands out as the top-performing method among the four evaluated approaches. Its high accuracy can be attributed to SVM's effectiveness in handling high-dimensional data, which is further enhanced by the dimensionality reduction achieved through PCA. SVM is a robust classifier widely recognized for its excellent performance across diverse datasets, including image classification tasks like the MNIST dataset.

Decision Tree:

The Decision Tree classifier performs well, but slightly lower than the SVM model. Decision Trees are known for their interpretability and ease of understanding, making them suitable for tasks where explainability is important. The dimensionality reduction provided by PCA helps to improve the model's generalization ability, but there is still room for improvement compared to SVM.

Three-layer feedforward neural network:

The feedforward neural network achieves a decent accuracy, but lower than both SVM and Decision Tree models. Neural networks are highly flexible and capable of learning complex patterns, but they typically require a larger amount of data to perform well. The reduced dimensionality from PCA might have affected the neural network's ability to capture intricate patterns, resulting in a slightly reduced performance.

LSTM:

The LSTM model, a type of recurrent neural network (RNN), shows the lowest accuracy among the four methods. LSTMs are widely used for sequence modeling tasks and are effective in capturing long-term dependencies. However, LSTMs typically benefit from longer sequences and a larger input space, which might explain the lower accuracy in this case with reduced dimensionality.

In summary, the SVM model exhibits the highest accuracy, showcasing the benefits of PCA dimensionality reduction in improving the performance of traditional machine learning algorithms. While the Decision Tree model also performs well, the neural network models (both feedforward and LSTM) show slightly lower accuracy, suggesting that they may require more information or a larger input space to achieve their full potential.


## Conclusion

In report, we explored two different analyses. Firstly, we reconsidered the data from [Optimization project](https://github.com/tuongv-1736461/Machine-Learning-projects/tree/main/Optimization), which consisted of X and Y arrays. We applied a three-layer feed-forward neural network to fit the data and evaluated its performance on both training and test sets. Additionally, we compared the results of different training set configurations. The neural network showed mixed performance, with varying errors across different training set configurations. Further analysis and tuning may be necessary to improve its performance and make it more competitive with other models.

Secondly, we trained a feed-forward neural network on the MNIST dataset after performing PCA dimensionality reduction. We compared its results with those of LSTM, SVM, and decision tree classifiers. The SVM model achieved the highest accuracy among the four methods, benefiting from its ability to handle high-dimensional data effectively, combined with the dimensionality reduction achieved through PCA. The decision tree classifier also performed well, while the neural network models showed slightly lower accuracy, indicating the need for more data or a larger input space to fully leverage their capabilities. The LSTM model exhibited the lowest accuracy, likely due to its preference for longer sequences and larger input spaces.

Overall, this report provided valuable insights into the performance of different models in various scenarios. It highlighted the importance of model selection, dimensionality reduction, and training set configurations in achieving accurate predictions. The SVM model demonstrated its effectiveness in image classification tasks like MNIST, while the decision tree model showcased its interpretability. The neural network models showed potential but required further exploration and optimization. 
