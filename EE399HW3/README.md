
# Exploring MNIST with SVD and Machine Learning: LDA, SVM, and Decision Trees




## Author

- [Jenny Van](https://github.com/tuongv-1736461)



## Abstract

In this report, we perform an analysis of the MNIST data set, which consists of images of handwritten digits. We start by using Singular Value Decomposition (SVD) to analyze the images and determine the necessary number of modes for good image reconstruction. We interpret the U, Σ, and V matrices obtained from the SVD analysis. Then, we project the data onto PCA space and build a classifier to identify individual digits in the training set. We use Linear Discriminant Analysis (LDA) to classify two and three digits and quantify the accuracy of the separation with LDA on the test data. We compare the performance of LDA, SVM, and decision trees on the hardest and easiest pair of digits to separate. 
## Introduction
The MNIST data set consists of 70,000 images of handwritten digits. In this report, we aim to analyze this data set and build a classifier to identify individual digits. To do this, we first perform an SVD analysis of the digit images. We reshape each image into a column vector. We then examine the singular value spectrum obtained from the SVD analysis to determine the necessary number of modes for good image reconstruction, which corresponds to the rank r of the digit space. We interpret the U, Σ, and V matrices obtained from the SVD analysis and use them to project the data onto PCA space.

We then proceed to build a classifier to identify individual digits in the training set. We start by using LDA to classify two digits and then move on to three digits. We quantify the accuracy of the separation with LDA on the test data and compare it with the performance on the training set. We also investigate which two digits are the most and the least difficult to separate and quantify the accuracy of the separation with LDA on the test data. Finally, we compare the performance of LDA, SVM, and decision trees on the hardest and easiest pair of digits to separate. 

## Theoretical Background

#### Understanding Singular Value Decomposition (SVD) in Matrix Factorization for Dimensionality Reduction

Singular Value Decomposition (SVD) is a powerful technique used in matrix factorization that breaks down a matrix A into three matrices: U, Σ, and V. U is a square orthogonal matrix that contains the left singular vectors representing the principal components of the data. Σ is a diagonal matrix containing the singular values of A, which represent the amount of variance captured by each principal component. V is another square orthogonal matrix that contains the right singular vectors used for reconstructing the data. The singular value spectrum is a graph showing the importance of each singular value of A, sorted in descending order, and can be used to determine the necessary number of modes for good image reconstruction. U, Σ, and V matrices have specific interpretations, where U is used for projecting the data onto a lower-dimensional space (PCA space). PCA spaces refer to the space formed by the principal components of the data, which are orthogonal to each other and ordered by the amount of variance they capture, with the first principal component capturing the most variance.

#### Classification Techniques and Performance Evaluation

Linear Discriminant Analysis (LDA) is a classification technique that finds a linear combination of features to maximally separate the classes by minimizing the within-class variance and maximizing the between-class variance. It is assumed that the data follows a Gaussian distribution and that the covariance matrices of the classes are equal.

Support Vector Machines (SVM) is a classification technique that finds the hyperplane which maximally separates the classes by identifying support vectors, i.e., points closest to the hyperplane. SVM can handle non-linearly separable data by using kernel functions to map the data into a higher-dimensional space.

Decision trees are a classification technique that splits the data based on a set of rules. Each internal node of the tree represents a feature, and each leaf node represents a class label. The tree is constructed by recursively splitting the data based on the feature that maximally separates the classes.

To evaluate the performance of classifiers, it is necessary to compute their accuracy on both the training and test sets. The training set is used to train the classifier, while the test set is used to evaluate its performance on unseen data. Evaluating the performance on the test set ensures that the classifier generalizes well to new data.

## Algorithm Implementation and Development
First, we import the necessary libraries. Next, we load the data by fetching the mnist_784 dataset using fetch_openml function and store the data in X and the corresponding labels in y. To ensure compatibility with our models, we convert the labels to integers using the astype function.

After that, we split the data into training and testing sets using the train_test_split function from sklearn.model_selection. We allocate 80% of the data for training and 20% for testing. The resulting training data is stored in X_train and y_train, while the testing data is stored in X_test and y_test.

```
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from itertools import combinations
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
mnist = fetch_openml('mnist_784')
X = mnist.data
y = mnist.target.astype(int)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```
#### SVD analysis

The code first reshapes the image data into column vectors, then performs SVD on the data to obtain U, S, and V. We then plot the singular value spectrum, which visualizes the strength of each mode in the dataset. The total percentage of variance captured by the first 50 modes is calculated using a formula involving the singular values. This helps us determine the number of modes to use for projecting the data to a lower-dimensional space.

```
# Reshape the images into column vectors
X = X.T

# Compute the SVD of the data matrix
U, S, V = np.linalg.svd(X, full_matrices=False)

# Plot the singular value spectrum
plt.figure(figsize=(8,6))
plt.plot(S)
plt.xlabel('Singular Value Index')
plt.ylabel('Singular Value')
plt.title('Singular Value Spectrum')
plt.show()

# Calculate total percentage of variance
n_modes = 50
variance_captured = [(S[i]**2 / np.sum(S**2)) * 100 for i in range(n_modes)]
print("Total percentage of variance captured by the first 50 modes: %f" %np.sum(variance_captured))
``` 
We select three columns of the left singular vectors (V) obtained from SVD (columns 1, 2, and 3) and use them to project the data onto a 3D space. 

```
from mpl_toolkits.mplot3d import Axes3D
# Select three columns of V
V_selected = U[:, [0, 1, 2]]

# Project the data onto the selected V-modes
x_projected = np.dot(V_selected.T, X)

# Create a 3D scatter plot with colored points
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_projected[0,:], x_projected[1,:], x_projected[2,:], c=y.astype(int))
ax.set_xlabel('V1')
ax.set_ylabel('V2')
ax.set_zlabel('V3')
ax.set_title('Project the data onto the selected V-mode 1, 2, and 3')
plt.show()
``` 
In this code, we use the Scikit-learn library's PCA function to perform dimensionality reduction on the data by selecting the top 50 principal components. The PCA function fits the model on the data and transforms it to its principal components. Then, we plot the first two principal components against each other using a scatter plot, with the color of each point representing its corresponding class label. Finally, we add a color bar to the plot to display the color labels.

```
from sklearn.decomposition import PCA
# Apply PCA to reduce the dimensionality of the data
pca = PCA(n_components=50)
X_pca = pca.fit_transform(X)

# Plot the first two principal components
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=mnist.target[:X_pca.shape[0]].astype(int), s=8)
plt.colorbar()
plt.title('The First Two Principal Components of PCA')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()
```

This code creates a 3D scatter plot using the first three principal components of the data. The color of each point in the plot is determined by the corresponding target value from the MNIST dataset. 

```
# Create the 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=mnist.target[:X_pca.shape[0]].astype(int), s=20)

# Set the labels and limits of the plot
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.set_title('Plot of data projected onto the first three principal components')
plt.show()
```
#### Classification Techniques and Performance Evaluation

We start by using PCA to project the data onto a 50-mode space
```
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
```
We then create a function to train and test classifiers on digit pairs. The function takes in a list of digit pairs and a classifier. For each digit pair, the function gets the indices of the current pair in the training set, creates the feature and target datasets for the current digit pair, trains the classifier on the training data, predicts the labels of the test data, and calculates the accuracy of the classifier. The average accuracy for all digit pairs is calculated and printed. The pair with the lowest accuracy and highest accuracy is also identified and printed. 
```
def test_digit_pairs(digit_pairs, classifier):
    accuracy_scores = []
    
    for digit_pair in digit_pairs:
        # Get the indices for the current digit pair
        digit_indices = ((y_train == digit_pair[0]) | (y_train == digit_pair[1]))

        # Create the feature and target datasets for the current digit pair
        X_pair_train = X_train_pca[digit_indices]
        y_pair_train = y_train[digit_indices]
        X_pair_test = X_test_pca[((y_test == digit_pair[0]) | (y_test == digit_pair[1]))]
        y_pair_test = y_test[((y_test == digit_pair[0]) | (y_test == digit_pair[1]))]

        # Train the classifier on the training data
        clf = classifier
        clf.fit(X_pair_train, y_pair_train)

        # Predict the labels of the test data
        y_pred = clf.predict(X_pair_test)

        # Calculate the accuracy of the classifier
        accuracy = accuracy_score(y_pair_test, y_pred)

        # Append the accuracy to the list of accuracy scores
        accuracy_scores.append(accuracy)

    # Calculate the average accuracy for all digit pairs
    avg_accuracy = sum(accuracy_scores) / len(accuracy_scores)
    print(f"Average accuracy for all digit pairs: {avg_accuracy:.5f}")

    # Find the pair with the lowest accuracy
    most_difficult_pair = digit_pairs[np.argmin(accuracy_scores)]
    lowest_accuracy = min(accuracy_scores)
    print(f"The most difficult pair to separate is {most_difficult_pair} with an accuracy of {lowest_accuracy:.5f}.")

    # Find the pair with the highest accuracy
    least_difficult_pair = digit_pairs[np.argmax(accuracy_scores)]
    highest_accuracy = max(accuracy_scores)
    print(f"The least difficult pair to separate is {least_difficult_pair} with an accuracy of {highest_accuracy:.5f}.")
```
We then test three different classifiers, LDA, SVM, and decision tree, on all digit pairs. The digit pairs are created using the combinations function and a range of numbers from 0 to 9. Finally, the test_digit_pairs function is called for each classifier and the results are printed.

```
# Test the LDA classifier on all digit pairs
lda_pairs = list(combinations(np.unique(y), 2))
lda_classifier = LDA()
print("LDA")
test_digit_pairs(lda_pairs, lda_classifier)

# Test the SVM classifier on all digit pairs using 50 modes PCA
svm_pairs = list(combinations(range(10), 2))
svm_classifier = SVC()
print("SVM")
test_digit_pairs(svm_pairs, svm_classifier)

# Test the decision tree classifier on all digit pairs
dt_pairs = list(combinations(range(10), 2))
dt_classifier = DecisionTreeClassifier(random_state=42)
print("Decision Tree")
test_digit_pairs(dt_pairs, dt_classifier)
```
In this implementation, we are testing the previously trained classifiers (SVM, Decision Tree, and LDA) on a specific pair of digits, which are (3, 5) and (6, 9). These pairs were identified earlier as the most difficult and least difficult pairs to separate using the LDA classifier. We pass the digit pairs and the classifiers to the function test_digit_pairs(), which performs the testing process for each digit pair and classifier combination. We print out the average accuracy for all digit pairs, and the accuracy of each classifier for the given digit pairs.
```
two_digit_pairs = [(3, 5), (6, 9)]
print("SVM")
test_digit_pairs(two_digit_pairs, svm_classifier)
print("Decision Tree")
test_digit_pairs(two_digit_pairs, dt_classifier)
print("LDA")
test_digit_pairs(two_digit_pairs, lda_classifier)
```
We select three digits (2, 3, and 5) and trains an LDA classifier on the training samples of those digits. Then, we make predictions on the test data, evaluate the performance of the model, and print the results.
```
# Define the digits to include in the classification task
digits = [2, 3, 5]

# Extract the samples for the selected digits from the training data
X_train_digits = X_train[np.isin(y_train, digits)]
y_train_digits = y_train[np.isin(y_train, digits)]

# Train the LDA classifier on the selected training samples
lda = LDA()
lda.fit(X_train_digits, y_train_digits)

# Extract the test samples for the selected digits
X_test_digits = X_test[np.isin(y_test, digits)]
y_test_digits = y_test[np.isin(y_test, digits)]

# Make predictions on the test data
y_pred = lda.predict(X_test_digits)

# Evaluate the performance of the model
acc = accuracy_score(y_test_digits, y_pred)

# Print the results
print(f"Selected digits: {digits}")
print(f"Accuracy: {acc}")
```


## Results
#### SVD analysis
![download1](https://github.com/tuongv-1736461/EE399/blob/main/EE399HW3/download1.png)

The total percentage of variance captured by the first 50 modes is 89.528712

![download2](https://github.com/tuongv-1736461/EE399/blob/main/EE399HW3/download2.png)

![download3](https://github.com/tuongv-1736461/EE399/blob/main/EE399HW3/download3.png)

![download4](https://github.com/tuongv-1736461/EE399/blob/main/EE399HW3/download4.png)

#### Classification Techniques and Performance Evaluation
We tested three different classifiers, LDA, SVM, and decision tree, on all digit pairs. The results are shown in the table below.

| Model           | Average Accuracy | Most Difficult Pair | Least Difficult Pair |
|----------------|-----------------|---------------------|----------------------|
| LDA            | 0.98090         | (3, 5) - 0.94463     | (6, 9) - 0.99720      |
| SVM            | 0.99714         | (4, 9) - 0.99103     | (1, 4) - 1.00000      |
| Decision Tree  | 0.96731         | (4, 9) - 0.89164     | (0, 1) - 0.99536      |

We tested three different classifiers on the most and least difficult pairs identified earlier using the LDA classifier.
| Digits Pair | LDA Accuracy | SVM Accuracy | Decision Tree Accuracy |
| ----------- | ------------ | ------------ | ---------------------- |
| (3, 5)      | 0.94463      | 0.99360      | 0.93070                |
| (6, 9)      | 0.99720      | 0.99930      | 0.98668                |

We evaluated our LDA classification model on a specific set of digits, which were [2, 3, 5]. The accuracy achieved by the model was 0.9388997.

## Conclusion 
#### SVD analysis
In conclusion, by analyzing the singular value spectrum obtained from the SVD analysis, we were able to determine the optimal number of modes required for high-quality image reconstruction, which corresponds to the rank r of the digit space. Through the interpretation of the U, Σ, and V matrices obtained from the SVD analysis, we were able to project the data onto PCA space. This allowed us to gain a deeper understanding of the underlying structure of the data and make more informed decisions when applying machine learning techniques to it. Overall, the use of SVD and PCA provides a powerful tool for data analysis and dimensionality reduction in machine learning applications.

#### Classification Techniques and Performance Evaluation
After reducing the dimensionality of the data to 50 principal components, we were still able to achieve high accuracy during classification. Our comprehensive evaluation of three different classifiers, namely LDA, SVM, and decision tree, on all digit pairs revealed that SVM outperforms the other two classifiers with an impressive average accuracy of 0.99714. It is noteworthy, however, that the most and least difficult pairs of digits varied across the different classifiers. Further analysis of the most and least difficult digit pairs identified using the LDA classifier demonstrated that SVM performed exceptionally well on both pairs. In conclusion, our findings indicate that SVM is the most accurate classifier among the three evaluated and is consistently reliable across different digit pairs for this particular dataset.
