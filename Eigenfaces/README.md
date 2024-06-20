
# Application of Machine Learning: Eigenfaces

## Abstract:
In this project, we work with a dataset of 2414 grayscale images of 39 different faces, each downsampled to 32x32 pixels and stored as columns in a matrix X of size 1024x2414. We perform various operations on the dataset, including computing correlation matrices, finding eigenvectors and principal component directions, and calculating the percentage of variance captured by the first six SVD modes. By analyzing the correlation matrices, we identify the most and least correlated images. Additionally, we obtain the eigenfaces which capture the most significant variations in the dataset. This project provides an introduction to applications of machine learning in facial recognition.


## Author

- [Jenny Van](https://github.com/tuongv-1736461)


## Introduction:
We will begin by computing a correlation matrix of the first 100 images in X and then plotting it. From this correlation matrix, we will identify the most and least correlated images and plot their corresponding faces.  

Next, we will compute a 10x10 correlation matrix between images and plot it. We will also create a matrix Y=XX^T and find the first six eigenvectors with the largest magnitude eigenvalue. Additionally, we will perform SVD on matrix X and find the first six principal component directions. We will compare the first eigenvector from the eigenvectors with the first SVD mode and compute the norm of their absolute values.

Finally, we will compute the percentage of variance captured by each of the first six SVD modes and plot the first six SVD modes. Through this project, we aim to gain a deeper understanding of the dataset and develop skills in computing correlation matrices, finding eigenvectors and principal component directions, and calculating the percentage of variance captured by SVD modes.

## Theoretical Background:

Correlation matrix:

A correlation matrix is a square matrix that shows the pairwise correlations between variables. In this project, we compute the correlation matrix between the first 100 images of the dataset by computing the dot product (correlation) between each pair of images. The diagonal of the correlation matrix contains the variance measures of each image, while the off-diagonal elements are symmetric and represent the covariance between all pairs. A small covariance value indicates that the pair of images are statistically independent, while a large covariance value indicates that they are dependent, and the data may be redundant. By examining the off-diagonal values, we can identify the most and least correlated images in the dataset.

Eigendecomposition of AA.T:

The eigendecomposition of the product of a matrix A and its transpose A.T is equivalent to the singular value decomposition (SVD) of A. Specifically, if we take A = UΣV.T, then AA.T = UΣ^2U.T, where U is an orthogonal matrix containing the eigenvectors of AA.T and Σ contains the singular values. U is given by the decomposition of the product of a matrix A and its transpose A.T 

Percentage of variance and eigenfaces:

Principal component analysis (PCA) is a technique that uses eigenvectors to reduce the dimensionality of data by projecting it onto a lower-dimensional space. In image processing, the set of orthogonal vectors obtained from performing an SVD on an image matrix are called eigenfaces. The first eigenface captures the largest amount of variance in the data, and each subsequent eigenface captures progressively less variance.

By computing the percentage of total variance captured by each eigenface, we can understand how much information is retained when we reduce the dimensionality of the data by keeping only a subset of the eigenfaces. The sum of the percentage of variance of the first k eigenfaces gives us the percentage of total variance captured by the first k components.

## Algorithm Implementation and Development

First, we begin by import the dataset of 2414 gray scale images of size 32x32 pixels and store it in a matrix X of size 1024x2414. 
``` 
results=loadmat('yalefaces.mat')
X=results['X'] 
```
#### Part a: The correlation matrix between the first 100 images in matrix X 
Part (a) of the project involves computing a 100x100 correlation matrix C by computing the dot product (correlation) between the first 100 images in the matrix X. The code does this by extracting the first 100 columns of X and then computing their dot product (correlation) to get the 100x100 correlation matrix C. The code then plots the correlation matrix using the imshow function from the matplotlib.pyplot library. 
```
# extract the first 100 columns of X
X_100 = X[:, :100]

# compute the correlation matrix
C = np.dot(X_100.T, X_100)

# plot the correlation matrix
plt.imshow(C)
plt.colorbar()
plt.title("Correlation matrix of the first 100 columns of X")
plt.show()
```
#### Part b: The two most and least correlated images 
Part (b) of the project requires identifying the two most highly correlated images and the two most uncorrelated images from the correlation matrix computed in part (a), and plotting these images. 

To find the most highly correlated images, the code creates a copy of the correlation matrix C called C_copy, sets the diagonal elements to 0, and finds the location of the maximum value in the matrix using np.argmax(). 
```
# Create a copy of the correlation matrix C
C_copy = np.copy(C)

# Set the diagonal elements to 0
np.fill_diagonal(C_copy, 0)

# Find the location of the maximum value in the matrix
max_loc = np.unravel_index(np.argmax(C_copy), C_copy.shape)
```

To find the most uncorrelated images, the same steps are followed, but the diagonal elements are set to 1 million to find the location of the minimum value in off diagonal of the matrix.
```
# Fill the diagonal to a large number (1 million) 
np.fill_diagonal(C_copy, 1e6)

# Find the location of the minimum value in the matrix
min_loc = np.unravel_index(np.argmin(C_copy), C_copy.shape)
```

Finally, the function graph_img is used to plot the most and least correlated images found in the previous steps. It takes in two indices i and j corresponding to the two images to plot, and a string indicating whether they are the most or least correlated. The function reshapes the 1D array representing each image into a 2D array of dimensions 32x32 and plots them side-by-side using matplotlib. 

```
# Function that plot the image
def graph_img(i, j, string): 
    # Reshape images 1D array into a 2D array with dimensions 32x32
    img1 = X[:, i].reshape(32, 32)
    img2 = X[:, j].reshape(32, 32)    
    # Plot images
    fig, ax = plt.subplots(1, 2)
    fig.suptitle('The Two {} Correlated Images'.format(string), fontsize=16)
    ax[0].imshow(img1, cmap='gray')
    ax[0].set_title('Image %d' % (i+1))
    ax[1].imshow(img2, cmap='gray')
    ax[1].set_title('Image %d' % (j+1))
    plt.show()

# Plot the most and least correlated faces
graph_img(max_loc[0], max_loc[1], "Most")
graph_img(min_loc[0], min_loc[1], "Least")
```
#### Part c: The correlation matrix between the 10 images in matrix X [1, 313, 512, 5, 2400, 113, 1024, 87, 314, 2005].
Part (c) requires computing the 10x10 correlation matrix between images [1, 313, 512, 5, 2400, 113, 1024, 87, 314, 2005]. To extract the images, we subtract 1 from each index to account for 0-indexing and then plotting the correlation matrix between them. 
```
# extract 10 columns of X
index_10 = [1, 313, 512, 5, 2400, 113, 1024, 87, 314, 2005]
index_10 = [i-1 for i in index_10] # subtract 1 from each index to account for 0-indexing
X_10 = X[:, index_10]
```
The correlation matrix can be compute and plot by repeating part (a) 
```
# compute the correlation matrix
C_10 = np.dot(X_10.T, X_10)

# plot the correlation matrix
plt.imshow(C_10)
plt.colorbar()
plt.title("Correlation matrix of 10 faces from X")
plt.show()
```
#### Part d: The first six eigenvectors with the largest magnitude eigenvalue of matrix Y 
Part (d) requires to computes the matrix Y as the product of X and its transpose X.T and obtains six eigenvectors with the largest magnitude eigenvalues. The eigenvalue decomposition of Y is computed by using the numpy.linalg.eigh function, which returns the eigenvalues and eigenvectors of the symmetric matrix Y. The eigenvalues and eigenvectors are then sorted in descending order to get the six eigenvectors with the largest magnitude eigenvalues.
```
# Compute matrix Y
Y = np.dot(X, X.T)

# compute the eigenvalue decomposition of Y
eigenvalues, eigenvectors = np.linalg.eigh(Y)

# sort eigenvalues and eigenvectors in descending order
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:,idx]

print("The first six eigenvectors with the largest magnitude eigenvalue:")
print(eigenvectors[:,:6])
```
#### Part e: The first six principal component directions matrix X
Part (e) uses the numpy.linalg.svd function to perform singular value decomposition (SVD) of the matrix X. The function returns three matrices: U, S, and V. The first six principal component directions are printed by extracting the first six columns of the matrix U. 
```
U, S, V = np.linalg.svd(X)
# print the resulting matrices
print("The first six principal component directions:")
print(U[:,:6])
```
#### Part f: The norm of difference of the absolute values between the first eigenvector v1 from (d) and the first SVD mode u1 from (e)
In part (f), the first eigenvector v1 from part (d) and the first SVD mode u1 from part (e) are compared. The absolute values of the two vectors are subtracted from each other, and then the norm of the resulting difference is computed. 
```
# eigenvector v1 from part (d)
v1 = np.array(eigenvectors[:,0])

# SVD mode u1 from part (e)
u1 = np.array(U[:,0])

# compute the norm of the difference of their absolute values
diff_norm = np.linalg.norm(np.abs(v1) - np.abs(u1))
print("U1:")
print(u1)
print("V1:")
print(v1)
# print the result
print("Norm of difference:", diff_norm)
```
#### Part g: The percentage of variance captured by each of the first 6 SVD modes
Part (g) computes and prints the percentage of variance captured by each of the first six SVD modes of the matrix X, which provides a measure of how much of the overall variability in the dataset is captured by each mode. It then plots the first six SVD modes, which are referred to as "Eigenfaces" in the context of facial recognition.

To compute the percentage of variance captured by each mode, the code first sets the number of modes to consider to six (n_modes = 6), then uses a list comprehension to compute the variance captured by each mode. This is done by taking the square of the singular values (contained in the vector S) for each mode, dividing by the sum of the squares of all singular values, and multiplying by 100 to get a percentage. The resulting percentages are printed to the console using a for loop.

```
# Compute the percentage of variance captured by the first 6 modes
n_modes = 6
variance_captured = [(S[i]**2 / np.sum(S**2)) * 100 for i in range(n_modes)]
print("Percentage of variance captured by the first %d modes:" % n_modes)
for i in range(n_modes):
   print("Mode %d: %.2f%%" % (i+1, variance_captured[i]))
print (np.sum(variance_captured))
```
Next, the six eigenfaces are plotted by repeating part a
```
# Plot images
fig, ax = plt.subplots(1, 6, figsize=(16, 4))
fig.suptitle('The Six Eigenfaces', fontsize=16)
for i in range(n_modes):
    img = U[:, i].reshape(32, 32)
    ax[i].imshow(img, cmap='gray')
    ax[i].set_title('Image %d' % (i+1))
plt.show()
```


## Computational Results
#### Part a: The correlation matrix between the first 100 images in matrix X 

![C_matrix_100](https://user-images.githubusercontent.com/104536898/233248514-0014cc76-d947-4fe1-a30e-df5b989bd63d.png)


#### Part b: The two most and least correlated images 

![most_correlated](https://user-images.githubusercontent.com/104536898/233248419-26b13c27-57bc-4c64-bf37-5693f192c0ca.png)
![least_correlated](https://user-images.githubusercontent.com/104536898/233248402-65a06995-3855-4e2e-968c-db9881c475c7.png)

#### Part c: The correlation matrix between the 10 images in matrix X [1, 313, 512, 5, 2400, 113, 1024, 87, 314, 2005]. 

![C_matrix_10](https://user-images.githubusercontent.com/104536898/233248351-efaca886-3603-4eaf-837c-40aa6ea0bd93.png)


#### Part d: The first six eigenvectors with the largest magnitude eigenvalue of matrix Y 

```
The first six eigenvectors with the largest magnitude eigenvalue:
[[-0.02384327  0.04535378 -0.05653196  0.04441826 -0.03378603  0.02207542]
 [-0.02576146  0.04567536 -0.04709124  0.05057969 -0.01791442  0.03378819]
 [-0.02728448  0.04474528 -0.0362807   0.05522219 -0.00462854  0.04487476]
 ...
 [-0.02082937 -0.03737158 -0.06455006 -0.01006919  0.06172201  0.03025485]
 [-0.0193902  -0.03557383 -0.06196898 -0.00355905  0.05796353  0.02850199]
 [-0.0166019  -0.02965746 -0.05241684  0.00040934  0.05757412  0.00941028]]
```

#### Part e: The first six principal component directions matrix X
```
The first six principal component directions:
[[-0.02384327 -0.04535378 -0.05653196  0.04441826 -0.03378603  0.02207542]
 [-0.02576146 -0.04567536 -0.04709124  0.05057969 -0.01791442  0.03378819]
 [-0.02728448 -0.04474528 -0.0362807   0.05522219 -0.00462854  0.04487476]
 ...
 [-0.02082937  0.03737158 -0.06455006 -0.01006919  0.06172201  0.03025485]
 [-0.0193902   0.03557383 -0.06196898 -0.00355905  0.05796353  0.02850199]
 [-0.0166019   0.02965746 -0.05241684  0.00040934  0.05757412  0.00941028]]
 ```

#### Part f: The norm of difference of the absolute values between the first eigenvector v1 from (d) and the first SVD mode u1 from (e) 

```
U1:
[-0.02384327 -0.02576146 -0.02728448 ... -0.02082937 -0.0193902
 -0.0166019 ]
V1:
[-0.02384327 -0.02576146 -0.02728448 ... -0.02082937 -0.0193902
 -0.0166019 ]
Norm of difference: 7.650346874731423e-16
```

#### Part g: The percentage of variance captured by each of the first 6 SVD modes
```
Percentage of variance captured by the first 6 modes:
Mode 1: 72.93%
Mode 2: 15.28%
Mode 3: 2.57%
Mode 4: 1.88%
Mode 5: 0.64%
Mode 6: 0.59%
Total percentage of variance captured by the first six modes: 93.885337%
```

![eigenfaces](https://user-images.githubusercontent.com/104536898/233248124-c9635531-f366-4acc-a778-b07691fbe9e4.png)

## Conclusion

#### Correlation matrix

This project demonstrated how the correlation matrix can be used to analyze relationships among images. The plots of the correlation matrix in parts a and c revealed that the diagonal elements corresponded to the variance measures and were represented by the lightest color, whereas off-diagonal values were darker, indicating lower correlation. In part b, we used the correlation matrix to identify the most and least correlated images. Specifically, the image pairs with the highest and lowest correlation values were found by examining the maximum and minimum off-diagonal elements of the correlation matrix. These results demonstrate the usefulness of the correlation matrix for exploring relationships between variables in complex datasets.

#### Comparison of Eigenvectors and Principal Component Directions 
The results obtained from part (d) and part (e) indicate that the first six eigenvectors with the largest magnitude eigenvalue of the matrix Y = X*X.T are equivalent to the first six principal component directions of the SVD of the matrix X. This was confirmed by comparing the output and computing the norm of difference of the absolute values between the eigenvector with the largest magnitude eigenvalue and the first principal component directions of the SVD of the matrix X. The similarity between the two sets of vectors suggests that they capture the same underlying structure in the data. This observation is consistent with the theory of principal component analysis, which states that the eigenvectors of the covariance matrix (in this case, Y) correspond to the principal axes of variation in the data, while the SVD provides an alternative method for computing these vectors. 

#### Eigenfaces 

In part (g), we explored the percentage of variance captured by the first six SVD modes, which correspond to the six eigenfaces. The total percentage of variance captured by these modes was found to be 93.885337%. This result suggests that the first six modes can capture a considerable amount of the total variance in the dataset. As such, they provide a good representation of the original images. Overall, the high percentage of variance captured by the first six modes indicates that these eigenfaces are a powerful tool for analyzing and understanding the underlying structure of the dataset.
