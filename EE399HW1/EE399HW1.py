# https://github.com/tuongv-1736461 
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

# Define the model function
def model(x, params):
    A, B, C, D = params
    return A*np.cos(B*x) + C*x + D

# Define the least squares error function
def error(params, x, y):
    return np.sqrt(np.mean((model(x, params) - y)**2))

# Load data
X = np.arange(0,31)
Y = np.array([30, 35, 33, 32, 34, 37, 39, 38, 36, 36, 37, 39, 42, 45, 45, 41,
              40, 39, 42, 44, 47, 49, 50, 49, 46, 48, 50, 53, 55, 54, 53])

# Part (i): Find the minimum error and determine the parameters A, B, C, D
initial_guess = [1, 1, 1, 1]
result = minimize(error, initial_guess, args=(X, Y))
params = result.x
print("Minimum error:", result.fun)
print("A:", params[0])
print("B:", params[1])
print("C:", params[2])
print("D:", params[3])

# Part (ii): Sweep through values of two parameters and generate a 2D loss landscape
A = params[0]
B = params[1]
C = params[2]
D = params[3]

delta_a = 300
delta_b = 1000
delta_c = 500
delta_d = 60

# Sweep through values of A and B
loss_landscape_AB = np.zeros((100, 100))
for i, a in enumerate(np.linspace(A-delta_a, A+delta_a, 100)):
    for j, b in enumerate(np.linspace(B-delta_b, B+delta_b, 100)):
        params = [a, b, C, D]
        loss_landscape_AB[j, i] = error(params, X, Y)
# Generate a plot of the loss landscape
plt.imshow(loss_landscape_AB, cmap='viridis', extent=[A-delta_a, A+delta_a, B-delta_b, B+delta_b], origin='lower', aspect='auto')
plt.xlabel('A')
plt.ylabel('B')
plt.title('Loss Landscape')
plt.colorbar()
plt.show()

# Sweep through values of A and C
plt.figure() # create a new figure
loss_landscape_AC = np.zeros((100, 100))
for i, a in enumerate(np.linspace(A-delta_a, A+delta_a, 100)):
    for j, c in enumerate(np.linspace(C-delta_c, C+delta_c, 100)):
        params = [a, B, c, D]
        loss_landscape_AC[j, i] = error(params, X, Y)
# Generate a plot of the loss landscape
plt.imshow(loss_landscape_AC, cmap='viridis', extent=[A-delta_a, A+delta_a, C-delta_c, C+delta_c], origin='lower', aspect='auto')
plt.xlabel('A')
plt.ylabel('C')
plt.title('Loss Landscape')
plt.colorbar()
plt.show()


# Sweep through values of A and D
loss_landscape_AD = np.zeros((100, 100))
for i, a in enumerate(np.linspace(A-delta_a, A+delta_a, 100)):
    for j, d in enumerate(np.linspace(D-delta_d, D+delta_d, 100)):
        params = [a, B, C, d]
        loss_landscape_AD[j, i] = error(params, X, Y)
# Generate a plot of the loss landscape
plt.imshow(loss_landscape_AD, cmap='viridis', extent=[A-delta_a, A+delta_a, D-delta_d, D+delta_d], origin='lower', aspect='auto')
plt.xlabel('A')
plt.ylabel('D')
plt.title('Loss Landscape')
plt.colorbar()
plt.show()

# Sweep through values of B and C
loss_landscape_BC = np.zeros((100, 100))
for i, b in enumerate(np.linspace(B-delta_b, B+delta_b, 100)):
    for j, c in enumerate(np.linspace(C-delta_c, C+delta_c, 100)):
        params = [A, b, c, D]
        loss_landscape_BC[j, i] = error(params, X, Y)
# Generate a plot of the loss landscape
plt.imshow(loss_landscape_BC, cmap='viridis', extent=[B-delta_b, B+delta_b, C-delta_c, C+delta_c], origin='lower', aspect='auto')
plt.xlabel('B')
plt.ylabel('C')
plt.title('Loss Landscape')
plt.colorbar()
plt.show()

# Sweep through values of B and D
loss_landscape_BD = np.zeros((100, 100))
for i, b in enumerate(np.linspace(B-delta_b, B+delta_b, 100)):
    for j, d in enumerate(np.linspace(D-delta_d, D+delta_d, 100)):
        params = [A, b, C, d]
        loss_landscape_BD[j, i] = error(params, X, Y)
# Generate a plot of the loss landscape
plt.imshow(loss_landscape_BC, cmap='viridis', extent=[B-delta_b, B+delta_b, D-delta_d, D+delta_d], origin='lower', aspect='auto')
plt.xlabel('B')
plt.ylabel('D')
plt.title('Loss Landscape')
plt.colorbar()
plt.show()

# Sweep through values of C and D
loss_landscape_CD = np.zeros((100, 100))
for i, c in enumerate(np.linspace(C-delta_c, C+delta_c, 100)):
    for j, d in enumerate(np.linspace(D-delta_d, D+delta_d, 100)):
        params = [A, B, c, d]
        loss_landscape_CD[j, i] = error(params, X, Y)
# Generate a plot of the loss landscape
plt.imshow(loss_landscape_BC, cmap='viridis', extent=[C-delta_c, C+delta_c, D-delta_d, D+delta_d], origin='lower', aspect='auto')
plt.xlabel('C')
plt.ylabel('D')
plt.title('Loss Landscape')
plt.colorbar()
plt.show()

# Part iii
# Split the data into training and test sets
X_train, Y_train = X[:20].reshape(-1, 1), Y[:20].reshape(-1, 1)
X_test, Y_test = X[20:].reshape(-1, 1), Y[20:].reshape(-1, 1)

# Fit a linear regression model
model_lr = LinearRegression()
model_lr.fit(X_train, Y_train)
Y_lr_train = model_lr.predict(X_train)
Y_lr_test = model_lr.predict(X_test)
err_lr_train = mean_squared_error(Y_train, Y_lr_train)
err_lr_test = mean_squared_error(Y_test, Y_lr_test)

# Fit a quadratic polynomial
poly = PolynomialFeatures(degree=2)
X_poly_train = poly.fit_transform(X_train)
X_poly_test = poly.fit_transform(X_test)
model_quad = LinearRegression()
model_quad.fit(X_poly_train, Y_train)
Y_quad_train = model_quad.predict(X_poly_train)
Y_quad_test = model_quad.predict(X_poly_test)
err_quad_train = mean_squared_error(Y_train, Y_quad_train)
err_quad_test = mean_squared_error(Y_test, Y_quad_test)

# Fit a 19th degree polynomial
poly19 = PolynomialFeatures(degree=19)
X_poly19_train = poly19.fit_transform(X_train)
X_poly19_test = poly19.fit_transform(X_test)
model_poly19 = LinearRegression()
model_poly19.fit(X_poly19_train, Y_train)
Y_poly19_train = model_poly19.predict(X_poly19_train)
Y_poly19_test = model_poly19.predict(X_poly19_test)
err_poly19_train = mean_squared_error(Y_train, Y_poly19_train)
err_poly19_test = mean_squared_error(Y_test, Y_poly19_test)

# Print the errors
print("Least square errors on the training set:")
print(f"Linear regression: {err_lr_train}")
print(f"Quadratic polynomial: {err_quad_train}")
print(f"19th degree polynomial: {err_poly19_train}")

print("\nLeast square errors on the test set:")
print(f"Linear regression: {err_lr_test}")
print(f"Quadratic polynomial: {err_quad_test}")
print(f"19th degree polynomial: {err_poly19_test}")

# Part iv
# Split the data into training and test sets
X_train = np.concatenate((X[:10], X[-10:]))
Y_train = np.concatenate((Y[:10], Y[-10:]))
X_test = X[10:20]
Y_test = Y[10:20]

# Fit a linear model to the training data
lin_reg = LinearRegression()
lin_reg.fit(X_train.reshape(-1, 1), Y_train)
lin_train_error = np.mean((lin_reg.predict(X_train.reshape(-1, 1)) - Y_train)**2)
lin_test_error = np.mean((lin_reg.predict(X_test.reshape(-1, 1)) - Y_test)**2)

# Fit a quadratic model to the training data
poly_reg = PolynomialFeatures(degree=2)
X_poly_train = poly_reg.fit_transform(X_train.reshape(-1, 1))
quad_reg = LinearRegression()
quad_reg.fit(X_poly_train, Y_train)
quad_train_error = np.mean((quad_reg.predict(X_poly_train) - Y_train)**2)
X_poly_test = poly_reg.transform(X_test.reshape(-1, 1))
quad_test_error = np.mean((quad_reg.predict(X_poly_test) - Y_test)**2)

# Fit a 19th degree polynomial to the training data
poly_reg = PolynomialFeatures(degree=19)
X_poly_train = poly_reg.fit_transform(X_train.reshape(-1, 1))
poly_reg_model = LinearRegression()
poly_reg_model.fit(X_poly_train, Y_train)
poly_train_error = np.mean((poly_reg_model.predict(X_poly_train) - Y_train)**2)
X_poly_test = poly_reg.transform(X_test.reshape(-1, 1))
poly_test_error = np.mean((poly_reg_model.predict(X_poly_test) - Y_test)**2)


# Print the results
print("Least square errors on the training set:")
print(f"Linear model: {lin_train_error:.2f}")
print(f"Quadratic model: {quad_train_error:.2f}")
print(f"19th degree polynomial: {poly_train_error:.2f}")

print("\nLeast square errors on the test set:")
print(f"Linear model: {lin_test_error:.2f}")
print(f"Quadratic model: {quad_test_error:.2f}")
print(f"19th degree polynomial: {poly_test_error:.2f}")