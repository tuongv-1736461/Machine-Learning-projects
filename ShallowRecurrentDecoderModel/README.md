
# Analysis of Sea-Surface Temperature using Shallow Recurrent Decoder model: Performance Evaluation based on Time Lag, Noise, and Number of Sensors

## Author
- [Jenny Van](https://github.com/tuongv-1736461)

## Abstract

This report presents an analysis of sea-surface temperature data using an Shallow Recurrent Decoder model (SHRED). The objectives were to train the model, plot the results, and assess its performance based on time lag, noise, and the number of sensors. The model was trained using provided code and data, and the results were visualized. Performance analysis revealed the impact of varying time lag and noise on the model's accuracy. Additionally, the influence of the number of sensors on performance was examined. The findings contribute to understanding the capabilities and limitations of the LSTM/decoder model for sea-surface temperature analysis.

## Introduction 

This report explores the application of SHRED (SHallow REcurrent Decoder) models for reconstructing sea-surface temperature using the NOAA Optimum Interpolation SST V2 dataset. SHRED models integrate a Long Short-Term Memory (LSTM) network with a shallow decoder network, enabling the reconstruction of complex spatio-temporal fields from sparse sensor measurements.

Our study aims to evaluate the performance of SHRED models in sea-surface temperature reconstruction and analyze the impact of time lag, noise, and the number of sensors. By gaining insights into the effectiveness of SHRED models, we can enhance the reconstruction of high-dimensional spatio-temporal fields for applications like weather forecasting, climate modeling, and environmental monitoring.

## Theoretical Background

Sensing plays a crucial role in science and engineering tasks such as system identification, control decisions, and forecasting. However, limited sensors, noisy measurements, and corrupt data pose challenges. Existing techniques rely on current sensor measurements and require careful sensor placement or a large number of random sensors.

In contrast, the SHallow REcurrent Decoder (SHRED) neural network offers an alternative approach. It combines a recurrent neural network (LSTM) to capture temporal dynamics and a shallow decoder to reconstruct high-dimensional fields. By considering the trajectory of sensor measurements, SHRED enables accurate reconstructions with fewer sensors, outperforming existing methods when more measurements are available and allowing flexible sensor placement.

SHRED provides a compressed representation of the high-dimensional state directly from sensor measurements, facilitating on-the-fly compression in modeling physical and engineering systems. Additionally, it enables efficient forecasting based solely on sensor time-series data, even with a limited number of sensors.

In various applications, including turbulent flows and complex spatio-temporal dynamics, SHRED effectively characterizes dynamics with a minimal number of randomly placed sensors, ensuring high performance.
## Algorithm Implementation and Development

The code begins by importing the necessary libraries and modules, and load the data. 

```
import numpy as np
from processdata import load_data
from processdata import TimeSeriesDataset
from processdata import load_full_SST
import models
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

load_X = load_data('SST')
n = load_X.shape[0]
m = load_X.shape[1]
```
The function "run_shred" serves as the structure for training and testing the SHRED model. It accepts the number of sensors and the number of time lags as input parameters and provides the test errors, as well as the reshaped reconstruction images and the reshaped ground truth images

Inside the function, the sensor locations are randomly selected, and the data is divided into training, validation, and test sets. The data is then preprocessed using MinMaxScaler.
```
def run_shred(num_sensors, lags):

    sensor_locations = np.random.choice(m, size=num_sensors, replace=False)

    # We now select indices to divide the data into training, validation, and test sets.
    train_indices = np.random.choice(n - lags, size=1000, replace=False)
    mask = np.ones(n - lags)
    mask[train_indices] = 0
    valid_test_indices = np.arange(0, n - lags)[np.where(mask != 0)[0]]
    valid_indices = valid_test_indices[::2]
    test_indices = valid_test_indices[1::2]

    # sklearn's MinMaxScaler is used to preprocess the data for training
    sc = MinMaxScaler()
    sc = sc.fit(load_X[train_indices])
    transformed_X = sc.transform(load_X)
```
Input sequences for the SHRED model are generated based on the selected sensor measurements. Training, validation, and test datasets are created for reconstructing states and forecasting sensors.
```
    # Generate input sequences to a SHRED model
    all_data_in = np.zeros((n - lags, lags, num_sensors))
    for i in range(len(all_data_in)):
        all_data_in[i] = transformed_X[i:i+lags, sensor_locations]

    # Generate training, validation, and test datasets for reconstruction of states and forecasting sensors
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_data_in = torch.tensor(all_data_in[train_indices], dtype=torch.float32).to(device)
    valid_data_in = torch.tensor(all_data_in[valid_indices], dtype=torch.float32).to(device)
    test_data_in = torch.tensor(all_data_in[test_indices], dtype=torch.float32).to(device)

    # -1 to have output at the same time as final sensor measurements
    train_data_out = torch.tensor(transformed_X[train_indices + lags - 1], dtype=torch.float32).to(device)
    valid_data_out = torch.tensor(transformed_X[valid_indices + lags - 1], dtype=torch.float32).to(device)
    test_data_out = torch.tensor(transformed_X[test_indices + lags - 1], dtype=torch.float32).to(device)

    train_dataset = TimeSeriesDataset(train_data_in, train_data_out)
    valid_dataset = TimeSeriesDataset(valid_data_in, valid_data_out)
    test_dataset = TimeSeriesDataset(test_data_in, test_data_out)
```
The SHRED model is instantiated with the specified parameters, and the training and validation datasets are used for training. 
```
    # Train the model using the training and validation datasets
    shred = models.SHRED(num_sensors, m, hidden_size=64, hidden_layers=2, l1=350, l2=400, dropout=0.1).to(device)
    validation_errors = models.fit(shred, train_dataset, valid_dataset, batch_size=64, num_epochs=100, lr=1e-3, verbose=True, patience=5)
```
Reconstructions are generated from the test set using the trained model, and the mean square error (MSE) is calculated between the reconstructions and the ground truth data.
```
    # Generate reconstructions from the test set and print mean square error compared to the ground truth
    test_recons = sc.inverse_transform(shred(test_dataset.X).detach().cpu().numpy())
    test_ground_truth = sc.inverse_transform(test_dataset.Y.detach().cpu().numpy())
    mse = np.linalg.norm(test_recons - test_ground_truth) / np.linalg.norm(test_ground_truth)
    print("Test Error:", mse)
```
The sea-surface temperature (SST) data and world map indices are loaded for visualization. The reconstructed and ground truth data are reshaped into 2D frames to facilitate plotting.
```
    # SST data with world map indices for plotting
    full_SST, sst_locs = load_full_SST()
    full_test_truth = full_SST[test_indices, :]

    # replacing SST data with our reconstruction
    full_test_recon = full_test_truth.copy()
    full_test_recon[:,sst_locs] = test_recons

    # reshaping to 2d frames
    for x in [full_test_truth, full_test_recon]:
        x.resize(len(x),180,360)
        
    return mse, full_test_recon, full_test_truth
```
### SHRED model in reconstructing the images during noise-free training
The run_shred function is used to train and test the SHRED model, and then visualizes the reconstructed and ground truth images in a plot with two subplots. The plot provides a visual representation of the performance of the SHRED model in reconstructing the images during noise-free training.
```
num_sensors = 3 
lags = 52
mse, recons, truth = run_shred(num_sensors, lags)

# Plot the recons vs truth 
plotdata = [truth, recons]
labels = ['truth','recon']
fig, ax = plt.subplots(1,2,constrained_layout=True,sharey=True)
plt.title('Visualization of Test Images During Noise-Free Training')
for axis,p,label in zip(ax, plotdata, labels):
    axis.imshow(p[0])
    axis.set_aspect('equal')
    axis.text(0.1,0.1,label,color='w',transform=axis.transAxes)
```
### SHRED model in reconstructing the images with added noise during training
We start by selecting the number of sensors and the number of time lags. It then randomly selects sensor locations and defines a function, add_noise, to add Gaussian noise to the data.
```
num_sensors = 3
lags = 52
sensor_locations = np.random.choice(m, size=num_sensors, replace=False)

# Function to add Gaussian noise to data
def add_noise(data, mean=0, std=0.1):
    noise = np.random.normal(mean, std, size=data.shape)
    noisy_data = data + noise
    return noisy_data
```
Next, it divides the data into training, validation, and test sets. The data is preprocessed using sklearn's MinMaxScaler and transformed. 
```
# We now select indices to divide the data into training, validation, and test sets.
train_indices = np.random.choice(n - lags, size=1000, replace=False)
mask = np.ones(n - lags)
mask[train_indices] = 0
valid_test_indices = np.arange(0, n - lags)[np.where(mask != 0)[0]]
valid_indices = valid_test_indices[::2]
test_indices = valid_test_indices[1::2]

# sklearn's MinMaxScaler is used to preprocess the data for training and we generate input/output pairs for the training, validation, and test sets.
sc = MinMaxScaler()
sc = sc.fit(load_X[train_indices])
transformed_X = sc.transform(load_X)
```
We define an array of noise standard deviation values and initialize an empty list to store the test errors.
```
# Generate input sequences to a SHRED model with Gaussian noise
noise_std_values = np.linspace(0, 0.5, num=10)  # Noise standard deviation values
test_errors = []
```
For each noise standard deviation value, we add Gaussian noise to the transformed input data using the add_noise function. We reshape the data into input sequences suitable for the SHRED model. We generate the training, validation, and test datasets using the noisy input data and corresponding output data.
```
for noise_std in noise_std_values:
    noisy_data_in = add_noise(transformed_X, std=noise_std)
    all_data_in = np.zeros((n - lags, lags, num_sensors))
    for i in range(len(all_data_in)):
        all_data_in[i] = noisy_data_in[i:i + lags, sensor_locations]
```
The input sequences for the SHRED model are generated from the noisy data, and the training, validation, and test datasets are created for both state reconstruction and sensor forecasting.
```
    ### Generate training validation and test datasets both for reconstruction of states and forecasting sensors
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_data_in = torch.tensor(all_data_in[train_indices], dtype=torch.float32).to(device)
    valid_data_in = torch.tensor(all_data_in[valid_indices], dtype=torch.float32).to(device)
    test_data_in = torch.tensor(all_data_in[test_indices], dtype=torch.float32).to(device)

    ### -1 to have output be at the same time as final sensor measurements
    train_data_out = torch.tensor(transformed_X[train_indices + lags - 1], dtype=torch.float32).to(device)
    valid_data_out = torch.tensor(transformed_X[valid_indices + lags - 1], dtype=torch.float32).to(device)
    test_data_out = torch.tensor(transformed_X[test_indices + lags - 1], dtype=torch.float32).to(device)

    train_dataset = TimeSeriesDataset(train_data_in, train_data_out)
    valid_dataset = TimeSeriesDataset(valid_data_in, valid_data_out)
    test_dataset = TimeSeriesDataset(test_data_in, test_data_out)
```
The SHRED model is instantiated and trained using the training and validation datasets. The model's performance is evaluated by generating reconstructions from the test set and computing the test error. The test error is then appended to the list of test errors.
```
    # Train the model using the training and validation datasets
    shred = models.SHRED(num_sensors, m, hidden_size=64, hidden_layers=2, l1=350, l2=400, dropout=0.1).to(device)
    validation_errors = models.fit(shred, train_dataset, valid_dataset, batch_size=64, num_epochs=100, lr=1e-3, verbose=True, patience=5)

    # Generate reconstructions from the test set
    test_recons = sc.inverse_transform(shred(test_dataset.X).detach().cpu().numpy())
    test_ground_truth = sc.inverse_transform(test_dataset.Y.detach().cpu().numpy())
    noise_error = np.linalg.norm(test_recons - test_ground_truth) / np.linalg.norm(test_ground_truth)
    test_errors.append(noise_error)
```
Finally, the test error as a function of the noise standard deviation is plotted using Matplotlib.
```
# Plotting the test error as a function of noise
plt.plot(noise_std_values, test_errors)
plt.xlabel('Noise Standard Deviation')
plt.ylabel('Test Error')
plt.title('Test Error as a Function of Noise')
plt.grid(True)
plt.show()
```
### Performance as a function of the number of lags
The list of lag values is defined, and for each lag value in the list, the run_shred function is called with a fixed number of sensors and the specified lag value. The MSE, along with the reconstructed and ground truth data, is obtained and stored in the mse_errors list.

After iterating over all the lag values, a plot is generated to visualize the relationship between the lag values and the MSE errors. The plot displays the lag values on the x-axis and the MSE errors on the y-axis.
```
# Define the list of lag values
lag_val = [13, 32, 52, 71, 91, 110, 130, 149, 169, 188]
num_sensors = 3 
mse_errors = []

for lag in lag_val:
    mse, recons, truth = run_shred(num_sensors, lags)
    mse_errors.append(mse)

# Plot MSE errors vs lags
plt.figure()
plt.plot(lag_val, mse_errors, marker='o')
plt.xlabel("Lag")
plt.ylabel("MSE Error")
plt.title("MSE Errors For Different Lag Values")
plt.show()
```
### Performance as a function of the number of sensors 
The list of sensor numbers is defined, and for each number of sensors in the list, the run_shred function is called with the specified number of sensors and a fixed number of lags. The mean square error (MSE), along with the reconstructed and ground truth data, is obtained from the run_shred function and stored in the mse_errors list.

After iterating over all the sensor numbers, a plot is generated to visualize the relationship between the sensor numbers and the MSE errors. The plot displays the sensor numbers on the x-axis and the MSE errors on the y-axis.
```
# Define the list of number of sensors
sensor_num = [1, 3, 5, 7, 9, 11, 13, 15, 17, 20]
lag = 52
mse_errors = []

for num_sensors in sensor_num:
    mse, recons, truth = run_shred(num_sensors, lags)
    mse_errors.append(mse)

# Plot MSE errors vs sensor numbers
plt.figure()
plt.plot(sensor_num, mse_errors, marker='o')
plt.xlabel("Sensor Numbers")
plt.ylabel("MSE Error")
plt.title("MSE Errors For Different Sensor Numbers")
plt.show()
```




## Result

### SHRED model in reconstructing the images during noise-free training
```
Test Error: 0.034932088
```
![nonoise_visual](https://github.com/tuongv-1736461/Machine-Learning-projects/blob/main/ShallowRecurrentDecoderModel/nonoise_visual.png)
### SHRED model in reconstructing the images with added noise during training
![error_vs_noise](https://github.com/tuongv-1736461/Machine-Learning-projects/blob/main/ShallowRecurrentDecoderModel/error_vs_noise.png)

From the graph, we can observe that there is no clear correlation between the noise standard deviation and the test error. The test error values fluctuate within a narrow range (0.0335 to 0.0365) regardless of the noise standard deviation. This suggests that the model's performance is relatively stable and not significantly influenced by the amount of noise in the input data.

Comparing the test error of noise-free training (0.0349) to the test error of noise-added training, we see no significant difference. Both errors fall within the same range, indicating that the added noise during training does not provide substantial benefits in terms of reducing the test error.

![noise_visualization](https://github.com/tuongv-1736461/Machine-Learning-projects/blob/main/ShallowRecurrentDecoderModel/noise_visualization.png)
### Performance as a function of the number of lags
![error_vs_lag](https://github.com/tuongv-1736461/Machine-Learning-projects/blob/main/ShallowRecurrentDecoderModel/error_vs_lag.png)

The MSE errors do not exhibit a clear relationship with the lag values, such as a consistent increase or decrease, it suggests that the lag value may not have a strong influence on the model's performance.
### Performance as a function of the number of sensors
![error_vs_sensor](https://github.com/tuongv-1736461/Machine-Learning-projects/blob/main/ShallowRecurrentDecoderModel/error_vs_sensor.png)

From the graph, we can observe a decreasing trend in the MSE error as the number of sensors increases. This indicates that increasing the number of sensors leads to improved reconstruction accuracy. The availability of more spatial information allows the SHRED model to capture finer details and patterns in the data, resulting in reduced error in the reconstructed images.

## Conclusion
In conclusion, the evaluation of the SHRED model for image reconstruction and forecasting tasks yielded several key findings. First, the model demonstrated high accuracy when trained without noise, as evidenced by the low mean squared error (MSE) values. Additionally, when Gaussian noise was introduced during training, the model's performance was not significantly affected, with test errors comparable to the noise-free training scenario. Interestingly, the test error values for different noise standard deviations showed no clear correlation. 

Furthermore, the number of sensors used had a notable impact on the model's performance. As the number of sensors increased, the MSE decreased, indicating improved accuracy in both image reconstruction and forecasting. On the other hand, varying the lag values did not exhibit a consistent correlation with the MSE error, suggesting that the choice of lag has a limited influence on the model's performance.

In summary, the SHRED model demonstrates promising capabilities in image reconstruction and forecasting tasks by effectively utilizing sensor measurements. However, its performance appears to be relatively stable and not significantly affected by noise levels or specific lag values.

