
# Sea-Surface Temperature Reconstruction using SHRED (Shallow Recurrent Decoder)

## Author
- [Jenny Van](https://github.com/tuongv-1736461)

## Abstract

This project investigates the use of the SHRED (Shallow Recurrent Decoder) model for reconstructing sea-surface temperature (SST) fields from sparse sensor data. The SHRED model combines a Long Short-Term Memory (LSTM) network with a shallow decoder to reconstruct high-dimensional spatio-temporal fields from limited and noisy sensor inputs. Performance was evaluated under varying conditions of time lag, Gaussian noise, and number of sensors. Results show that the SHRED model performs reliably across noise levels, benefits from increased sensor count, and is largely insensitive to lag length, highlighting its robustness for environmental modeling tasks.

## Introduction

Accurate reconstruction of environmental data like SST is vital for climate modeling, weather prediction, and ecosystem monitoring. Traditional sensing methods often require dense and expensive sensor networks. SHRED offers a promising alternative, using LSTM networks to model temporal dependencies and a shallow decoder to reconstruct high-dimensional spatial fields from a sparse set of sensors.

This study evaluates SHRED’s performance using the NOAA Optimum Interpolation SST V2 dataset by testing its robustness against time lags, input noise, and varying sensor configurations.

## Methodology

This section explains how the SHRED model works, how the data was prepared, and how experiments were conducted to evaluate model performance.

### SHRED Overview

SHRED (SHallow REcurrent Decoder) is a hybrid neural network architecture designed to reconstruct high-dimensional spatio-temporal data from a limited number of sensors. It is composed of two main components:

- **LSTM (Long Short-Term Memory) Network**: A special kind of recurrent neural network (RNN) that can learn long-term dependencies in sequential data. In this project, it processes sequences of sensor readings over multiple past time steps (known as “lags”) to understand the temporal evolution of sea-surface temperature (SST).

- **Shallow Decoder**: A simple feedforward neural network that takes the encoded temporal features from the LSTM and reconstructs the entire spatial SST map. This decoder acts as a bridge from the low-dimensional temporal representation to the high-dimensional spatial domain.

Together, these components allow the SHRED model to “fill in the blanks” and reconstruct complete temperature fields from just a few spatial measurements taken over time.

### Data Preprocessing

To train the SHRED model, the raw SST data undergoes several preprocessing steps:

1. **Normalization**  
   The data is scaled using `MinMaxScaler` from scikit-learn. This transformation brings all values into the same range (typically between 0 and 1), which speeds up and stabilizes neural network training.

2. **Sensor Sampling**  
   For each experiment, a small number of sensor locations (e.g., 3, 5, or 10) are randomly selected. These sensors provide partial observations of the SST field. Only data from these locations is used as input to the model.

3. **Time Lagging**  
   The model uses information from multiple past time steps to make predictions. A “lag” of 52, for example, means that 52 previous observations from each sensor are fed into the model to help it understand the current state.

4. **Dataset Splitting**  
   The dataset is divided into three parts:
   - **Training set**: Used to fit the model’s parameters.
   - **Validation set**: Used to tune hyperparameters and prevent overfitting.
   - **Test set**: Used to evaluate the final performance of the trained model.


### Model Training

The training process is encapsulated in a function called `run_shred()`:

1. **Initialization**  
   The function randomly selects sensor locations and splits the data into input (past sensor readings) and output (ground truth temperature field at the final timestep).

2. **Dataset Creation**  
   Custom PyTorch datasets are created for training, validation, and testing. These datasets package the input sequences and expected outputs into mini-batches for efficient learning.

3. **Model Instantiation**  
   A SHRED model is instantiated with specified hyperparameters such as hidden layer size, number of layers, dropout rate, etc.

4. **Training**  
   The model is trained using the training and validation datasets with early stopping to prevent overfitting. The optimization minimizes the mean squared error (MSE) between predicted and actual SST fields.

5. **Evaluation**  
   The trained model is tested on the unseen test set. It outputs reconstructed SST fields, and the final test error (MSE) is computed to quantify accuracy.

## Experiments

To evaluate the SHRED model’s performance, we conducted a series of experiments that tested its robustness across three key dimensions: noise, time lag, and number of sensors. These scenarios mimic real-world conditions such as noisy sensor data, sparse sensor networks, and limited historical records.

### 1. Noise-Free Training

In this baseline experiment, the SHRED model was trained on clean SST data without any added noise.

```python
mse, recons, truth = run_shred(num_sensors=3, lags=52)
```

- **Setup**: 3 sensors, 52 time lags  
- **Test Error (MSE)**: 0.0349

The model accurately reconstructed global SST fields from minimal input data.

![nonoise_visual](https://github.com/tuongv-1736461/Machine-Learning-projects/blob/main/ShallowRecurrentDecoderModel/nonoise_visual.png)

### 2. Performance Under Gaussian Noise

To assess robustness to sensor error, we added Gaussian noise to the input data with standard deviation ranging from 0.0 to 0.5.

- **Setup**: 3 sensors, 52 lags  
- **Result**: MSE remained stable (approx. 0.0335–0.0365) across noise levels.

![error_vs_noise](https://github.com/tuongv-1736461/Machine-Learning-projects/blob/main/ShallowRecurrentDecoderModel/error_vs_noise.png)

> **Insight**: SHRED is highly robust to noisy inputs, making it suitable for real-world scenarios with imperfect sensor data.

![noise_visualization](https://github.com/tuongv-1736461/Machine-Learning-projects/blob/main/ShallowRecurrentDecoderModel/noise_visualization.png)

*This image shows the visual output of the SHRED model when trained with noisy input data. Despite the added Gaussian noise, spatial structures are preserved, indicating strong robustness.*

### 3. Performance vs Time Lag

This experiment varied the number of historical time steps (lags) given to the model to evaluate its sensitivity to temporal context.

```python
lag_val = [13, 32, 52, 71, 91, 110, 130, 149, 169, 188]
```

- **Setup**: 3 sensors, variable lag lengths  
- **Result**: No clear correlation between lag length and performance was observed.

![error_vs_lag](https://github.com/tuongv-1736461/Machine-Learning-projects/blob/main/ShallowRecurrentDecoderModel/error_vs_lag.png)

> **Insight**: SHRED can perform reliably with short to moderate lag histories. This makes it computationally efficient without requiring long-term temporal data.

### 4. Performance vs Number of Sensors

This experiment tested the impact of increasing spatial input (number of sensors) on reconstruction accuracy.

```python
sensor_num = [1, 3, 5, 7, 9, 11, 13, 15, 17, 20]
```

- **Setup**: 52 lags, sensor count ranging from 1 to 20  
- **Result**: As the number of sensors increased, MSE consistently decreased.

![error_vs_sensor](https://github.com/tuongv-1736461/Machine-Learning-projects/blob/main/ShallowRecurrentDecoderModel/error_vs_sensor.png)

> **Insight**: More spatial information leads to better reconstructions. However, SHRED still performs well with just a few sensors, making it cost-effective for sparse deployments.


## Conclusion

The SHRED model effectively reconstructs SST fields from sparse, noisy data. Key findings:
- **Noise Tolerance:** Performance remains stable across a wide range of noise levels.
- **Sensor Sensitivity:** More sensors lead to lower reconstruction errors.
- **Lag Insensitivity:** Varying lag length has minimal effect on accuracy.

These results validate SHRED as a powerful tool for spatio-temporal field reconstruction with limited sensor availability, making it suitable for real-world environmental monitoring and forecasting applications.

## How to Run

### Requirements

- Python 3.8+
- PyTorch
- NumPy
- scikit-learn
- matplotlib

Install dependencies:
```bash
pip install -r requirements.txt
```

Run main experiment:
```bash
python shred_eval.py
```

## Acknowledgments

This project builds on the SHRED (SHallow REcurrent Decoder) framework introduced in the paper:  
**"Sensing with Shallow Recurrent Decoder Networks"** by Jan P. Williams, Olivia Zahn, and J. Nathan Kutz.  
[Read the paper on arXiv](https://arxiv.org/abs/2209.07550)

SHRED models learn mappings from sparse sensor measurement trajectories to high-dimensional spatio-temporal states.  
The datasets used in this project include:
- **Sea-surface temperature (SST)** (downloaded with the repository)
- Additional datasets (e.g., turbulent flow and ozone concentration) are referenced in the supplementary material of the paper.

This repository builds upon the starter code provided by the authors for model architecture and data handling. Training procedures, experimental design, and result visualizations were implemented independently.
