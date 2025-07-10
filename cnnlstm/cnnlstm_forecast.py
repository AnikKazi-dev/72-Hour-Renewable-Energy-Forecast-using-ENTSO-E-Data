#!/usr/bin/env python
# coding: utf-8

# # 72-Hour Renewable Energy Forecast (CNN-LSTM) using ENTSO-E Data

# %%
# Cell 1: Necessary Imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, LSTM, Dense
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

# %%
# Cell 2: Configuration

config = {
    'data_params': {
        'country_code': 'DE',
        'years_history': 5,
        'target_variable': 'renewable_percentage'
    },
    'model_params': {
        'input_length': 72,       # Use past 72 hours (3 days) of data
        'output_length': 72,      # Predict next 72 hours
        'n_features': 1           # Number of input features
    },
    'training_params': {
        'batch_size': 32,
        'num_epochs': 500,
        'initial_learning_rate': 0.001
    }
}

DATA_FILENAME = f"energy_data_{config['data_params']['country_code']}_{config['data_params']['years_history']}years.csv"

# ### Data Loading and Preprocessing

# %%
# Cell 3: Data Handling and Splitting Functions

def train_val_test_split(data, train_ratio=0.7, val_ratio=0.15):
    """Splits the dataset into train, validation, and test sets."""
    total_len = len(data)
    train_end = int(total_len * train_ratio)
    val_end = train_end + int(total_len * val_ratio)
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    return train_data, val_data, test_data

def fit_scaler(train_data):
    """Fits a MinMaxScaler using the training data."""
    scaler = MinMaxScaler()
    scaler.fit(train_data)
    return scaler

def create_sequences(data_values_scaled, look_back, forecast_horizon):
    """Creates sequences of X (input) and y (target) for time series forecasting."""
    X_list, y_list = [], []
    if len(data_values_scaled) < look_back + forecast_horizon:
        print(f"Not enough data to create sequences. Data length: {len(data_values_scaled)}, "
              f"Required: {look_back + forecast_horizon}")
        return np.array(X_list), np.array(y_list)

    for i in range(len(data_values_scaled) - look_back - forecast_horizon + 1):
        X_list.append(data_values_scaled[i:(i + look_back)])
        y_list.append(data_values_scaled[(i + look_back):(i + look_back + forecast_horizon)])
    return np.array(X_list), np.array(y_list)


# %%
# Cell 4: Load and Prepare Data for TensorFlow

print(f"Loading data from file: {DATA_FILENAME}")
try:
    cached_data = pd.read_csv(DATA_FILENAME, index_col=0, parse_dates=True)
    renewable_series_data = cached_data.squeeze() # Use squeeze() to get a Series
    print("Data loaded successfully.")
except FileNotFoundError:
    print(f"CRITICAL: Data file not found at '{os.path.abspath(DATA_FILENAME)}'.")
    renewable_series_data = None

if renewable_series_data is not None:
    # Scale the data
    data_for_scaling = renewable_series_data.values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data_values = scaler.fit_transform(data_for_scaling).flatten()

    # Create sequences
    X_seq, y_seq = create_sequences(
        scaled_data_values,
        config['model_params']['input_length'],
        config['model_params']['output_length']
    )

    if X_seq.shape[0] > 0:
        X_seq = X_seq.reshape((X_seq.shape[0], X_seq.shape[1], config['model_params']['n_features']))

        # Chronological split
        train_size_idx = int(len(X_seq) * 0.70)
        valid_size_idx = int(len(X_seq) * 0.15)

        X_train, y_train = X_seq[:train_size_idx], y_seq[:train_size_idx]
        X_valid, y_valid = X_seq[train_size_idx : train_size_idx + valid_size_idx], y_seq[train_size_idx : train_size_idx + valid_size_idx]
        X_test, y_test = X_seq[train_size_idx + valid_size_idx:], y_seq[train_size_idx + valid_size_idx:]

        print("Data prepared and split.")
        print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

# ### CNN-LSTM Model Definition (TensorFlow)

# %%
# Cell 5: Model Definition
def build_cnnlstm_model(input_look_back, n_features, forecast_horizon):
    """Builds the CNN-LSTM model."""
    input_layer = Input(shape=(input_look_back, n_features), name="input_sequence")
    x = input_layer

    x = Conv1D(128, kernel_size=2, padding='causal', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv1D(128, kernel_size=2, padding='causal', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv1D(128, kernel_size=2, padding='causal', activation='relu')(x)
    x = BatchNormalization()(x)

    x = LSTM(1024, return_sequences=False)(x)

    x = Dense(1024, activation='relu')(x)
    x = BatchNormalization()(x)
    output_layer = Dense(forecast_horizon, activation='sigmoid')(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model

# %%
# Cell 6: Build and Compile the Model

model = None
if 'X_train' in locals() and X_train.shape[0] > 0:
    model = build_cnnlstm_model(
        config['model_params']['input_length'],
        config['model_params']['n_features'],
        config['model_params']['output_length']
    )
    optimizer = keras.optimizers.Adam(learning_rate=config['training_params']['initial_learning_rate'])
    model.compile(optimizer=optimizer, loss='mse')
    model.summary()
else:
    print("Skipping model building: No training data available.")


# ### Training and Evaluation

# %%
# Cell 7: Plotting Utility Functions

def plot_training_history(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss (MSE)")
    plt.legend()
    plt.grid(True)
    plt.savefig("training_validation_loss.png") # Save the plot

# %%
# Cell 8: Run Training

if 'model' in locals() and model is not None:
    print("---\nStarting CNN-LSTM Model Training (TensorFlow) ---\n")
    history = model.fit(
        X_train, y_train,
        epochs=config['training_params']['num_epochs'],
        batch_size=config['training_params']['batch_size'],
        validation_data=(X_valid, y_valid),
        callbacks=[],  # Callbacks removed to ensure it runs for all epochs
        verbose=1
    )
    plot_training_history(history)
else:
    print("Data or model not available. Cannot run training and evaluation.")


# %%
# Cell 9: Final Model Evaluation on Test Set

if 'model' in locals() and model is not None and 'X_test' in locals() and X_test.shape[0] > 0:
    print("\n--- Final Model Evaluation on Test Set ---")
    # Make predictions
    y_pred_scaled = model.predict(X_test)

    # Invert scaling to get actual values
    y_test_inversed = scaler.inverse_transform(y_test)
    y_pred_inversed = scaler.inverse_transform(y_pred_scaled)

    # Calculate metrics
    mae_overall = mean_absolute_error(y_test_inversed.flatten(), y_pred_inversed.flatten())
    mse_overall = mean_squared_error(y_test_inversed.flatten(), y_pred_inversed.flatten())
    rmse_overall = np.sqrt(mse_overall)

    print(f"\nOverall Test Set Metrics (on inverse-transformed data):")
    print(f"  Mean Absolute Error (MAE): {mae_overall:.4f}")
    print(f"  Mean Squared Error (MSE):  {mse_overall:.4f}")
    print(f"  Root Mean Squared Error (RMSE): {rmse_overall:.4f}")

    # Plot a few examples
    num_plots = min(3, len(X_test))
    if num_plots > 0:
        plt.figure(figsize=(15, 5 * num_plots))
        for i in range(num_plots):
            sample_idx = np.random.randint(0, len(X_test))
            plt.subplot(num_plots, 1, i + 1)
            plt.plot(y_test_inversed[sample_idx, :], label='Actual Future')
            plt.plot(y_pred_inversed[sample_idx, :], label='Predicted Future', linestyle='--')
            plt.title(f"Forecast vs Actuals (Test Sample {sample_idx})")
            plt.xlabel("Time (Hours into the future)")
            plt.ylabel("Renewable Percentage (%)")
            plt.legend()
            plt.grid(True)
        plt.tight_layout()
        plt.savefig("forecast_examples.png")
else:
    print("Model not trained or no test data available for final evaluation.")