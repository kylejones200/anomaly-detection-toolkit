"""Autoencoder-based anomaly detection methods."""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from .base import BaseDetector

# Optional imports for deep learning methods
# Use string forward references in type hints to avoid runtime import issues
try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment,misc]
    nn = None  # type: ignore[assignment,misc]

try:
    from keras.callbacks import EarlyStopping
    from keras.layers import LSTM, Dense, RepeatVector, TimeDistributed
    from keras.models import Sequential

    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False
    EarlyStopping = None  # type: ignore[assignment,misc]
    LSTM = None  # type: ignore[assignment,misc]
    Dense = None  # type: ignore[assignment,misc]
    RepeatVector = None  # type: ignore[assignment,misc]
    TimeDistributed = None  # type: ignore[assignment,misc]
    Sequential = None  # type: ignore[assignment,misc]


class LSTMAutoencoderDetector(BaseDetector):
    """
    LSTM Autoencoder-based anomaly detector.

    Uses an LSTM autoencoder to learn normal patterns and flags
    points with high reconstruction error as anomalies.

    Parameters
    ----------
    window_size : int, default=20
        Size of sliding window for time series segments.
    lstm_units : list, default=[32, 16]
        Number of units in encoder/decoder LSTM layers.
    contamination : float, default=0.05
        Expected proportion of outliers (used for threshold).
    threshold_std : float, default=3.0
        Number of standard deviations for threshold.
    epochs : int, default=50
        Number of training epochs.
    batch_size : int, default=32
        Batch size for training.
    random_state : int, optional
        Random state for reproducibility.
    """

    def __init__(
        self,
        window_size: int = 20,
        lstm_units: list = None,
        contamination: float = 0.05,
        threshold_std: float = 3.0,
        epochs: int = 50,
        batch_size: int = 32,
        random_state: Optional[int] = None,
    ):
        super().__init__(random_state)
        if not KERAS_AVAILABLE or Sequential is None:
            raise ImportError(
                "Keras/TensorFlow is required for LSTM autoencoder. "
                "Install with: pip install tensorflow"
            )

        self.window_size = window_size
        self.lstm_units = lstm_units or [32, 16]
        self.contamination = contamination
        self.threshold_std = threshold_std
        self.epochs = epochs
        self.batch_size = batch_size
        self.model_ = None
        self.scaler_ = MinMaxScaler()
        self.reconstruction_errors_ = None

    def _create_windows(self, data: np.ndarray) -> np.ndarray:
        """Create sliding windows from time series using vectorized approach."""
        if len(data) < self.window_size:
            return np.empty((0, self.window_size, 1))

        # Flatten to 1D if needed
        if data.ndim > 1:
            data = data.flatten()

        # Use broadcasting with vectorized approach - more memory efficient than loops
        n_windows = len(data) - self.window_size + 1
        indices = np.arange(self.window_size) + np.arange(n_windows)[:, np.newaxis]
        windows = data[indices]

        # Add feature dimension for LSTM input shape
        return windows[:, :, np.newaxis]

    def _create_model(self, input_shape: Tuple[int, int]) -> "Sequential":
        """Create LSTM autoencoder model."""
        if not KERAS_AVAILABLE or Sequential is None:
            raise ImportError("Keras/TensorFlow is required.")
        if LSTM is None or Dense is None or RepeatVector is None or TimeDistributed is None:
            raise ImportError("Keras/TensorFlow is required.")

        model = Sequential(
            [
                LSTM(
                    self.lstm_units[0],
                    activation="relu",
                    input_shape=input_shape,
                    return_sequences=True,
                ),
                LSTM(self.lstm_units[1], activation="relu", return_sequences=False),
                RepeatVector(self.window_size),
                LSTM(self.lstm_units[1], activation="relu", return_sequences=True),
                LSTM(self.lstm_units[0], activation="relu", return_sequences=True),
                TimeDistributed(Dense(1)),
            ]
        )
        model.compile(optimizer="adam", loss="mse")
        return model

    def fit(self, X: Union[np.ndarray, pd.DataFrame, pd.Series]):
        """
        Fit the LSTM autoencoder.

        Parameters
        ----------
        X : array-like of shape (n_samples,)
            Training time series data.
        """
        if not KERAS_AVAILABLE or Sequential is None or EarlyStopping is None:
            raise ImportError("Keras/TensorFlow is required.")

        X = self._validate_input(X)
        if X.ndim > 1 and X.shape[1] > 1:
            raise ValueError("LSTMAutoencoderDetector only supports univariate time series.")

        if X.ndim > 1:
            X = X.flatten()

        # Scale data
        X_scaled = self.scaler_.fit_transform(X.reshape(-1, 1)).flatten()

        # Create windows
        X_windows = self._create_windows(X_scaled)
        if len(X_windows) == 0:
            raise ValueError(f"Data too short for window size {self.window_size}")

        # Create and train model
        self.model_ = self._create_model((self.window_size, 1))
        if self.model_ is None:
            raise RuntimeError("Failed to create model.")
        if EarlyStopping is None:
            raise ImportError("Keras/TensorFlow is required.")

        early_stopping = EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True, verbose=0
        )

        self.model_.fit(
            X_windows,
            X_windows,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=0,
        )

    def predict(self, X: Union[np.ndarray, pd.DataFrame, pd.Series]) -> np.ndarray:
        """
        Predict anomalies in the data.

        Parameters
        ----------
        X : array-like of shape (n_samples,)
            Time series data to predict.

        Returns
        -------
        predictions : ndarray of shape (n_samples,)
            Anomaly predictions. -1 for anomalies, 1 for normal.
            First window_size-1 samples are set to 1 (normal).
        """
        scores = self.score_samples(X)
        mean = np.mean(scores)
        std = np.std(scores)
        threshold = mean + self.threshold_std * std

        predictions = np.where(scores > threshold, -1, 1)
        # Pad with 1s for samples before window_size
        if len(predictions) < len(X):
            padding = np.ones(len(X) - len(predictions))
            predictions = np.concatenate([padding, predictions])
        return predictions

    def score_samples(self, X: Union[np.ndarray, pd.DataFrame, pd.Series]) -> np.ndarray:
        """
        Compute reconstruction error scores.

        Parameters
        ----------
        X : array-like of shape (n_samples,)
            Time series data to score.

        Returns
        -------
        scores : ndarray of shape (n_samples - window_size + 1,)
            Reconstruction error scores. Higher values indicate anomalies.
        """
        if self.model_ is None:
            raise ValueError("Detector must be fitted before scoring.")

        X = self._validate_input(X)
        if X.ndim > 1:
            X = X.flatten()

        # Scale data
        X_scaled = self.scaler_.transform(X.reshape(-1, 1)).flatten()

        # Create windows
        X_windows = self._create_windows(X_scaled)
        if len(X_windows) == 0:
            return np.zeros(len(X))

        # Predict and calculate reconstruction error
        X_pred = self.model_.predict(X_windows, verbose=0)
        reconstruction_error = np.mean(np.abs(X_windows - X_pred), axis=(1, 2))

        self.reconstruction_errors_ = reconstruction_error
        return reconstruction_error


class PyTorchAutoencoderDetector(BaseDetector):
    """
    PyTorch Autoencoder-based anomaly detector.

    Uses a simple feedforward autoencoder to learn normal patterns
    and flags points with high reconstruction error as anomalies.

    Parameters
    ----------
    window_size : int, default=24
        Size of sliding window for time series segments.
    hidden_dims : list, default=[64, 16, 4]
        Hidden dimensions for encoder: [input_dim, hidden1, hidden2, latent].
    learning_rate : float, default=1e-3
        Learning rate for optimizer.
    epochs : int, default=200
        Number of training epochs.
    batch_size : int, default=32
        Batch size for training.
    threshold_std : float, default=3.0
        Number of standard deviations for threshold.
    random_state : int, optional
        Random state for reproducibility.
    """

    def __init__(
        self,
        window_size: int = 24,
        hidden_dims: list = None,
        learning_rate: float = 1e-3,
        epochs: int = 200,
        batch_size: int = 32,
        threshold_std: float = 3.0,
        random_state: Optional[int] = None,
    ):
        super().__init__(random_state)
        if not TORCH_AVAILABLE or torch is None or nn is None:
            raise ImportError(
                "PyTorch is required for PyTorch autoencoder. Install with: pip install torch"
            )

        self.window_size = window_size
        self.hidden_dims = hidden_dims or [64, 16, 4]
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.threshold_std = threshold_std
        self.model_ = None
        self.device_ = None
        self.reconstruction_errors_ = None
        self.X_mean_ = None
        self.X_std_ = None

        if TORCH_AVAILABLE and torch is not None:
            self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if self.random_state is not None:
                torch.manual_seed(self.random_state)

    def _create_windows(self, data: np.ndarray) -> np.ndarray:
        """Create sliding windows from time series using vectorized approach."""
        if len(data) < self.window_size:
            return np.empty((0, self.window_size))

        # Use stride_tricks for efficient sliding window view
        n_windows = len(data) - self.window_size + 1
        shape = (n_windows, self.window_size)
        strides = (data.strides[0], data.strides[0])
        windows = np.lib.stride_tricks.as_strided(
            data, shape=shape, strides=strides, writeable=False
        )
        # Copy to avoid memory issues with strided view
        return windows.copy()

    def _create_model(self, input_dim: int) -> "nn.Module":
        """Create PyTorch autoencoder model."""
        if not TORCH_AVAILABLE or torch is None or nn is None:
            raise ImportError("PyTorch is required.")

        class Autoencoder(nn.Module):
            def __init__(self, input_dim, hidden_dims):
                super().__init__()
                # Encoder
                encoder_layers = []
                prev_dim = input_dim
                for hidden_dim in hidden_dims:
                    encoder_layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU()])
                    prev_dim = hidden_dim

                # Decoder (reverse)
                decoder_layers = []
                for i in range(len(hidden_dims) - 2, -1, -1):
                    decoder_layers.extend(
                        [nn.Linear(hidden_dims[i + 1], hidden_dims[i]), nn.ReLU()]
                    )
                decoder_layers.append(nn.Linear(hidden_dims[0], input_dim))

                self.encoder = nn.Sequential(*encoder_layers)
                self.decoder = nn.Sequential(*decoder_layers)

            def forward(self, x):
                z = self.encoder(x)
                return self.decoder(z)

        return Autoencoder(input_dim, self.hidden_dims).to(self.device_)

    def fit(self, X: Union[np.ndarray, pd.DataFrame, pd.Series]):
        """
        Fit the PyTorch autoencoder.

        Parameters
        ----------
        X : array-like of shape (n_samples,)
            Training time series data.
        """
        if not TORCH_AVAILABLE or torch is None or nn is None:
            raise ImportError("PyTorch is required.")

        X = self._validate_input(X)
        if X.ndim > 1 and X.shape[1] > 1:
            raise ValueError("PyTorchAutoencoderDetector only supports univariate time series.")

        if X.ndim > 1:
            X = X.flatten()

        # Normalize data
        X_mean = np.mean(X)
        X_std = np.std(X)
        X_std = X_std if X_std > 0 else 1.0
        X_normalized = (X - X_mean) / X_std

        # Create windows
        X_windows = self._create_windows(X_normalized)
        if len(X_windows) == 0:
            raise ValueError(f"Data too short for window size {self.window_size}")

        # Train on middle 80% to reduce edge effects
        n = len(X_windows)
        lo, hi = int(0.1 * n), int(0.9 * n)
        X_train = X_windows[lo:hi]

        # Create model
        self.model_ = self._create_model(self.window_size)
        if self.model_ is None:
            raise RuntimeError("Failed to create model.")
        if not TORCH_AVAILABLE or torch is None or nn is None:
            raise ImportError("PyTorch is required.")
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        loss_fn = nn.MSELoss()

        # Convert to tensors
        if torch is None:
            raise ImportError("PyTorch is required.")
        from torch.utils.data import DataLoader, TensorDataset

        dataset = TensorDataset(torch.from_numpy(X_train).float())
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Train
        self.model_.train()
        for _ in range(self.epochs):
            for (batch,) in dataloader:
                batch = batch.to(self.device_)
                optimizer.zero_grad()
                recon = self.model_(batch)
                loss = loss_fn(recon, batch)
                loss.backward()
                optimizer.step()

        # Store normalization parameters
        self.X_mean_ = X_mean
        self.X_std_ = X_std

    def predict(self, X: Union[np.ndarray, pd.DataFrame, pd.Series]) -> np.ndarray:
        """
        Predict anomalies in the data.

        Parameters
        ----------
        X : array-like of shape (n_samples,)
            Time series data to predict.

        Returns
        -------
        predictions : ndarray of shape (n_samples,)
            Anomaly predictions. -1 for anomalies, 1 for normal.
        """
        scores = self.score_samples(X)
        mean = np.mean(scores)
        std = np.std(scores)
        std = std if std > 0 else 1.0
        threshold = mean + self.threshold_std * std

        predictions = np.where(scores > threshold, -1, 1)
        # Pad with 1s for samples before window_size
        if len(predictions) < len(X):
            padding = np.ones(len(X) - len(predictions))
            predictions = np.concatenate([padding, predictions])
        return predictions

    def score_samples(self, X: Union[np.ndarray, pd.DataFrame, pd.Series]) -> np.ndarray:
        """
        Compute reconstruction error scores.

        Parameters
        ----------
        X : array-like of shape (n_samples,)
            Time series data to score.

        Returns
        -------
        scores : ndarray of shape (n_samples - window_size + 1,)
            Reconstruction error scores (MSE per window).
        """
        if self.model_ is None:
            raise ValueError("Detector must be fitted before scoring.")

        X = self._validate_input(X)
        if X.ndim > 1:
            X = X.flatten()

        # Normalize (defensive check for stored normalization params)
        if (
            not hasattr(self, "X_mean_")
            or not hasattr(self, "X_std_")
            or self.X_mean_ is None
            or self.X_std_ is None
        ):
            raise ValueError("Detector must be fitted before scoring.")
        X_std = self.X_std_ if self.X_std_ > 0 else 1.0
        X_normalized = (X - self.X_mean_) / X_std

        # Create windows
        X_windows = self._create_windows(X_normalized)
        if len(X_windows) == 0:
            return np.zeros(len(X))

        # Compute reconstruction errors
        if self.model_ is None:
            raise ValueError("Detector must be fitted before scoring.")
        if not TORCH_AVAILABLE or torch is None:
            raise ImportError("PyTorch is required.")
        if self.device_ is None:
            raise RuntimeError("Device not initialized. PyTorch may not be available.")

        self.model_.eval()
        with torch.no_grad():
            X_tensor = torch.from_numpy(X_windows).float().to(self.device_)
            X_recon = self.model_(X_tensor).cpu().numpy()

        reconstruction_errors = np.mean((X_windows - X_recon) ** 2, axis=1)
        self.reconstruction_errors_ = reconstruction_errors
        return reconstruction_errors
