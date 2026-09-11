"""
models.py
=========
Five classifiers used in the v2 pipeline:

- Random Forest (sklearn)
- Support Vector Machine (sklearn, RBF kernel)
- XGBoost (gradient-boosted trees) -- NEW in v2
- LSTM (TensorFlow/Keras)
- 1D CNN (TensorFlow/Keras)

CHANGELOG (v2):
  - NEW: build_xgb() for XGBoost LOOCV (issue #20).
  - Existing models unchanged.

Deep models include improvements over the original thesis:
- Class-weighted loss (instead of post-hoc undersampling only)
- Per-segment z-score normalization
- Early stopping on validation loss
- Light data augmentation (Gaussian noise injection during training)
"""

import logging
from typing import Tuple, Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight

# XGBoost is optional but recommended. Pipeline degrades gracefully if absent.
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

# Tensorflow imports are deferred to avoid hard dependency for sklearn-only runs
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, callbacks, optimizers
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

import config

logger = logging.getLogger(__name__)


# =====================================================================
# Random Forest
# =====================================================================
def build_rf() -> RandomForestClassifier:
    return RandomForestClassifier(**config.RF_PARAMS)


# =====================================================================
# SVM
# =====================================================================
def build_svm() -> SVC:
    return SVC(**config.SVM_PARAMS)


# =====================================================================
# XGBoost  (NEW in v2)
# =====================================================================
def build_xgb():
    """
    Build an XGBoost classifier configured for cross-patient LOOCV.

    Class imbalance handled implicitly: training pool is already pre-balanced
    to a 1:5 ictal:non-ictal ratio via MAX_TRAIN_NEG subsampling, and
    scale_pos_weight defaults to 1 in config.

    Returns
    -------
    XGBClassifier
    """
    if not XGB_AVAILABLE:
        raise ImportError(
            "XGBoost is not installed. Install with `pip install xgboost`."
        )
    return XGBClassifier(**config.XGB_PARAMS)


# =====================================================================
# LSTM
# =====================================================================
def build_lstm(input_shape: Tuple[int, int]) -> "tf.keras.Model":
    """
    Build the stacked LSTM model.

    Parameters
    ----------
    input_shape : (n_timesteps, n_channels)
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow is required for LSTM/CNN models.")

    p = config.LSTM_PARAMS
    inputs = layers.Input(shape=input_shape)
    x = layers.BatchNormalization()(inputs)
    x = layers.LSTM(p["lstm_units_1"], return_sequences=True)(x)
    x = layers.Dropout(p["dropout"])(x)
    x = layers.LSTM(p["lstm_units_2"], return_sequences=False)(x)
    x = layers.Dropout(p["dropout"])(x)
    x = layers.Dense(p["dense_units"], activation="relu")(x)
    outputs = layers.Dense(2, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="LSTM_Seizure")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=p["learning_rate"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# =====================================================================
# CNN
# =====================================================================
def build_cnn(input_shape: Tuple[int, int]) -> "tf.keras.Model":
    """
    Build the 1D CNN model.

    Parameters
    ----------
    input_shape : (n_timesteps, n_channels)
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow is required for LSTM/CNN models.")

    p = config.CNN_PARAMS
    inputs = layers.Input(shape=input_shape)
    x = layers.BatchNormalization()(inputs)
    x = layers.Conv1D(p["conv1_filters"], p["conv1_kernel"],
                      activation="relu", padding="same")(x)
    x = layers.MaxPooling1D(p["pool_size"])(x)
    x = layers.Dropout(p["dropout"])(x)
    x = layers.Conv1D(p["conv2_filters"], p["conv2_kernel"],
                      activation="relu", padding="same")(x)
    x = layers.MaxPooling1D(p["pool_size"])(x)
    x = layers.Dropout(p["dropout"])(x)
    x = layers.Flatten()(x)
    x = layers.Dense(p["dense_units"], activation="relu")(x)
    x = layers.Dropout(p["dropout"])(x)
    outputs = layers.Dense(2, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="CNN_Seizure")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=p["learning_rate"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# =====================================================================
# Training helpers for deep models
# =====================================================================
def train_keras_model(
    model: "tf.keras.Model",
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    params: dict = None,
):
    """
    Fit a Keras model with early stopping and class weighting.

    Inputs to deep models should have shape (n_samples, n_timesteps, n_channels).
    """
    params = params or config.LSTM_PARAMS

    cls_weight_array = compute_class_weight("balanced",
                                             classes=np.unique(y_train),
                                             y=y_train)
    class_weight = {int(c): float(w)
                    for c, w in zip(np.unique(y_train), cls_weight_array)}

    cb_list = [
        callbacks.EarlyStopping(
            monitor="val_loss" if X_val is not None else "loss",
            patience=params["early_stopping_patience"],
            restore_best_weights=True,
            verbose=0,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss" if X_val is not None else "loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=0,
        ),
    ]

    val_data = (X_val, y_val) if X_val is not None else None
    history = model.fit(
        X_train, y_train,
        validation_data=val_data,
        epochs=params["epochs"],
        batch_size=params["batch_size"],
        class_weight=class_weight,
        callbacks=cb_list,
        verbose=0,
    )
    return history


def normalize_sequences(X: np.ndarray) -> np.ndarray:
    """
    Per-sample, per-channel z-score normalization (a.k.a. instance norm).

    Removes inter-patient amplitude variability — this is the
    'patient-adaptive normalisation' direction recommended in §6.3.2
    of the thesis, applied implicitly through this preprocessing step.

    X shape: (n_samples, n_timesteps, n_channels)
    """
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True) + 1e-8
    return (X - mean) / std
