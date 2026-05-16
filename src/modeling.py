"""Model definitions, training and evaluation helpers."""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score


# ---------------------------------------------------------------------------
# Classical ML baselines
# ---------------------------------------------------------------------------
def make_logreg(seed: int = 42):
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(max_iter=200, random_state=seed)


def make_knn(n_neighbors: int = 5):
    from sklearn.neighbors import KNeighborsClassifier

    return KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1)


def make_random_forest(seed: int = 42, n_estimators: int = 200):
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(
        n_estimators=n_estimators,
        n_jobs=-1,
        random_state=seed,
    )


def make_xgboost(seed: int = 42):
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        num_class=10,
        random_state=seed,
        n_jobs=-1,
        tree_method="hist",
        eval_metric="mlogloss",
    )


def make_lightgbm(seed: int = 42):
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        n_estimators=400,
        max_depth=-1,
        num_leaves=63,
        learning_rate=0.05,
        random_state=seed,
        n_jobs=-1,
    )


# ---------------------------------------------------------------------------
# Deep learning models (Keras)
# ---------------------------------------------------------------------------
def make_simple_cnn(input_shape: Tuple[int, int, int], num_classes: int = 10):
    """Small 3-block CNN — shallow baseline for deep models."""
    from tensorflow.keras import layers, models

    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPool2D(),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.MaxPool2D(),
        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def make_deeper_cnn(input_shape: Tuple[int, int, int], num_classes: int = 10):
    """Deeper CNN with BatchNorm and Dropout."""
    from tensorflow.keras import layers, models

    def conv_block(x, filters):
        x = layers.Conv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPool2D()(x)
        x = layers.Dropout(0.25)(x)
        return x

    inputs = layers.Input(shape=input_shape)
    x = conv_block(inputs, 32)
    x = conv_block(x, 64)
    x = conv_block(x, 128)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def make_resnet_like(input_shape: Tuple[int, int, int], num_classes: int = 10):
    """ResNet-style CNN with residual connections — final model."""
    from tensorflow.keras import layers, models

    def res_block(x, filters):
        shortcut = x
        if shortcut.shape[-1] != filters:
            shortcut = layers.Conv2D(filters, 1, padding="same")(shortcut)
        x = layers.Conv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([x, shortcut])
        x = layers.Activation("relu")(x)
        return x

    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = res_block(x, 64)
    x = layers.MaxPool2D()(x)
    x = layers.Dropout(0.25)(x)
    x = res_block(x, 128)
    x = layers.MaxPool2D()(x)
    x = layers.Dropout(0.3)(x)
    x = res_block(x, 256)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------
def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute accuracy, macro-F1 and weighted-F1."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
    }


def report(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    return classification_report(y_true, y_pred, digits=4)
