"""
FraudDetector — core class for training, saving, loading, and predicting.
Built on Random Forest, extracted from production fraud detection research.
"""

import pickle
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

from .result import PredictionResult


class FraudDetector:
    """
    Train, save, load, and run fraud detection on transaction data.

    Quick start:
        detector = FraudDetector()
        detector.train("transactions.csv", target_col="Class")
        detector.save("fraud_model.pkl")

        result = detector.predict({"V1": -1.36, "Amount": 149.62, ...})
        print(result.label, result.fraud_probability)
    """

    def __init__(
        self,
        n_estimators: int = 100,
        random_state: int = 42,
        high_confidence_threshold: float = 0.80,
        low_confidence_threshold: float = 0.40,
    ):
        """
        Args:
            n_estimators:               Number of trees (default: 100)
            random_state:               Reproducibility seed (default: 42)
            high_confidence_threshold:  P(fraud) above this = high confidence (default: 0.80)
            low_confidence_threshold:   P(fraud) below this = low confidence (default: 0.40)
        """
        self._model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        )
        self._high_threshold = high_confidence_threshold
        self._low_threshold = low_confidence_threshold
        self._feature_names: list[str] = []
        self._is_trained = False

    # ── Training ───────────────────────────────────────────────────────────────

    def train(
        self,
        data: Union[str, pd.DataFrame],
        target_col: str = "Class",
        test_size: float = 0.3,
        verbose: bool = True,
    ) -> dict:
        """
        Train the detector on a CSV file or DataFrame.

        Args:
            data:       Path to CSV or a pandas DataFrame.
            target_col: Name of the label column (0 = legit, 1 = fraud).
            test_size:  Fraction used for evaluation (default: 0.3).
            verbose:    Print training summary (default: True).

        Returns:
            dict: Evaluation metrics on the held-out test set.
        """
        df = pd.read_csv(data) if isinstance(data, str) else data.copy()

        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in data.")

        X = df.drop(columns=[target_col])
        y = df[target_col]
        self._feature_names = list(X.columns)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        if verbose:
            print(f"[fraud-shield] Training on {len(X_train)} samples...")

        self._model.fit(X_train, y_train)
        self._is_trained = True

        metrics = self._compute_metrics(X_test, y_test)

        if verbose:
            print(f"[fraud-shield] Training complete.")
            print(f"  Balanced Accuracy : {metrics['balanced_accuracy']:.4f}")
            print(f"  F1 Score (macro)  : {metrics['f1_macro']:.4f}")
            print(f"  ROC-AUC           : {metrics['roc_auc']:.4f}")

        return metrics

    # ── Prediction ─────────────────────────────────────────────────────────────

    def predict(self, transaction: Union[dict, pd.Series]) -> PredictionResult:
        """
        Predict fraud probability for a single transaction.

        Args:
            transaction: Dict or Series of feature values.

        Returns:
            PredictionResult with is_fraud, fraud_probability, confidence, label.
        """
        self._check_trained()
        row = pd.DataFrame([transaction])[self._feature_names]
        prob = self._model.predict_proba(row)[0][1]
        return self._make_result(prob)

    def predict_batch(self, data: Union[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Predict fraud for a batch of transactions.

        Args:
            data: Path to CSV or DataFrame (must have same features as training data).

        Returns:
            Original DataFrame with added columns:
            fraud_probability, is_fraud, confidence, label.
        """
        self._check_trained()
        df = pd.read_csv(data) if isinstance(data, str) else data.copy()
        X = df[self._feature_names]
        probs = self._model.predict_proba(X)[:, 1]

        df["fraud_probability"] = probs
        df["is_fraud"] = probs >= 0.5
        df["confidence"] = [self._confidence(p) for p in probs]
        df["label"] = df["is_fraud"].map({True: "FRAUD", False: "LEGITIMATE"})
        return df

    # ── Evaluation ─────────────────────────────────────────────────────────────

    def evaluate(
        self,
        data: Union[str, pd.DataFrame],
        target_col: str = "Class",
        verbose: bool = True,
    ) -> dict:
        """
        Evaluate model on labelled data and return metrics.

        Args:
            data:       Path to CSV or DataFrame with ground-truth labels.
            target_col: Name of the label column.
            verbose:    Print full classification report (default: True).

        Returns:
            dict: balanced_accuracy, precision, recall, f1_macro, roc_auc,
                  confusion_matrix, classification_report.
        """
        self._check_trained()
        df = pd.read_csv(data) if isinstance(data, str) else data.copy()
        X = df[self._feature_names]
        y = df[target_col]
        metrics = self._compute_metrics(X, y)

        if verbose:
            print(metrics["classification_report"])
            print("Confusion Matrix:")
            print(metrics["confusion_matrix"])

        return metrics

    def feature_importances(self, top_n: int = 15) -> pd.Series:
        """
        Return top N most important features.

        Args:
            top_n: Number of features to return (default: 15).

        Returns:
            pd.Series sorted by importance descending.
        """
        self._check_trained()
        return (
            pd.Series(self._model.feature_importances_, index=self._feature_names)
            .nlargest(top_n)
        )

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save the trained model to a .pkl file."""
        self._check_trained()
        payload = {
            "model": self._model,
            "feature_names": self._feature_names,
            "high_threshold": self._high_threshold,
            "low_threshold": self._low_threshold,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        print(f"[fraud-shield] Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> "FraudDetector":
        """Load a saved model from a .pkl file."""
        with open(path, "rb") as f:
            payload = pickle.load(f)
        detector = cls(
            high_confidence_threshold=payload["high_threshold"],
            low_confidence_threshold=payload["low_threshold"],
        )
        detector._model = payload["model"]
        detector._feature_names = payload["feature_names"]
        detector._is_trained = True
        print(f"[fraud-shield] Model loaded from {path}")
        return detector

    # ── Internal ───────────────────────────────────────────────────────────────

    def _make_result(self, prob: float) -> PredictionResult:
        is_fraud = prob >= 0.5
        return PredictionResult(
            is_fraud=is_fraud,
            fraud_probability=round(float(prob), 6),
            confidence=self._confidence(prob),
            label="FRAUD" if is_fraud else "LEGITIMATE",
        )

    def _confidence(self, prob: float) -> str:
        distance = abs(prob - 0.5)
        if distance >= (self._high_threshold - 0.5):
            return "high"
        elif distance >= (self._low_threshold - 0.5) if self._low_threshold > 0.5 else True:
            return "medium"
        return "low"

    def _compute_metrics(self, X, y) -> dict:
        y_pred = self._model.predict(X)
        y_proba = self._model.predict_proba(X)[:, 1]
        return {
            "balanced_accuracy": balanced_accuracy_score(y, y_pred),
            "precision_macro": precision_score(y, y_pred, average="macro", zero_division=0),
            "recall_macro": recall_score(y, y_pred, average="macro", zero_division=0),
            "f1_macro": f1_score(y, y_pred, average="macro", zero_division=0),
            "roc_auc": roc_auc_score(y, y_proba),
            "confusion_matrix": confusion_matrix(y, y_pred),
            "classification_report": classification_report(
                y, y_pred, target_names=["Legitimate", "Fraud"]
            ),
        }

    def _check_trained(self):
        if not self._is_trained:
            raise RuntimeError("Model not trained. Call .train() or .load() first.")
