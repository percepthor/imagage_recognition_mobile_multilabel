"""
Multi-label classification metrics.

Computes:
- Precision, Recall, F1 (macro and micro)
- Per-class metrics
- No-label rate
"""

import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    precision_recall_fscore_support
)
import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)


def compute_multilabel_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list = None,
    threshold: float = 0.5
) -> Dict:
    """
    Compute comprehensive multi-label metrics.

    Args:
        y_true: Ground truth multi-hot labels, shape (N, num_classes)
        y_pred: Predicted probabilities, shape (N, num_classes)
        class_names: List of class names (optional)
        threshold: Threshold for converting probabilities to binary predictions

    Returns:
        Dictionary with all metrics
    """
    # Convert probabilities to binary predictions
    y_pred_binary = (y_pred >= threshold).astype(int)

    # Macro metrics (average per class)
    precision_macro = precision_score(y_true, y_pred_binary, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred_binary, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred_binary, average='macro', zero_division=0)

    # Micro metrics (global aggregation)
    precision_micro = precision_score(y_true, y_pred_binary, average='micro', zero_division=0)
    recall_micro = recall_score(y_true, y_pred_binary, average='micro', zero_division=0)
    f1_micro = f1_score(y_true, y_pred_binary, average='micro', zero_division=0)

    # Per-class metrics
    per_class_precision, per_class_recall, per_class_f1, per_class_support = \
        precision_recall_fscore_support(y_true, y_pred_binary, average=None, zero_division=0)

    # Organize per-class metrics
    per_class_metrics = {}
    num_classes = y_true.shape[1]
    for i in range(num_classes):
        class_name = class_names[i] if class_names else f"class_{i}"
        per_class_metrics[class_name] = {
            "precision": float(per_class_precision[i]),
            "recall": float(per_class_recall[i]),
            "f1": float(per_class_f1[i]),
            "support": int(per_class_support[i])
        }

    # No-label rate (samples with no predictions)
    num_samples = y_pred_binary.shape[0]
    no_predictions = np.sum(y_pred_binary, axis=1) == 0
    no_label_count = np.sum(no_predictions)
    no_label_rate = no_label_count / num_samples if num_samples > 0 else 0

    # Package results
    metrics = {
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_micro": float(precision_micro),
        "recall_micro": float(recall_micro),
        "f1_micro": float(f1_micro),
        "per_class": per_class_metrics,
        "no_label_rate": float(no_label_rate),
        "no_label_count": int(no_label_count),
        "num_samples": int(num_samples),
        "threshold": float(threshold)
    }

    return metrics


def log_metrics(metrics: Dict, prefix: str = ""):
    """
    Log metrics in a readable format.

    Args:
        metrics: Metrics dictionary from compute_multilabel_metrics
        prefix: Prefix for log messages (e.g., "VAL", "TEST")
    """
    logger.info(f"\n{prefix} Metrics:")
    logger.info(f"  F1 Macro:      {metrics['f1_macro']:.4f}")
    logger.info(f"  F1 Micro:      {metrics['f1_micro']:.4f}")
    logger.info(f"  Precision Macro: {metrics['precision_macro']:.4f}")
    logger.info(f"  Recall Macro:   {metrics['recall_macro']:.4f}")
    logger.info(f"  Precision Micro: {metrics['precision_micro']:.4f}")
    logger.info(f"  Recall Micro:   {metrics['recall_micro']:.4f}")
    logger.info(f"  No-label rate:  {metrics['no_label_rate']:.2%} ({metrics['no_label_count']}/{metrics['num_samples']})")
    logger.info(f"  Threshold:      {metrics['threshold']:.3f}")

    logger.info(f"\n{prefix} Per-class metrics:")
    for class_name, class_metrics in metrics['per_class'].items():
        logger.info(
            f"  {class_name:15s} - "
            f"P: {class_metrics['precision']:.3f}, "
            f"R: {class_metrics['recall']:.3f}, "
            f"F1: {class_metrics['f1']:.3f}, "
            f"Support: {class_metrics['support']}"
        )


def evaluate_model_on_dataset(
    model,
    dataset,
    class_names: list,
    threshold: float = 0.5,
    return_predictions: bool = False
) -> Tuple[Dict, np.ndarray, np.ndarray]:
    """
    Evaluate a Keras model on a tf.data.Dataset.

    Args:
        model: Keras model (outputs logits or probabilities)
        dataset: tf.data.Dataset yielding (images, labels)
        class_names: List of class names
        threshold: Threshold for binary classification
        return_predictions: Whether to return y_true and y_pred

    Returns:
        metrics: Dictionary of metrics
        y_true: Ground truth labels (if return_predictions=True)
        y_pred: Predicted probabilities (if return_predictions=True)
    """
    y_true_list = []
    y_pred_logits_list = []

    # Iterate over dataset and collect predictions
    for images, labels in dataset:
        # Get model predictions (logits)
        logits = model(images, training=False)

        # Convert to numpy
        y_true_list.append(labels.numpy())
        y_pred_logits_list.append(logits.numpy())

    # Concatenate
    y_true = np.concatenate(y_true_list, axis=0)
    y_pred_logits = np.concatenate(y_pred_logits_list, axis=0)

    # Convert logits to probabilities
    import tensorflow as tf
    y_pred = tf.nn.sigmoid(y_pred_logits).numpy()

    # Compute metrics
    metrics = compute_multilabel_metrics(
        y_true,
        y_pred,
        class_names=class_names,
        threshold=threshold
    )

    if return_predictions:
        return metrics, y_true, y_pred
    else:
        return metrics, None, None


class F1MacroMetric:
    """
    Custom metric for F1 macro score (for callbacks).

    This can be used with ModelCheckpoint to save best model by F1 macro.
    """

    def __init__(self, threshold: float = 0.5, name: str = 'f1_macro'):
        self.threshold = threshold
        self.name = name

    def __call__(self, y_true, y_pred):
        """
        Compute F1 macro from logits.

        Args:
            y_true: Ground truth labels (multi-hot)
            y_pred: Predicted logits

        Returns:
            F1 macro score
        """
        import tensorflow as tf

        # Convert logits to probabilities
        y_pred_prob = tf.nn.sigmoid(y_pred)

        # Convert to binary predictions
        y_pred_binary = tf.cast(y_pred_prob >= self.threshold, tf.int32)
        y_true_int = tf.cast(y_true, tf.int32)

        # Compute F1 per class
        # Note: This is a simplified implementation for TensorFlow metric
        # For exact sklearn compatibility, use evaluate_model_on_dataset

        # TP, FP, FN per class
        tp = tf.reduce_sum(y_true_int * y_pred_binary, axis=0)
        fp = tf.reduce_sum((1 - y_true_int) * y_pred_binary, axis=0)
        fn = tf.reduce_sum(y_true_int * (1 - y_pred_binary), axis=0)

        # Precision and recall per class
        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)

        # F1 per class
        f1 = 2 * precision * recall / (precision + recall + 1e-7)

        # Macro average
        f1_macro = tf.reduce_mean(f1)

        return f1_macro
