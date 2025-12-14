"""
Threshold optimization for multi-label classification.

Implements:
- Grid search for optimal global threshold
- (Optional) Per-class threshold optimization
- Deterministic and reproducible
"""

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)


def find_optimal_global_threshold(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    grid_min: float = 0.05,
    grid_max: float = 0.95,
    grid_step: float = 0.01,
    objective: str = 'f1_macro'
) -> Tuple[float, Dict]:
    """
    Find optimal global threshold via grid search.

    Args:
        y_true: Ground truth multi-hot labels (N, num_classes)
        y_pred_probs: Predicted probabilities (N, num_classes)
        grid_min: Minimum threshold to search
        grid_max: Maximum threshold to search
        grid_step: Step size for grid
        objective: Optimization objective ('f1_macro' or 'f1_micro')

    Returns:
        best_threshold: Optimal threshold
        results: Dictionary with search results
    """
    logger.info(f"Searching for optimal global threshold...")
    logger.info(f"  Grid: [{grid_min:.2f}, {grid_max:.2f}], step={grid_step:.3f}")
    logger.info(f"  Objective: {objective}")

    # Grid of thresholds
    thresholds = np.arange(grid_min, grid_max + grid_step, grid_step)

    best_threshold = 0.5
    best_score = 0.0
    best_metrics = {}

    results = {
        'thresholds': [],
        'f1_macro': [],
        'f1_micro': [],
        'precision_macro': [],
        'recall_macro': []
    }

    for threshold in thresholds:
        # Convert probabilities to binary predictions
        y_pred_binary = (y_pred_probs >= threshold).astype(int)

        # Compute metrics
        f1_macro = f1_score(y_true, y_pred_binary, average='macro', zero_division=0)
        f1_micro = f1_score(y_true, y_pred_binary, average='micro', zero_division=0)
        precision_macro = precision_score(y_true, y_pred_binary, average='macro', zero_division=0)
        recall_macro = recall_score(y_true, y_pred_binary, average='macro', zero_division=0)

        # Store results
        results['thresholds'].append(float(threshold))
        results['f1_macro'].append(float(f1_macro))
        results['f1_micro'].append(float(f1_micro))
        results['precision_macro'].append(float(precision_macro))
        results['recall_macro'].append(float(recall_macro))

        # Select objective score
        if objective == 'f1_macro':
            score = f1_macro
        elif objective == 'f1_micro':
            score = f1_micro
        else:
            raise ValueError(f"Unknown objective: {objective}")

        # Update best
        if score > best_score or (score == best_score and recall_macro > best_metrics.get('recall_macro', 0)):
            # Tie-breaker: prefer higher recall_macro
            best_score = score
            best_threshold = threshold
            best_metrics = {
                'f1_macro': f1_macro,
                'f1_micro': f1_micro,
                'precision_macro': precision_macro,
                'recall_macro': recall_macro
            }

    logger.info(f"  Found optimal threshold: {best_threshold:.3f}")
    logger.info(f"    F1 Macro:  {best_metrics['f1_macro']:.4f}")
    logger.info(f"    F1 Micro:  {best_metrics['f1_micro']:.4f}")
    logger.info(f"    Precision: {best_metrics['precision_macro']:.4f}")
    logger.info(f"    Recall:    {best_metrics['recall_macro']:.4f}")

    results['best_threshold'] = float(best_threshold)
    results['best_metrics'] = best_metrics

    return best_threshold, results


def find_optimal_per_class_thresholds(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    class_names: list,
    grid_min: float = 0.05,
    grid_max: float = 0.95,
    grid_step: float = 0.01,
    global_threshold: float = 0.5,
    max_deviation: float = 0.25
) -> Dict[str, float]:
    """
    Find optimal per-class thresholds.

    For each class, search for threshold that maximizes F1 for that class.
    Apply smoothing: clamp thresholds to global_threshold ± max_deviation.

    Args:
        y_true: Ground truth multi-hot labels (N, num_classes)
        y_pred_probs: Predicted probabilities (N, num_classes)
        class_names: List of class names
        grid_min: Minimum threshold
        grid_max: Maximum threshold
        grid_step: Step size
        global_threshold: Global threshold for reference
        max_deviation: Maximum allowed deviation from global threshold

    Returns:
        per_class_thresholds: Dict mapping class name to optimal threshold
    """
    logger.info(f"Searching for per-class thresholds...")

    num_classes = y_true.shape[1]
    per_class_thresholds = {}

    thresholds = np.arange(grid_min, grid_max + grid_step, grid_step)

    for class_idx in range(num_classes):
        class_name = class_names[class_idx]

        y_true_class = y_true[:, class_idx]
        y_pred_class = y_pred_probs[:, class_idx]

        best_threshold = global_threshold
        best_f1 = 0.0

        for threshold in thresholds:
            y_pred_binary = (y_pred_class >= threshold).astype(int)

            # F1 for this class
            # Note: We need to handle the case where precision/recall are undefined
            tp = np.sum((y_true_class == 1) & (y_pred_binary == 1))
            fp = np.sum((y_true_class == 0) & (y_pred_binary == 1))
            fn = np.sum((y_true_class == 1) & (y_pred_binary == 0))

            if tp + fp == 0:
                precision = 0
            else:
                precision = tp / (tp + fp)

            if tp + fn == 0:
                recall = 0
            else:
                recall = tp / (tp + fn)

            if precision + recall == 0:
                f1 = 0
            else:
                f1 = 2 * precision * recall / (precision + recall)

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        # Apply smoothing: clamp to global_threshold ± max_deviation
        clamped_threshold = np.clip(
            best_threshold,
            global_threshold - max_deviation,
            global_threshold + max_deviation
        )

        per_class_thresholds[class_name] = float(clamped_threshold)

        logger.info(f"  {class_name:15s}: {clamped_threshold:.3f} (F1={best_f1:.3f})")

    return per_class_thresholds


def optimize_thresholds(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    class_names: list,
    config: Dict
) -> Dict:
    """
    Full threshold optimization pipeline.

    Steps:
    1. Find optimal global threshold
    2. (Optional) Find per-class thresholds if improvement is significant

    Args:
        y_true: Ground truth labels (N, num_classes)
        y_pred_probs: Predicted probabilities (N, num_classes)
        class_names: List of class names
        config: Threshold search configuration

    Returns:
        Dictionary with threshold recommendations
    """
    # Extract config
    grid_min = config.get('grid_min', 0.05)
    grid_max = config.get('grid_max', 0.95)
    grid_step = config.get('grid_step', 0.01)
    objective = config.get('objective', 'f1_macro')
    per_class_enabled = config.get('per_class_enabled', True)
    improvement_threshold = config.get('per_class_improvement_threshold', 0.005)

    # Step 1: Find optimal global threshold
    global_threshold, global_results = find_optimal_global_threshold(
        y_true, y_pred_probs,
        grid_min=grid_min,
        grid_max=grid_max,
        grid_step=grid_step,
        objective=objective
    )

    global_f1_macro = global_results['best_metrics']['f1_macro']

    # Step 2: Per-class thresholds (if enabled)
    per_class_thresholds = None
    use_per_class = False

    if per_class_enabled:
        per_class_thresholds = find_optimal_per_class_thresholds(
            y_true, y_pred_probs,
            class_names=class_names,
            grid_min=grid_min,
            grid_max=grid_max,
            grid_step=grid_step,
            global_threshold=global_threshold,
            max_deviation=0.25
        )

        # Evaluate with per-class thresholds
        y_pred_binary_per_class = np.zeros_like(y_true, dtype=int)
        for class_idx, class_name in enumerate(class_names):
            threshold = per_class_thresholds[class_name]
            y_pred_binary_per_class[:, class_idx] = (y_pred_probs[:, class_idx] >= threshold).astype(int)

        f1_macro_per_class = f1_score(y_true, y_pred_binary_per_class, average='macro', zero_division=0)

        # Check if per-class improves by at least improvement_threshold
        relative_improvement = (f1_macro_per_class - global_f1_macro) / (global_f1_macro + 1e-7)

        logger.info(f"\nPer-class threshold evaluation:")
        logger.info(f"  Global F1 macro:    {global_f1_macro:.4f}")
        logger.info(f"  Per-class F1 macro: {f1_macro_per_class:.4f}")
        logger.info(f"  Improvement:        {relative_improvement:.2%}")

        if relative_improvement >= improvement_threshold:
            use_per_class = True
            logger.info(f"  -> Using per-class thresholds (improvement >= {improvement_threshold:.1%})")
        else:
            logger.info(f"  -> Using global threshold (improvement < {improvement_threshold:.1%})")

    # Package results
    result = {
        'global_threshold': global_threshold,
        'per_class_thresholds': per_class_thresholds if use_per_class else {},
        'use_per_class': use_per_class,
        'optimization_target': objective,
        'grid': {
            'min': grid_min,
            'max': grid_max,
            'step': grid_step
        },
        'calibration_notes': (
            f"Optimized on VAL set using {objective}. "
            f"Per-class thresholds {'enabled' if use_per_class else 'disabled'} "
            f"(improvement {'>='{improvement_threshold:.1%} if use_per_class else '<' + f{improvement_threshold:.1%}})."
        )
    }

    return result
