"""
Custom callbacks for training.

Includes:
- F1MacroCallback for model checkpointing
- Custom learning rate schedules
- Logging utilities
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class F1MacroCallback(keras.callbacks.Callback):
    """
    Callback to compute F1 macro score on validation set.

    This allows ModelCheckpoint to save the best model based on F1 macro
    instead of just loss or accuracy.
    """

    def __init__(self, validation_data, threshold=0.5, name='val_f1_macro'):
        """
        Args:
            validation_data: tf.data.Dataset for validation
            threshold: Threshold for binary classification
            name: Metric name
        """
        super().__init__()
        self.validation_data = validation_data
        self.threshold = threshold
        self.name = name
        self.f1_scores = []

    def on_epoch_end(self, epoch, logs=None):
        """Compute F1 macro at end of each epoch."""
        from ..eval.metrics import compute_multilabel_metrics

        # Collect predictions
        y_true_list = []
        y_pred_list = []

        for images, labels in self.validation_data:
            # Get predictions (logits)
            logits = self.model(images, training=False)

            # Convert to probabilities
            probs = tf.nn.sigmoid(logits)

            y_true_list.append(labels.numpy())
            y_pred_list.append(probs.numpy())

        # Concatenate
        y_true = np.concatenate(y_true_list, axis=0)
        y_pred = np.concatenate(y_pred_list, axis=0)

        # Compute metrics
        metrics = compute_multilabel_metrics(
            y_true, y_pred,
            threshold=self.threshold
        )

        f1_macro = metrics['f1_macro']
        self.f1_scores.append(f1_macro)

        # Add to logs
        if logs is not None:
            logs[self.name] = f1_macro

        logger.info(f"Epoch {epoch + 1} - {self.name}: {f1_macro:.4f}")


class EMACallback(keras.callbacks.Callback):
    """
    Exponential Moving Average of model weights.

    Maintains a shadow copy of model weights that are updated with EMA.
    Can improve generalization.
    """

    def __init__(self, decay=0.999):
        """
        Args:
            decay: EMA decay factor (typical: 0.999)
        """
        super().__init__()
        self.decay = decay
        self.shadow_weights = None

    def on_train_begin(self, logs=None):
        """Initialize shadow weights."""
        self.shadow_weights = [w.numpy() for w in self.model.weights]

    def on_batch_end(self, batch, logs=None):
        """Update shadow weights after each batch."""
        for i, w in enumerate(self.model.weights):
            self.shadow_weights[i] = (
                self.decay * self.shadow_weights[i] +
                (1 - self.decay) * w.numpy()
            )

    def on_epoch_end(self, epoch, logs=None):
        """Optionally log EMA status."""
        pass

    def apply_ema_weights(self):
        """Apply EMA weights to model."""
        for w, shadow_w in zip(self.model.weights, self.shadow_weights):
            w.assign(shadow_w)
        logger.info("Applied EMA weights to model")


class DetailedLoggingCallback(keras.callbacks.Callback):
    """
    Enhanced logging callback with detailed metrics.
    """

    def __init__(self, log_dir=None):
        super().__init__()
        self.log_dir = Path(log_dir) if log_dir else None
        self.epoch_logs = []

    def on_epoch_end(self, epoch, logs=None):
        """Log detailed metrics."""
        if logs is None:
            return

        logger.info(f"\nEpoch {epoch + 1} Summary:")
        logger.info(f"  Train Loss: {logs.get('loss', 0):.4f}")
        logger.info(f"  Val Loss:   {logs.get('val_loss', 0):.4f}")

        if 'accuracy' in logs:
            logger.info(f"  Train Acc:  {logs.get('accuracy', 0):.4f}")
        if 'val_accuracy' in logs:
            logger.info(f"  Val Acc:    {logs.get('val_accuracy', 0):.4f}")

        if 'val_f1_macro' in logs:
            logger.info(f"  Val F1:     {logs.get('val_f1_macro', 0):.4f}")

        # Store logs - convert tensors to Python types for JSON serialization
        serializable_logs = {'epoch': epoch + 1}
        for key, value in logs.items():
            if hasattr(value, 'numpy'):
                serializable_logs[key] = float(value.numpy())
            elif isinstance(value, (np.floating, np.integer)):
                serializable_logs[key] = float(value)
            else:
                serializable_logs[key] = value
        self.epoch_logs.append(serializable_logs)

    def on_train_end(self, logs=None):
        """Save training history."""
        if self.log_dir:
            import json
            self.log_dir.mkdir(parents=True, exist_ok=True)
            history_path = self.log_dir / 'training_history.json'
            with open(history_path, 'w') as f:
                json.dump(self.epoch_logs, f, indent=2)
            logger.info(f"Saved training history to {history_path}")


def create_standard_callbacks(
    config: dict,
    monitor: str = 'val_loss',
    checkpoint_path: str = None,
    validation_data=None,
    log_dir=None
):
    """
    Create standard set of callbacks for training.

    Args:
        config: Training configuration
        monitor: Metric to monitor
        checkpoint_path: Path to save best model
        validation_data: Validation dataset (for F1 callback)
        log_dir: Directory for logs

    Returns:
        List of callbacks
    """
    callbacks = []

    # Early stopping
    if 'early_stopping_patience' in config:
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor=monitor,
                patience=config['early_stopping_patience'],
                restore_best_weights=True,
                verbose=1
            )
        )
        logger.info(f"Added EarlyStopping (patience={config['early_stopping_patience']})")

    # Reduce LR on plateau
    if 'reduce_lr_patience' in config:
        callbacks.append(
            keras.callbacks.ReduceLROnPlateau(
                monitor=monitor,
                factor=config.get('reduce_lr_factor', 0.5),
                patience=config['reduce_lr_patience'],
                min_lr=1e-7,
                verbose=1
            )
        )
        logger.info(f"Added ReduceLROnPlateau (patience={config['reduce_lr_patience']})")

    # Model checkpoint - use save_weights_only for TF 2.10 JSON serialization compatibility
    if checkpoint_path:
        # Change extension from .keras to .weights.h5 for weights-only saving
        weights_path = str(checkpoint_path).replace('.keras', '.weights.h5')
        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                filepath=weights_path,
                monitor=monitor,
                save_best_only=True,
                save_weights_only=True,
                verbose=1
            )
        )
        logger.info(f"Added ModelCheckpoint (monitor={monitor}, weights_only)")

    # F1 Macro callback (if validation data provided)
    if validation_data is not None and monitor == 'val_f1_macro':
        callbacks.append(
            F1MacroCallback(
                validation_data=validation_data,
                threshold=0.5
            )
        )
        logger.info(f"Added F1MacroCallback")

    # TensorBoard disabled for TF 2.10 compatibility
    # TODO: Re-enable when migrating to newer TF version
    # if log_dir:
    #     tb_path = Path(log_dir) / 'tensorboard'
    #     tb_path.mkdir(parents=True, exist_ok=True)
    #     callbacks.append(
    #         keras.callbacks.TensorBoard(
    #             log_dir=str(tb_path),
    #             histogram_freq=0,
    #             write_graph=False,
    #             update_freq='epoch'
    #         )
    #     )
    #     logger.info(f"Added TensorBoard (log_dir={tb_path})")

    # Detailed logging
    callbacks.append(
        DetailedLoggingCallback(log_dir=log_dir)
    )

    return callbacks
