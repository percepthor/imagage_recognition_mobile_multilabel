"""
Loss functions for teacher-student distillation in multi-label classification.

Implements:
- Hard loss: BCE with ground truth labels
- Soft loss: Distillation from teacher logits
- Combined distillation loss
"""

import tensorflow as tf
from tensorflow import keras


class DistillationLoss(keras.losses.Loss):
    """
    Combined distillation loss for multi-label classification.

    Loss = alpha * L_hard + (1 - alpha) * (T^2) * L_soft

    Where:
    - L_hard: Binary cross-entropy with ground truth labels
    - L_soft: Binary cross-entropy between soft teacher and student predictions
    - alpha: Weight for hard loss (typical: 0.7)
    - T: Temperature (typical: 2.0)
    """

    def __init__(
        self,
        alpha: float = 0.7,
        temperature: float = 2.0,
        name: str = 'distillation_loss',
        **kwargs
    ):
        """
        Args:
            alpha: Weight for hard loss (1-alpha for soft loss)
            temperature: Temperature for distillation
            name: Loss name
        """
        super().__init__(name=name, **kwargs)
        self.alpha = alpha
        self.temperature = temperature

        # Hard loss: BCE with logits
        self.hard_loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)

        # Soft loss: BCE without logits (on probabilities)
        self.soft_loss_fn = keras.losses.BinaryCrossentropy(from_logits=False)

    def call(self, y_true, y_pred):
        """
        Compute distillation loss.

        Args:
            y_true: Tuple of (ground_truth_labels, teacher_logits)
                    - ground_truth_labels: shape (batch, num_classes)
                    - teacher_logits: shape (batch, num_classes)
            y_pred: Student logits, shape (batch, num_classes)

        Returns:
            Combined distillation loss
        """
        # Unpack y_true
        y_ground_truth, teacher_logits = y_true[0], y_true[1]

        # Hard loss: student vs ground truth
        # y_pred is student logits, y_ground_truth is multi-hot labels
        hard_loss = self.hard_loss_fn(y_ground_truth, y_pred)

        # Soft loss: student vs teacher (temperature-scaled)
        # Convert logits to soft probabilities with temperature
        teacher_soft = tf.nn.sigmoid(teacher_logits / self.temperature)
        student_soft = tf.nn.sigmoid(y_pred / self.temperature)

        # BCE between soft predictions
        soft_loss = self.soft_loss_fn(teacher_soft, student_soft)

        # Combined loss with temperature scaling
        loss = self.alpha * hard_loss + (1.0 - self.alpha) * (self.temperature ** 2) * soft_loss

        return loss

    def get_config(self):
        config = super().get_config()
        config.update({
            'alpha': self.alpha,
            'temperature': self.temperature
        })
        return config


def create_distillation_loss(alpha: float = 0.7, temperature: float = 2.0):
    """
    Factory function to create distillation loss.

    Args:
        alpha: Weight for hard loss
        temperature: Temperature for distillation

    Returns:
        DistillationLoss instance
    """
    return DistillationLoss(alpha=alpha, temperature=temperature)


def focal_binary_crossentropy(
    y_true,
    y_pred,
    gamma: float = 2.0,
    alpha: float = 0.25,
    from_logits: bool = True
):
    """
    Focal loss for multi-label classification.

    Useful for class imbalance. Can be used instead of standard BCE.

    Args:
        y_true: Ground truth labels
        y_pred: Predictions (logits or probabilities)
        gamma: Focusing parameter
        alpha: Balancing parameter
        from_logits: Whether y_pred is logits

    Returns:
        Focal loss
    """
    # Convert logits to probabilities if needed
    if from_logits:
        y_pred_prob = tf.nn.sigmoid(y_pred)
    else:
        y_pred_prob = y_pred

    # Clip for numerical stability
    y_pred_prob = tf.clip_by_value(y_pred_prob, 1e-7, 1.0 - 1e-7)

    # Compute focal weight
    # For positive class: (1 - p)^gamma
    # For negative class: p^gamma
    focal_weight = tf.where(
        tf.equal(y_true, 1.0),
        tf.pow(1.0 - y_pred_prob, gamma),
        tf.pow(y_pred_prob, gamma)
    )

    # Compute BCE
    bce = -y_true * tf.math.log(y_pred_prob) - (1 - y_true) * tf.math.log(1 - y_pred_prob)

    # Apply focal weight and alpha
    focal_loss = alpha * focal_weight * bce

    # Reduce
    return tf.reduce_mean(focal_loss)


class FocalBinaryCrossentropy(keras.losses.Loss):
    """Focal loss as a Keras loss class."""

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.25,
        from_logits: bool = True,
        name: str = 'focal_binary_crossentropy',
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        return focal_binary_crossentropy(
            y_true, y_pred,
            gamma=self.gamma,
            alpha=self.alpha,
            from_logits=self.from_logits
        )

    def get_config(self):
        config = super().get_config()
        config.update({
            'gamma': self.gamma,
            'alpha': self.alpha,
            'from_logits': self.from_logits
        })
        return config
