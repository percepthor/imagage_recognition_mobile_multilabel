"""
Student model training with knowledge distillation.

Trains EfficientNet-Lite B1 student using:
- Teacher guidance (soft targets)
- Ground truth labels (hard targets)
- Combined distillation loss
"""

import tensorflow as tf
from tensorflow import keras
import logging
from pathlib import Path

from ..models.student import (
    build_student_model,
    freeze_backbone,
    unfreeze_top_layers,
    compile_student
)
from ..models.losses import DistillationLoss
from .callbacks import create_standard_callbacks

logger = logging.getLogger(__name__)


class DistillationModel(keras.Model):
    """
    Wrapper model for student training with distillation.

    Accepts both teacher and student inputs, produces student predictions.
    """

    def __init__(self, teacher, student, alpha=0.7, temperature=2.0):
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.alpha = alpha
        self.temperature = temperature

        # Freeze teacher
        self.teacher.trainable = False

    def call(self, inputs, training=False):
        """
        Forward pass.

        Args:
            inputs: Dictionary with 'teacher_input' and 'student_input'
            training: Training mode

        Returns:
            Student logits
        """
        # Get teacher logits (no gradient)
        teacher_logits = self.teacher(inputs['teacher_input'], training=False)

        # Get student logits
        student_logits = self.student(inputs['student_input'], training=training)

        # Store teacher logits for loss computation
        self.teacher_logits = teacher_logits

        return student_logits

    def train_step(self, data):
        """Custom training step with distillation loss."""
        inputs, y_true = data

        with tf.GradientTape() as tape:
            # Forward pass
            student_logits = self(inputs, training=True)

            # Compute distillation loss
            # Hard loss: student vs ground truth
            hard_loss = self.compiled_loss(
                y_true,
                student_logits,
                regularization_losses=self.losses
            )

            # Soft loss: student vs teacher (with temperature)
            teacher_soft = tf.nn.sigmoid(self.teacher_logits / self.temperature)
            student_soft = tf.nn.sigmoid(student_logits / self.temperature)

            soft_loss = keras.losses.binary_crossentropy(
                teacher_soft,
                student_soft,
                from_logits=False
            )
            soft_loss = tf.reduce_mean(soft_loss)

            # Combined loss
            total_loss = (
                self.alpha * hard_loss +
                (1.0 - self.alpha) * (self.temperature ** 2) * soft_loss
            )

        # Compute gradients (only for student)
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update metrics
        self.compiled_metrics.update_state(y_true, student_logits)

        # Return metrics
        metrics = {m.name: m.result() for m in self.metrics}
        metrics['loss'] = total_loss
        metrics['hard_loss'] = hard_loss
        metrics['soft_loss'] = soft_loss

        return metrics

    def test_step(self, data):
        """Custom test step."""
        inputs, y_true = data

        # Forward pass (no training)
        student_logits = self(inputs, training=False)

        # Compute loss (only hard loss for validation)
        loss = self.compiled_loss(
            y_true,
            student_logits,
            regularization_losses=self.losses
        )

        # Update metrics
        self.compiled_metrics.update_state(y_true, student_logits)

        # Return metrics
        metrics = {m.name: m.result() for m in self.metrics}
        metrics['loss'] = loss

        return metrics


def train_student_with_distillation(
    config: dict,
    teacher_model,
    train_dataset,
    val_dataset,
    out_dir: Path,
    num_classes: int = 7
):
    """
    Train student model with knowledge distillation from teacher.

    Phase A: Warmup head only (backbone frozen)
    Phase B: Fine-tune with distillation (partial backbone unfrozen)

    Args:
        config: Configuration dictionary
        teacher_model: Trained teacher model
        train_dataset: Training dataset (dual inputs)
        val_dataset: Validation dataset (dual inputs)
        out_dir: Output directory
        num_classes: Number of classes

    Returns:
        Trained student model
    """
    logger.info("=" * 80)
    logger.info("TRAINING STUDENT MODEL WITH DISTILLATION")
    logger.info("=" * 80)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    student_config = config['student']
    distill_config = config['distillation']

    # ========== Build Student Model ==========
    logger.info("\nBuilding student model...")
    student_model = build_student_model(
        num_classes=num_classes,
        input_size=student_config['input_size'],
        dropout=student_config['dropout'],
        weights='imagenet'
    )

    # ========== Create Distillation Model ==========
    logger.info("\nCreating distillation wrapper...")
    distill_model = DistillationModel(
        teacher=teacher_model,
        student=student_model,
        alpha=distill_config['alpha'],
        temperature=distill_config['temperature']
    )

    logger.info(f"Distillation config:")
    logger.info(f"  Alpha (hard loss weight): {distill_config['alpha']}")
    logger.info(f"  Temperature: {distill_config['temperature']}")

    # ========== PHASE A: Warmup Head ==========
    logger.info("\n" + "=" * 80)
    logger.info("PHASE A: HEAD WARMUP (student backbone frozen)")
    logger.info("=" * 80)

    # Freeze student backbone
    freeze_backbone(student_model)

    # Compile
    distill_model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=student_config['lr_head'],
            weight_decay=student_config.get('weight_decay', 1e-4)
        ),
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.BinaryAccuracy(name='accuracy'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
        ]
    )

    # Callbacks
    callbacks_warmup = create_standard_callbacks(
        config=student_config,
        monitor='val_loss',
        checkpoint_path=out_dir / 'student_warmup_best.h5',
        log_dir=out_dir / 'logs_student_warmup'
    )

    # Train
    logger.info(f"\nTraining for {student_config['epochs_head']} epochs...")
    history_warmup = distill_model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=student_config['epochs_head'],
        callbacks=callbacks_warmup,
        verbose=2
    )

    logger.info(f"\nPhase A complete!")
    logger.info(f"  Best val_loss: {min(history_warmup.history['val_loss']):.4f}")

    # ========== PHASE B: Fine-tuning with Distillation ==========
    logger.info("\n" + "=" * 80)
    logger.info("PHASE B: FINE-TUNING WITH DISTILLATION")
    logger.info("=" * 80)

    # Unfreeze top layers of student
    unfreeze_top_layers(student_model, unfreeze_fraction=0.4)

    # Recompile with lower learning rate
    distill_model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=student_config['lr_finetune'],
            weight_decay=student_config.get('weight_decay', 1e-4)
        ),
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.BinaryAccuracy(name='accuracy'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
        ]
    )

    # Callbacks for fine-tuning
    callbacks_finetune = create_standard_callbacks(
        config=student_config,
        monitor='val_loss',
        checkpoint_path=out_dir / 'student_distill_best.h5',
        log_dir=out_dir / 'logs_student_finetune'
    )

    # Train
    logger.info(f"\nFine-tuning for {student_config['epochs_finetune']} epochs...")
    history_finetune = distill_model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=student_config['epochs_finetune'],
        callbacks=callbacks_finetune,
        verbose=2
    )

    logger.info(f"\nPhase B complete!")
    logger.info(f"  Best val_loss: {min(history_finetune.history['val_loss']):.4f}")

    # ========== Save Final Student Model ==========
    # Extract student from distillation wrapper
    final_student = distill_model.student

    final_path = out_dir / 'student_final.h5'
    final_student.save(final_path)
    logger.info(f"\nSaved final student model to: {final_path}")

    # ========== Training Summary ==========
    logger.info("\n" + "=" * 80)
    logger.info("STUDENT DISTILLATION TRAINING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total epochs: {student_config['epochs_head'] + student_config['epochs_finetune']}")
    logger.info(f"Models saved:")
    logger.info(f"  - {out_dir / 'student_warmup_best.h5'}")
    logger.info(f"  - {out_dir / 'student_distill_best.h5'}")
    logger.info(f"  - {final_path} (final)")

    return final_student
