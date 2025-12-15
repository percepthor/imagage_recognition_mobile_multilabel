"""
Quantization Aware Training (QAT) for student model.

Applies QAT to the student model to prepare it for INT8 quantization.
Fine-tunes with very low learning rate to maintain accuracy.
"""

import tensorflow as tf
from tensorflow import keras
import tensorflow_model_optimization as tfmot
import logging
from pathlib import Path

from .callbacks import create_standard_callbacks

logger = logging.getLogger(__name__)


def apply_qat_to_student(
    config: dict,
    student_model,
    train_dataset,
    val_dataset,
    out_dir: Path
):
    """
    Apply Quantization Aware Training to student model.

    QAT inserts fake quantization nodes to simulate INT8 quantization
    during training, allowing the model to adapt to quantization errors.

    Args:
        config: Configuration dictionary
        student_model: Trained student model (from distillation)
        train_dataset: Training dataset (student inputs only)
        val_dataset: Validation dataset (student inputs only)
        out_dir: Output directory

    Returns:
        QAT-trained model
    """
    logger.info("=" * 80)
    logger.info("APPLYING QUANTIZATION AWARE TRAINING (QAT)")
    logger.info("=" * 80)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    qat_config = config['qat']
    student_config = config['student']

    # ========== Apply QAT ==========
    logger.info("\nApplying QAT transformations to model...")

    # Clone model to ensure it's a proper Functional model
    # This is needed because tfmot.quantize_model requires Sequential or Functional
    try:
        cloned_model = keras.models.clone_model(student_model)
        cloned_model.set_weights(student_model.get_weights())
        logger.info("Cloned model for QAT compatibility")
    except Exception as e:
        logger.warning(f"Could not clone model: {e}. Using original.")
        cloned_model = student_model

    # Apply quantization to the entire model with default 8-bit scheme
    logger.info("Applying default 8-bit quantization scheme...")
    try:
        qat_model = tfmot.quantization.keras.quantize_model(cloned_model)
        logger.info("QAT applied successfully with quantize_model()")
    except Exception as e:
        logger.warning(f"quantize_model failed: {e}")
        logger.info("Trying annotate + quantize_apply approach...")
        # Try alternative approach: annotate and then quantize
        try:
            annotated_model = tfmot.quantization.keras.quantize_annotate_model(cloned_model)
            qat_model = tfmot.quantization.keras.quantize_apply(annotated_model)
            logger.info("QAT applied successfully with annotate + apply")
        except Exception as e2:
            logger.error(f"Both QAT approaches failed: {e2}")
            raise RuntimeError(f"QAT failed: {e2}")

    logger.info(f"QAT model created")
    logger.info(f"  Original params: {student_model.count_params():,}")
    logger.info(f"  QAT params: {qat_model.count_params():,}")

    # ========== Compile QAT Model ==========
    logger.info("\nCompiling QAT model...")

    # Use very low learning rate for QAT fine-tuning
    # Use experimental.AdamW for TF 2.10 compatibility
    try:
        qat_optimizer = keras.optimizers.AdamW(
            learning_rate=qat_config['lr'],
            weight_decay=student_config.get('weight_decay', 1e-4)
        )
    except AttributeError:
        qat_optimizer = keras.optimizers.experimental.AdamW(
            learning_rate=qat_config['lr'],
            weight_decay=student_config.get('weight_decay', 1e-4)
        )
    qat_model.compile(
        optimizer=qat_optimizer,
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.BinaryAccuracy(name='accuracy'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
        ]
    )

    logger.info(f"  Learning rate: {qat_config['lr']}")
    logger.info(f"  Weight decay: {student_config.get('weight_decay', 1e-4)}")

    # ========== QAT Fine-tuning ==========
    logger.info("\n" + "=" * 80)
    logger.info("QAT FINE-TUNING")
    logger.info("=" * 80)

    # Callbacks
    callbacks_qat = create_standard_callbacks(
        config=qat_config,
        monitor='val_loss',
        checkpoint_path=out_dir / 'student_qat_best.keras',
        validation_data=val_dataset,
        log_dir=out_dir / 'logs_qat'
    )

    # Train
    logger.info(f"\nFine-tuning with QAT for {qat_config['epochs']} epochs...")
    logger.info(f"IMPORTANT: Using low learning rate to preserve learned features")

    history_qat = qat_model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=qat_config['epochs'],
        callbacks=callbacks_qat,
        verbose=2
    )

    logger.info(f"\nQAT fine-tuning complete!")
    logger.info(f"  Best val_loss: {min(history_qat.history['val_loss']):.4f}")

    # ========== Save QAT Model ==========
    # Save weights only for TF 2.10 compatibility
    final_path = out_dir / 'student_qat_final.weights.h5'
    qat_model.save_weights(final_path)
    logger.info(f"\nSaved QAT weights to: {final_path}")

    # ========== QAT Summary ==========
    logger.info("\n" + "=" * 80)
    logger.info("QAT TRAINING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total QAT epochs: {qat_config['epochs']}")
    logger.info(f"Models saved:")
    logger.info(f"  - {out_dir / 'student_qat_best.weights.h5'} (best)")
    logger.info(f"  - {final_path} (final)")
    logger.info(f"\nNext step: Export to TFLite INT8")

    return qat_model


def apply_qat_with_distillation(
    config: dict,
    teacher_model,
    student_model,
    train_dataset,
    val_dataset,
    out_dir: Path
):
    """
    Apply QAT with continued distillation from teacher.

    This is an advanced variant that keeps the teacher's guidance
    active during QAT fine-tuning.

    Args:
        config: Configuration dictionary
        teacher_model: Teacher model
        student_model: Student model (pre-trained)
        train_dataset: Training dataset (dual inputs)
        val_dataset: Validation dataset (dual inputs)
        out_dir: Output directory

    Returns:
        QAT model
    """
    logger.info("=" * 80)
    logger.info("APPLYING QAT WITH DISTILLATION")
    logger.info("=" * 80)

    out_dir = Path(out_dir)
    qat_config = config['qat']
    distill_config = config['distillation']

    # Apply QAT to student
    logger.info("\nApplying QAT to student model...")
    qat_student = tfmot.quantization.keras.quantize_model(student_model)

    # Create distillation wrapper with QAT student
    from .train_student_distill import DistillationModel

    distill_qat_model = DistillationModel(
        teacher=teacher_model,
        student=qat_student,
        alpha=distill_config['alpha'],
        temperature=distill_config['temperature']
    )

    # Compile - Use experimental.AdamW for TF 2.10 compatibility
    try:
        qat_distill_optimizer = keras.optimizers.AdamW(
            learning_rate=qat_config['lr'],
            weight_decay=config['student'].get('weight_decay', 1e-4)
        )
    except AttributeError:
        qat_distill_optimizer = keras.optimizers.experimental.AdamW(
            learning_rate=qat_config['lr'],
            weight_decay=config['student'].get('weight_decay', 1e-4)
        )
    distill_qat_model.compile(
        optimizer=qat_distill_optimizer,
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.BinaryAccuracy(name='accuracy'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
        ]
    )

    # Callbacks
    callbacks = create_standard_callbacks(
        config=qat_config,
        monitor='val_loss',
        checkpoint_path=out_dir / 'student_qat_distill_best.keras',
        log_dir=out_dir / 'logs_qat_distill'
    )

    # Train
    logger.info(f"\nFine-tuning QAT with distillation for {qat_config['epochs']} epochs...")
    history = distill_qat_model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=qat_config['epochs'],
        callbacks=callbacks,
        verbose=2
    )

    # Extract QAT student
    qat_model = distill_qat_model.student

    # Save weights only for TF 2.10 compatibility
    final_path = out_dir / 'student_qat_distill_final.weights.h5'
    qat_model.save_weights(final_path)
    logger.info(f"\nSaved QAT weights (with distillation) to: {final_path}")

    logger.info("\nQAT + Distillation complete!")
    logger.info(f"  Best val_loss: {min(history.history['val_loss']):.4f}")

    return qat_model
