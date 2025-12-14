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

    # Apply quantization to the entire model
    qat_model = tfmot.quantization.keras.quantize_model(student_model)

    logger.info(f"QAT model created")
    logger.info(f"  Original params: {student_model.count_params():,}")
    logger.info(f"  QAT params: {qat_model.count_params():,}")

    # ========== Compile QAT Model ==========
    logger.info("\nCompiling QAT model...")

    # Use very low learning rate for QAT fine-tuning
    qat_model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=qat_config['lr'],
            weight_decay=student_config.get('weight_decay', 1e-4)
        ),
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
        checkpoint_path=out_dir / 'student_qat_best.h5',
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
    final_path = out_dir / 'student_qat_final.h5'
    qat_model.save(final_path)
    logger.info(f"\nSaved QAT model to: {final_path}")

    # ========== QAT Summary ==========
    logger.info("\n" + "=" * 80)
    logger.info("QAT TRAINING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total QAT epochs: {qat_config['epochs']}")
    logger.info(f"Models saved:")
    logger.info(f"  - {out_dir / 'student_qat_best.h5'} (best)")
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

    # Compile
    distill_qat_model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=qat_config['lr'],
            weight_decay=config['student'].get('weight_decay', 1e-4)
        ),
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
        checkpoint_path=out_dir / 'student_qat_distill_best.h5',
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

    # Save
    final_path = out_dir / 'student_qat_distill_final.h5'
    qat_model.save(final_path)
    logger.info(f"\nSaved QAT model (with distillation) to: {final_path}")

    logger.info("\nQAT + Distillation complete!")
    logger.info(f"  Best val_loss: {min(history.history['val_loss']):.4f}")

    return qat_model
