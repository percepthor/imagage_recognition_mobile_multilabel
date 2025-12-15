"""
Teacher model training module.

Trains EfficientNet-B3 teacher model with:
- 2-phase training (warmup head + fine-tune)
- Anti-overfitting strategies
- AdamW optimizer with weight decay
"""

import tensorflow as tf
from tensorflow import keras
import logging
from pathlib import Path

from ..models.teacher import (
    build_teacher_model,
    freeze_backbone,
    unfreeze_top_layers,
    compile_teacher
)
from .callbacks import create_standard_callbacks

logger = logging.getLogger(__name__)


def train_teacher(
    config: dict,
    train_dataset,
    val_dataset,
    out_dir: Path,
    num_classes: int = 7,
    strategy=None
):
    """
    Train teacher model (EfficientNet-B3) with 2-phase strategy.

    Supports multi-GPU with tf.distribute.Strategy.

    Phase A: Warmup head only (backbone frozen)
    Phase B: Fine-tune with partial backbone unfrozen

    Args:
        config: Configuration dictionary
        train_dataset: Training tf.data.Dataset
        val_dataset: Validation tf.data.Dataset
        out_dir: Output directory for checkpoints
        num_classes: Number of classes
        strategy: tf.distribute.Strategy for multi-GPU (optional)

    Returns:
        Trained teacher model
    """
    logger.info("=" * 80)
    logger.info("TRAINING TEACHER MODEL (EfficientNet-B3)")
    logger.info("=" * 80)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    teacher_config = config['teacher']

    # ========== Build Model ==========
    logger.info("\nBuilding teacher model...")

    def create_model():
        return build_teacher_model(
            num_classes=num_classes,
            input_size=teacher_config['input_size'],
            dropout=teacher_config['dropout'],
            weights='imagenet'
        )

    if strategy:
        with strategy.scope():
            model = create_model()
    else:
        model = create_model()

    # ========== PHASE A: Warmup Head ==========
    logger.info("\n" + "=" * 80)
    logger.info("PHASE A: HEAD WARMUP (backbone frozen)")
    logger.info("=" * 80)

    # Freeze backbone
    freeze_backbone(model)

    # Compile (within strategy scope if provided)
    def compile_phase_a():
        compile_teacher(
            model,
            learning_rate=teacher_config['lr_head'],
            weight_decay=teacher_config.get('weight_decay', 1e-4)
        )

    if strategy:
        with strategy.scope():
            compile_phase_a()
    else:
        compile_phase_a()

    # Callbacks
    callbacks_warmup = create_standard_callbacks(
        config=teacher_config,
        monitor='val_loss',
        checkpoint_path=out_dir / 'teacher_warmup_best.keras',
        log_dir=out_dir / 'logs_warmup'
    )

    # Train
    logger.info(f"\nTraining for {teacher_config['epochs_head']} epochs...")
    history_warmup = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=teacher_config['epochs_head'],
        callbacks=callbacks_warmup,
        verbose=2
    )

    logger.info(f"\nPhase A complete!")
    logger.info(f"  Best val_loss: {min(history_warmup.history['val_loss']):.4f}")

    # ========== PHASE B: Fine-tuning ==========
    logger.info("\n" + "=" * 80)
    logger.info("PHASE B: FINE-TUNING (partial backbone unfrozen)")
    logger.info("=" * 80)

    # Unfreeze top layers
    unfreeze_top_layers(model, unfreeze_fraction=0.3)

    # Compile with lower learning rate (within strategy scope if provided)
    def compile_phase_b():
        compile_teacher(
            model,
            learning_rate=teacher_config['lr_finetune'],
            weight_decay=teacher_config.get('weight_decay', 1e-4)
        )

    if strategy:
        with strategy.scope():
            compile_phase_b()
    else:
        compile_phase_b()

    # Callbacks for fine-tuning
    callbacks_finetune = create_standard_callbacks(
        config=teacher_config,
        monitor='val_loss',
        checkpoint_path=out_dir / 'teacher_best.keras',
        validation_data=val_dataset,
        log_dir=out_dir / 'logs_finetune'
    )

    # Train
    logger.info(f"\nFine-tuning for {teacher_config['epochs_finetune']} epochs...")
    history_finetune = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=teacher_config['epochs_finetune'],
        callbacks=callbacks_finetune,
        verbose=2
    )

    logger.info(f"\nPhase B complete!")
    logger.info(f"  Best val_loss: {min(history_finetune.history['val_loss']):.4f}")

    # ========== Save Final Model ==========
    # Save weights only for TF 2.10 compatibility (avoids JSON serialization issues)
    final_path = out_dir / 'teacher_final.weights.h5'
    model.save_weights(final_path)
    logger.info(f"\nSaved final teacher weights to: {final_path}")

    # ========== Training Summary ==========
    logger.info("\n" + "=" * 80)
    logger.info("TEACHER TRAINING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total epochs: {teacher_config['epochs_head'] + teacher_config['epochs_finetune']}")
    logger.info(f"Models saved:")
    logger.info(f"  - {out_dir / 'teacher_warmup_best.weights.h5'} (best from warmup)")
    logger.info(f"  - {out_dir / 'teacher_best.weights.h5'} (best from fine-tuning)")
    logger.info(f"  - {final_path} (final)")

    return model
