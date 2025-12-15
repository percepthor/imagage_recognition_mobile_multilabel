"""
CLI for training multi-label classification with teacher-student distillation and QAT.

Supports multi-GPU training with MirroredStrategy.

Usage:
    python -m src.cli train --data_dir /data/dataset --out_dir /out/run_001 --config /app/configs/default.yaml
"""

import argparse
import logging
import yaml
from pathlib import Path
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_multi_gpu(config: dict):
    """
    Configure multi-GPU training with MirroredStrategy.

    Returns:
        tf.distribute.Strategy or None
    """
    import tensorflow as tf

    multi_gpu_config = config.get('multi_gpu', {})
    if not multi_gpu_config.get('enabled', False):
        logger.info("Multi-GPU disabled, using default strategy")
        return None

    gpus = tf.config.list_physical_devices('GPU')
    logger.info(f"Found {len(gpus)} GPU(s): {gpus}")

    if len(gpus) == 0:
        logger.warning("No GPUs found, falling back to CPU")
        return None

    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            logger.warning(f"Could not set memory growth for {gpu}: {e}")

    strategy = tf.distribute.MirroredStrategy()
    logger.info(f"Using MirroredStrategy with {strategy.num_replicas_in_sync} replicas")

    return strategy


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def train_pipeline(data_dir: str, out_dir: str, config: dict, strategy=None):
    """
    Full training pipeline with multi-GPU support:
    1. Parse and validate dataset
    2. Train teacher model (EfficientNet-B3)
    3. Train student model with distillation (EfficientNet-Lite B1)
    4. Apply QAT and fine-tune
    5. Export to TFLite INT8
    6. Optimize thresholds
    7. Generate all required outputs
    """
    import tensorflow as tf
    import numpy as np
    from .data.parsing import DatasetParser
    from .data.dataset import DatasetBuilder
    from .train.train_teacher import train_teacher
    from .train.train_student_distill import train_student_with_distillation
    from .train.train_student_qat import apply_qat_to_student
    from .eval.metrics import evaluate_model_on_dataset, log_metrics
    from .eval.thresholds import optimize_thresholds
    from .export.tflite_export import (
        export_model_to_tflite_int8,
        create_representative_dataset_generator,
        verify_tflite_model
    )
    from .export.metadata import (
        generate_inference_config,
        save_labels_file,
        save_threshold_recommendation,
        save_metrics_json
    )

    # Setup
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("MULTI-LABEL TRAINING PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Data dir: {data_dir}")
    logger.info(f"Output dir: {out_dir}")

    if strategy:
        logger.info(f"Strategy: MirroredStrategy with {strategy.num_replicas_in_sync} GPUs")
    else:
        logger.info("Strategy: Default (single device)")

    # Set random seed
    seed = config.get('seed', 1337)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    logger.info(f"Random seed: {seed}")

    # ========== STEP 1: Parse and validate dataset ==========
    logger.info(f"\n{'=' * 80}")
    logger.info(f"STEP 1: Parse and validate dataset")
    logger.info(f"{'=' * 80}")

    parser = DatasetParser(data_dir, expected_num_classes=config['num_classes'])
    parsed_data = parser.parse()

    # Save labels.txt
    labels_path = out_dir / "labels.txt"
    parser.save_labels(labels_path)

    # Get multi-hot encoded data
    train_data = parser.get_multihot_data('train')
    val_data = parser.get_multihot_data('val')
    test_data = parser.get_multihot_data('test')

    # Update parsed_data
    parsed_data['train'] = train_data
    parsed_data['val'] = val_data
    parsed_data['test'] = test_data

    # ========== STEP 2: Build datasets ==========
    logger.info(f"\n{'=' * 80}")
    logger.info(f"STEP 2: Build TF datasets")
    logger.info(f"{'=' * 80}")

    # Update parsed_data for DatasetBuilder
    parsed_data_for_builder = {
        'train': train_data,
        'val': val_data,
        'test': test_data,
        'num_classes': parsed_data['num_classes'],
        'classes': parsed_data['classes']
    }

    dataset_builder = DatasetBuilder(parsed_data_for_builder, config)

    # Build datasets for teacher
    logger.info("Building datasets for teacher training...")
    teacher_train_ds, teacher_val_ds, teacher_test_ds = dataset_builder.build_teacher_datasets()

    # Build datasets for student (without distillation first)
    logger.info("Building datasets for student training...")
    student_train_ds, student_val_ds, student_test_ds = dataset_builder.build_student_datasets()

    # Build dual datasets for distillation
    logger.info("Building dual datasets for distillation...")
    distill_train_ds, distill_val_ds = dataset_builder.build_distillation_datasets()

    # ========== STEP 3: Train Teacher ==========
    logger.info(f"\n{'=' * 80}")
    logger.info(f"STEP 3: Train teacher model")
    logger.info(f"{'=' * 80}")

    teacher_model = train_teacher(
        config=config,
        train_dataset=teacher_train_ds,
        val_dataset=teacher_val_ds,
        out_dir=out_dir,
        num_classes=config['num_classes'],
        strategy=strategy
    )

    # Evaluate teacher on validation set
    logger.info("\nEvaluating teacher on validation set...")
    teacher_val_metrics, _, _ = evaluate_model_on_dataset(
        teacher_model,
        teacher_val_ds,
        parsed_data['classes'],
        threshold=0.5
    )
    log_metrics(teacher_val_metrics, prefix="TEACHER VAL")

    # ========== STEP 4: Train Student with Distillation ==========
    logger.info(f"\n{'=' * 80}")
    logger.info(f"STEP 4: Train student with distillation")
    logger.info(f"{'=' * 80}")

    student_model = train_student_with_distillation(
        config=config,
        teacher_model=teacher_model,
        train_dataset=distill_train_ds,
        val_dataset=distill_val_ds,
        out_dir=out_dir,
        num_classes=config['num_classes'],
        strategy=strategy
    )

    # Evaluate student on validation set
    logger.info("\nEvaluating student on validation set...")
    student_val_metrics, _, _ = evaluate_model_on_dataset(
        student_model,
        student_val_ds,
        parsed_data['classes'],
        threshold=0.5
    )
    log_metrics(student_val_metrics, prefix="STUDENT VAL")

    # ========== STEP 5: Apply QAT ==========
    logger.info(f"\n{'=' * 80}")
    logger.info(f"STEP 5: Apply Quantization Aware Training")
    logger.info(f"{'=' * 80}")

    # Try QAT - this is important for INT8 quantization quality
    qat_model = None
    qat_applied = False
    try:
        qat_model = apply_qat_to_student(
            config=config,
            student_model=student_model,
            train_dataset=student_train_ds,
            val_dataset=student_val_ds,
            out_dir=out_dir
        )
        qat_applied = True
        logger.info("QAT applied successfully!")
    except (ValueError, RuntimeError) as e:
        logger.error(f"QAT failed: {e}")
        logger.warning("Will use post-training quantization (PTQ) instead")
        logger.warning("NOTE: PTQ may result in larger model or lower accuracy")
        qat_model = student_model
        qat_applied = False

    # Use student model if QAT failed
    if qat_model is None:
        qat_model = student_model
        qat_applied = False

    # Evaluate model on validation set (QAT or student)
    model_name = "QAT" if qat_model != student_model else "STUDENT"
    logger.info(f"\nEvaluating {model_name} model on validation set...")
    qat_val_metrics, y_true_val, y_pred_val = evaluate_model_on_dataset(
        qat_model,
        student_val_ds,
        parsed_data['classes'],
        threshold=0.5,
        return_predictions=True
    )
    log_metrics(qat_val_metrics, prefix=f"{model_name} VAL")

    # ========== STEP 6: Optimize Thresholds ==========
    logger.info(f"\n{'=' * 80}")
    logger.info(f"STEP 6: Optimize thresholds on validation set")
    logger.info(f"{'=' * 80}")

    thresholds = optimize_thresholds(
        y_true_val,
        y_pred_val,
        class_names=parsed_data['classes'],
        config=config['threshold_search']
    )

    # Save threshold recommendation
    threshold_path = out_dir / "threshold_recommendation.json"
    save_threshold_recommendation(thresholds, threshold_path)

    # ========== STEP 7: Export to TFLite INT8 ==========
    logger.info(f"\n{'=' * 80}")
    logger.info(f"STEP 7: Export to TFLite INT8")
    logger.info(f"{'=' * 80}")

    # Create representative dataset generator
    representative_gen = create_representative_dataset_generator(
        train_data,
        input_size=config['student']['input_size'],
        num_samples=config['tflite_export']['representative_dataset_size'],
    )

    # Export
    tflite_path = out_dir / "model_qat_int8.tflite"
    tflite_metadata = export_model_to_tflite_int8(
        qat_model,
        representative_gen,
        tflite_path,
        input_type=config['tflite_export']['input_type'],
        output_type=config['tflite_export']['output_type']
    )

    # Verify
    logger.info("\nVerifying TFLite model...")
    verify_tflite_model(str(tflite_path), num_classes=config['num_classes'])

    # ========== STEP 8: Generate inference config ==========
    logger.info(f"\n{'=' * 80}")
    logger.info(f"STEP 8: Generate inference configuration")
    logger.info(f"{'=' * 80}")

    inference_config_path = out_dir / "inference_config.json"
    inference_config = generate_inference_config(
        model_filename="model_qat_int8.tflite",
        labels_filename="labels.txt",
        labels=parsed_data['classes'],
        thresholds=thresholds,
        tflite_metadata=tflite_metadata,
        config=config,
        output_path=inference_config_path
    )

    # ========== STEP 9: Final evaluation on TEST set ==========
    logger.info(f"\n{'=' * 80}")
    logger.info(f"STEP 9: Final evaluation on TEST set")
    logger.info(f"{'=' * 80}")

    # Evaluate with optimized thresholds
    test_metrics, _, _ = evaluate_model_on_dataset(
        qat_model,
        student_test_ds,
        parsed_data['classes'],
        threshold=thresholds['global_threshold']
    )

    log_metrics(test_metrics, prefix="TEST (final)")

    # Save metrics
    metrics_path = out_dir / "metrics.json"
    save_metrics_json(test_metrics, metrics_path)

    # ========== Summary ==========
    logger.info(f"\n{'=' * 80}")
    logger.info(f"PIPELINE COMPLETE!")
    logger.info(f"{'=' * 80}")
    logger.info(f"\nAll required outputs generated in: {out_dir}")
    logger.info(f"\nMandatory outputs:")
    logger.info(f"  ✓ model_qat_int8.tflite")
    logger.info(f"  ✓ labels.txt")
    logger.info(f"  ✓ metrics.json")
    logger.info(f"  ✓ threshold_recommendation.json")
    logger.info(f"  ✓ inference_config.json")
    logger.info(f"\nFinal test metrics:")
    logger.info(f"  F1 Macro:  {test_metrics['f1_macro']:.4f}")
    logger.info(f"  F1 Micro:  {test_metrics['f1_micro']:.4f}")
    logger.info(f"  Precision: {test_metrics['precision_macro']:.4f}")
    logger.info(f"  Recall:    {test_metrics['recall_macro']:.4f}")
    logger.info(f"\nReady for mobile deployment!")


def main():
    parser = argparse.ArgumentParser(description='Multi-label classification training')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Train command
    train_parser = subparsers.add_parser('train', help='Run training pipeline')
    train_parser.add_argument('--data_dir', required=True, help='Dataset directory')
    train_parser.add_argument('--out_dir', required=True, help='Output directory')
    train_parser.add_argument('--config', required=True, help='Config YAML file')

    args = parser.parse_args()

    if args.command == 'train':
        try:
            config = load_config(args.config)
            strategy = setup_multi_gpu(config)
            train_pipeline(args.data_dir, args.out_dir, config, strategy=strategy)
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}", exc_info=True)
            sys.exit(1)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
