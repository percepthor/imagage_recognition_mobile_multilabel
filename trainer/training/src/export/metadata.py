"""
Metadata generation for mobile inference.

Generates inference_config.json with all information needed for deployment:
- Model details
- Preprocessing parameters
- Labels
- Thresholds
- Quantization details
"""

import json
import logging
from pathlib import Path
from typing import Dict
import hashlib
import time

logger = logging.getLogger(__name__)


def generate_inference_config(
    model_filename: str,
    labels_filename: str,
    labels: list,
    thresholds: dict,
    tflite_metadata: dict,
    config: dict,
    output_path: str
) -> dict:
    """
    Generate complete inference configuration for mobile deployment.

    Args:
        model_filename: Name of .tflite model file
        labels_filename: Name of labels.txt file
        labels: List of class names
        thresholds: Threshold recommendations dict
        tflite_metadata: Metadata from TFLite export
        config: Training configuration
        output_path: Path to save inference_config.json

    Returns:
        Inference configuration dictionary
    """
    logger.info(f"Generating inference configuration...")

    # Determine if output is logits or probabilities
    # Based on export configuration
    output_is_logits = True  # We export logits by default

    # Extract quantization details
    input_quant = tflite_metadata['input']['quantization']
    output_quant = tflite_metadata['output']['quantization']

    # Build configuration
    inference_config = {
        "model": {
            "filename": model_filename,
            "num_classes": len(labels),
            "output_activation": "sigmoid",
            "output_is_logits": output_is_logits,
            "quantization": {
                "input": {
                    "dtype": tflite_metadata['input']['dtype'],
                    "scale": input_quant['scale'],
                    "zero_point": input_quant['zero_point']
                },
                "output": {
                    "dtype": tflite_metadata['output']['dtype'],
                    "scale": output_quant['scale'],
                    "zero_point": output_quant['zero_point']
                }
            }
        },
        "labels": {
            "filename": labels_filename,
            "order": "file_order",
            "classes": labels
        },
        "preprocess": {
            "color_space": "RGB",
            "input_size": config['student']['input_size'],
            "resize": {
                "mode": "keep_aspect_letterbox",
                "interpolation": "bilinear",
                "pad_value_rgb": [0, 0, 0]
            }
        },
        "normalize": {
            "scheme": "efficientnet_lite",
            "formula": "(x - 127.0) / 128.0",
            "input_range": "0..255",
            "note": "For uint8 input, the model handles quantization internally"
        },
        "thresholds": {
            "global": thresholds['global_threshold'],
            "per_class": thresholds.get('per_class_thresholds', {}),
            "use_per_class": thresholds.get('use_per_class', False),
            "empty_result_message": "Sin etiquetas detectadas con suficiente confianza"
        },
        "provenance": {
            "trained_on": f"dataset_{int(time.time())}",
            "seed": config.get('seed', 1337),
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "optimization_target": thresholds.get('optimization_target', 'f1_macro'),
            "calibration_notes": thresholds.get('calibration_notes', '')
        }
    }

    # Save to file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(inference_config, f, indent=2, ensure_ascii=False)

    logger.info(f"  Saved inference config: {output_path}")

    return inference_config


def save_labels_file(labels: list, output_path: str):
    """
    Save labels to labels.txt (one per line).

    Args:
        labels: List of class names
        output_path: Path to save labels.txt
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for label in labels:
            f.write(f"{label}\n")

    logger.info(f"  Saved labels: {output_path}")


def save_threshold_recommendation(
    thresholds: dict,
    output_path: str
):
    """
    Save threshold recommendation to JSON.

    Args:
        thresholds: Threshold dictionary
        output_path: Path to save threshold_recommendation.json
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(thresholds, f, indent=2, ensure_ascii=False)

    logger.info(f"  Saved threshold recommendation: {output_path}")


def save_metrics_json(
    metrics: dict,
    output_path: str
):
    """
    Save metrics to metrics.json.

    Args:
        metrics: Metrics dictionary
        output_path: Path to save metrics.json
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    logger.info(f"  Saved metrics: {output_path}")


def compute_dataset_hash(data_dir: str) -> str:
    """
    Compute hash of dataset for provenance tracking.

    Args:
        data_dir: Dataset directory

    Returns:
        Hash string
    """
    # Simple hash based on file list and modification times
    data_dir = Path(data_dir)

    hasher = hashlib.md5()

    # Hash split files
    for split_file in ['train.txt', 'val.txt', 'test.txt']:
        filepath = data_dir / split_file
        if filepath.exists():
            hasher.update(filepath.read_bytes())

    return hasher.hexdigest()[:16]
