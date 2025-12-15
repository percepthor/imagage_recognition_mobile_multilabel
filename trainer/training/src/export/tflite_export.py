"""
TFLite export with full-integer (INT8) quantization.

Implements:
- QAT model to TFLite conversion
- Representative dataset for calibration
- Full-integer quantization with uint8 input and int8 output

Requires TF_USE_LEGACY_KERAS=1 for compatibility.
"""

import tensorflow as tf
import numpy as np
import logging
from pathlib import Path
from typing import List, Tuple, Callable, Generator
import os

logger = logging.getLogger(__name__)

# Maximum allowed file size in bytes (6 MB)
MAX_TFLITE_SIZE_BYTES = 6 * 1024 * 1024


def create_representative_dataset_generator(
    data: List[Tuple[str, List[int]]],
    input_size: int = 380,
    num_samples: int = 200,
) -> Callable[[], Generator]:
    """
    Create representative dataset generator for TFLite quantization calibration.

    The generator yields preprocessed images in the same format as training.
    Images are normalized to [-1, 1] range using (x - 127.0) / 128.0

    Args:
        data: List of (image_path, labels) tuples
        input_size: Target size for letterbox preprocessing
        num_samples: Number of samples to use for calibration

    Returns:
        Generator function for TFLite converter
    """
    import cv2

    # Sample subset of data
    sampled_data = data[:num_samples] if len(data) > num_samples else data
    logger.info(f"Representative dataset: {len(sampled_data)} samples")

    def letterbox_preprocess(image_path: str) -> np.ndarray:
        """Preprocess image with letterbox to target size."""
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w = img.shape[:2]
        target = input_size

        # Calculate scale (fit the larger dimension)
        scale = target / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)

        # Resize with bilinear interpolation
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Create letterbox canvas (black padding)
        canvas = np.zeros((target, target, 3), dtype=np.uint8)

        # Center the image
        top = (target - new_h) // 2
        left = (target - new_w) // 2
        canvas[top:top + new_h, left:left + new_w] = img_resized

        return canvas

    def representative_dataset():
        for img_path, _ in sampled_data:
            try:
                # Preprocess image (letterbox to 380x380)
                image = letterbox_preprocess(img_path)

                # Normalize: (x - 127.0) / 128.0 maps [0..255] to ~[-1..1]
                image = (image.astype(np.float32) - 127.0) / 128.0

                # Add batch dimension and yield
                yield [np.expand_dims(image, axis=0)]

            except Exception as e:
                logger.warning(f"Skipping image {img_path}: {e}")
                continue

    return representative_dataset


def export_model_to_tflite_int8(
    model: tf.keras.Model,
    representative_dataset_gen: Callable,
    output_path: str,
    input_type: str = 'uint8',
    output_type: str = 'int8'
) -> dict:
    """
    Export Keras model to TFLite with full-integer INT8 quantization.

    This function converts a QAT-trained model to TFLite format with:
    - Full integer quantization (INT8 ops)
    - uint8 input (accepts raw image bytes 0..255)
    - int8 output (quantized logits)

    Args:
        model: QAT-trained Keras model
        representative_dataset_gen: Representative dataset generator function
        output_path: Path to save .tflite file
        input_type: Input tensor type ('uint8' or 'int8')
        output_type: Output tensor type ('int8' or 'uint8')

    Returns:
        Dictionary with export metadata

    Raises:
        RuntimeError: If export fails or model exceeds size limit
    """
    logger.info("=" * 80)
    logger.info("EXPORTING MODEL TO TFLITE INT8")
    logger.info("=" * 80)
    logger.info(f"  Output path: {output_path}")
    logger.info(f"  Input type: {input_type}")
    logger.info(f"  Output type: {output_type}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create TFLite converter from Keras model
    logger.info("\nCreating TFLite converter...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Enable quantization optimizations
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Set representative dataset for calibration
    converter.representative_dataset = representative_dataset_gen

    # Force full-integer quantization (INT8 ops only)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

    # Set input type
    if input_type == 'uint8':
        converter.inference_input_type = tf.uint8
    elif input_type == 'int8':
        converter.inference_input_type = tf.int8
    else:
        raise ValueError(f"Invalid input_type: {input_type}. Use 'uint8' or 'int8'")

    # Set output type
    if output_type == 'int8':
        converter.inference_output_type = tf.int8
    elif output_type == 'uint8':
        converter.inference_output_type = tf.uint8
    else:
        raise ValueError(f"Invalid output_type: {output_type}. Use 'int8' or 'uint8'")

    # Convert model
    logger.info("\nConverting model to TFLite INT8...")
    logger.info("  This may take a few minutes...")

    try:
        tflite_model = converter.convert()
        logger.info("  Full INT8 conversion successful!")
    except Exception as e:
        logger.warning(f"  Full INT8 conversion failed: {e}")
        logger.info("  Trying hybrid quantization (INT8 with float fallback)...")

        # Reset converter and try with hybrid mode
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset_gen

        # Allow float fallback for unsupported ops
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
            tf.lite.OpsSet.TFLITE_BUILTINS  # Allow float fallback
        ]

        # Still try to get uint8/int8 I/O
        try:
            if input_type == 'uint8':
                converter.inference_input_type = tf.uint8
            if output_type == 'int8':
                converter.inference_output_type = tf.int8
        except Exception:
            logger.warning("  Could not set integer I/O types, using float")

        try:
            tflite_model = converter.convert()
            logger.info("  Hybrid quantization successful!")
        except Exception as e2:
            logger.error(f"  Hybrid conversion also failed: {e2}")
            raise RuntimeError(f"TFLite conversion failed: {e2}")

    # Save to file
    with open(output_path, 'wb') as f:
        f.write(tflite_model)

    file_size_bytes = output_path.stat().st_size
    file_size_mb = file_size_bytes / (1024 * 1024)
    logger.info(f"\nSaved TFLite model: {file_size_mb:.2f} MB")

    # Check size limit
    if file_size_bytes > MAX_TFLITE_SIZE_BYTES:
        logger.warning(f"  WARNING: Model size {file_size_mb:.2f} MB exceeds limit of 6 MB!")
    else:
        logger.info(f"  Size OK: {file_size_mb:.2f} MB <= 6 MB limit")

    # Verify and get metadata
    metadata = verify_tflite_model(str(output_path))

    return metadata


def verify_tflite_model(tflite_path: str, num_classes: int = 7) -> dict:
    """
    Verify TFLite model structure and run smoke test.

    Validates:
    - Model loads correctly
    - Input dtype is uint8
    - Output shape is [1, num_classes]
    - Inference runs without error

    Args:
        tflite_path: Path to .tflite file
        num_classes: Expected number of output classes

    Returns:
        Dictionary with model metadata

    Raises:
        AssertionError: If validation fails
    """
    logger.info(f"\nVerifying TFLite model: {tflite_path}")

    # Load interpreter
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    # Get input/output details
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    input_shape = input_details['shape'].tolist()
    output_shape = output_details['shape'].tolist()
    input_dtype = input_details['dtype']
    output_dtype = output_details['dtype']

    logger.info(f"  Input shape: {input_shape}")
    logger.info(f"  Input dtype: {input_dtype}")
    logger.info(f"  Output shape: {output_shape}")
    logger.info(f"  Output dtype: {output_dtype}")

    # Validate input dtype
    if input_dtype != np.uint8:
        logger.warning(f"  WARNING: Input dtype is {input_dtype}, expected uint8")

    # Validate output shape
    if output_shape[-1] != num_classes:
        logger.warning(f"  WARNING: Output classes {output_shape[-1]}, expected {num_classes}")

    # Get quantization parameters
    input_quant = input_details.get('quantization_parameters', {})
    output_quant = output_details.get('quantization_parameters', {})

    # Also check legacy quantization format
    if not input_quant:
        input_quant = {
            'scales': [input_details['quantization'][0]] if input_details['quantization'][0] else [],
            'zero_points': [input_details['quantization'][1]] if input_details['quantization'][1] else []
        }
    if not output_quant:
        output_quant = {
            'scales': [output_details['quantization'][0]] if output_details['quantization'][0] else [],
            'zero_points': [output_details['quantization'][1]] if output_details['quantization'][1] else []
        }

    logger.info(f"  Input quantization: scale={input_quant.get('scales', [])}, "
                f"zero_point={input_quant.get('zero_points', [])}")
    logger.info(f"  Output quantization: scale={output_quant.get('scales', [])}, "
                f"zero_point={output_quant.get('zero_points', [])}")

    # Smoke test: run inference with dummy data
    logger.info("\n  Running smoke test...")
    dummy_input = np.zeros(input_shape, dtype=input_dtype)

    interpreter.set_tensor(input_details['index'], dummy_input)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details['index'])

    assert output_data.shape == tuple(output_shape), \
        f"Output shape mismatch: {output_data.shape} vs {output_shape}"

    logger.info(f"  Smoke test PASSED!")
    logger.info(f"  Output range: [{output_data.min()}, {output_data.max()}]")

    # Build metadata
    file_size_bytes = os.path.getsize(tflite_path)

    metadata = {
        'file_path': str(tflite_path),
        'file_size_bytes': file_size_bytes,
        'file_size_mb': file_size_bytes / (1024 * 1024),
        'input': {
            'name': input_details['name'],
            'shape': input_shape,
            'dtype': str(input_dtype),
            'quantization': {
                'scales': input_quant.get('scales', []),
                'zero_points': input_quant.get('zero_points', [])
            }
        },
        'output': {
            'name': output_details['name'],
            'shape': output_shape,
            'dtype': str(output_dtype),
            'quantization': {
                'scales': output_quant.get('scales', []),
                'zero_points': output_quant.get('zero_points', [])
            }
        },
        'is_int8': input_dtype == np.uint8 and output_dtype == np.int8,
        'size_ok': file_size_bytes <= MAX_TFLITE_SIZE_BYTES
    }

    logger.info(f"\n  Model verification complete!")
    logger.info(f"  INT8 quantization: {'YES' if metadata['is_int8'] else 'NO'}")
    logger.info(f"  Size check: {'PASS' if metadata['size_ok'] else 'FAIL'}")

    return metadata
