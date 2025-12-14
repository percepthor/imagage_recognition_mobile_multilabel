"""
TFLite export with full-integer (INT8) quantization.

Implements:
- QAT model to TFLite conversion
- Representative dataset for calibration
- Full-integer quantization with uint8 input and int8 output
"""

import tensorflow as tf
import numpy as np
import logging
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger(__name__)


def create_representative_dataset_generator(
    data: List[Tuple[str, List[int]]],
    num_samples: int = 200,
    preprocess_fn=None
):
    """
    Create representative dataset generator for TFLite quantization calibration.

    Args:
        data: List of (image_path, labels) tuples
        num_samples: Number of samples to use
        preprocess_fn: Preprocessing function (image_path -> tensor)

    Returns:
        Generator function
    """
    # Sample subset of data
    sampled_data = data[:num_samples] if len(data) > num_samples else data

    def representative_dataset():
        for img_path, _ in sampled_data:
            # Preprocess image
            if preprocess_fn:
                image = preprocess_fn(img_path)
            else:
                # Default: read and preprocess for student
                from ..data.preprocessing import preprocess_image_path_for_student
                image = preprocess_image_path_for_student(img_path, target_size=240)

            # Add batch dimension and yield
            yield [np.expand_dims(image.numpy(), axis=0).astype(np.float32)]

    return representative_dataset


def export_model_to_tflite_int8(
    model: tf.keras.Model,
    representative_dataset_gen,
    output_path: str,
    input_type: str = 'uint8',
    output_type: str = 'int8'
) -> dict:
    """
    Export Keras model to TFLite with full-integer quantization.

    Args:
        model: QAT-trained Keras model
        representative_dataset_gen: Representative dataset generator
        output_path: Path to save .tflite file
        input_type: Input tensor type ('uint8' or 'int8')
        output_type: Output tensor type ('int8' or 'uint8')

    Returns:
        Dictionary with export metadata
    """
    logger.info(f"Exporting model to TFLite INT8...")
    logger.info(f"  Output path: {output_path}")
    logger.info(f"  Input type: {input_type}")
    logger.info(f"  Output type: {output_type}")

    # Create TFLite converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Enable quantization optimizations
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Set representative dataset for calibration
    converter.representative_dataset = representative_dataset_gen

    # Force full-integer quantization
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

    # Set input/output types
    if input_type == 'uint8':
        converter.inference_input_type = tf.uint8
    elif input_type == 'int8':
        converter.inference_input_type = tf.int8
    else:
        raise ValueError(f"Unknown input_type: {input_type}")

    if output_type == 'uint8':
        converter.inference_output_type = tf.uint8
    elif output_type == 'int8':
        converter.inference_output_type = tf.int8
    else:
        raise ValueError(f"Unknown output_type: {output_type}")

    # Convert
    try:
        tflite_model = converter.convert()
    except Exception as e:
        logger.error(f"Failed to convert model: {e}")
        raise

    # Save to file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'wb') as f:
        f.write(tflite_model)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"  Saved TFLite model: {file_size_mb:.2f} MB")

    # Get quantization details
    interpreter = tf.lite.Interpreter(model_path=str(output_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    metadata = {
        'file_path': str(output_path),
        'file_size_mb': file_size_mb,
        'input': {
            'name': input_details['name'],
            'shape': input_details['shape'].tolist(),
            'dtype': str(input_details['dtype']),
            'quantization': {
                'scale': float(input_details['quantization'][0]) if input_details['quantization'][0] else None,
                'zero_point': int(input_details['quantization'][1]) if input_details['quantization'][1] else None
            }
        },
        'output': {
            'name': output_details['name'],
            'shape': output_details['shape'].tolist(),
            'dtype': str(output_details['dtype']),
            'quantization': {
                'scale': float(output_details['quantization'][0]) if output_details['quantization'][0] else None,
                'zero_point': int(output_details['quantization'][1]) if output_details['quantization'][1] else None
            }
        }
    }

    logger.info(f"\nQuantization details:")
    logger.info(f"  Input:  dtype={metadata['input']['dtype']}, "
                f"scale={metadata['input']['quantization']['scale']}, "
                f"zero_point={metadata['input']['quantization']['zero_point']}")
    logger.info(f"  Output: dtype={metadata['output']['dtype']}, "
                f"scale={metadata['output']['quantization']['scale']}, "
                f"zero_point={metadata['output']['quantization']['zero_point']}")

    return metadata


def verify_tflite_model(tflite_path: str, test_image_path: str = None):
    """
    Verify TFLite model can be loaded and run inference.

    Args:
        tflite_path: Path to .tflite file
        test_image_path: (Optional) Path to test image for inference

    Returns:
        True if verification passes
    """
    logger.info(f"Verifying TFLite model: {tflite_path}")

    try:
        # Load interpreter
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()

        # Get input/output details
        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]

        logger.info(f"  Input shape: {input_details['shape']}")
        logger.info(f"  Output shape: {output_details['shape']}")

        # If test image provided, run inference
        if test_image_path:
            from ..data.preprocessing import preprocess_image_path_for_student

            # Preprocess image
            image = preprocess_image_path_for_student(test_image_path, target_size=240)
            image = image.numpy()

            # Convert to uint8 if needed
            if input_details['dtype'] == np.uint8:
                # De-normalize from [-1, 1] back to [0, 255]
                image = ((image + 1.0) * 127.5).astype(np.uint8)

            # Add batch dimension
            input_data = np.expand_dims(image, axis=0)

            # Set input tensor
            interpreter.set_tensor(input_details['index'], input_data)

            # Run inference
            interpreter.invoke()

            # Get output
            output_data = interpreter.get_tensor(output_details['index'])

            logger.info(f"  Test inference successful")
            logger.info(f"  Output shape: {output_data.shape}")
            logger.info(f"  Output dtype: {output_data.dtype}")
            logger.info(f"  Output range: [{output_data.min()}, {output_data.max()}]")

        logger.info(f"  Verification passed!")
        return True

    except Exception as e:
        logger.error(f"  Verification failed: {e}")
        return False
