"""
Student model: EfficientNet-Lite B1 for multi-label classification.

This is the deployment model:
- Optimized for mobile/edge devices
- Smaller and faster than teacher
- Will be quantized to INT8 via QAT
"""

import tensorflow as tf
from tensorflow import keras
import logging

try:
    from efficientnet_lite import EfficientNetLiteB1
except ImportError:
    raise ImportError(
        "efficientnet-lite-keras not found. Install with:\n"
        "pip install git+https://github.com/sebastian-sz/efficientnet-lite-keras@main"
    )

logger = logging.getLogger(__name__)


def build_student_model(
    num_classes: int = 7,
    input_size: int = 240,
    dropout: float = 0.3,
    weights: str = 'imagenet'
) -> keras.Model:
    """
    Build EfficientNet-Lite B1 student model.

    Architecture:
    - Backbone: EfficientNet-Lite B1 (pretrained on ImageNet)
    - Head: GlobalAveragePooling + Dropout + Dense(num_classes)
    - Output: Logits (no sigmoid)

    Args:
        num_classes: Number of output classes
        input_size: Input image size (default 240 for Lite B1)
        dropout: Dropout rate in head
        weights: 'imagenet' for pretrained, None for random init

    Returns:
        Keras Model
    """
    # Input
    inputs = keras.Input(shape=(input_size, input_size, 3), name='input_student')

    # Backbone: EfficientNet-Lite B1
    # Expected input range: [-1, 1] (preprocessing: (x-127)/128)
    backbone = EfficientNetLiteB1(
        include_top=False,
        weights=weights,
        input_tensor=inputs,
        pooling=None
    )

    # Get backbone output
    x = backbone.output

    # Head
    x = keras.layers.GlobalAveragePooling2D(name='gap')(x)
    x = keras.layers.Dropout(dropout, name='dropout')(x)
    outputs = keras.layers.Dense(num_classes, activation=None, name='logits')(x)

    # Create model
    model = keras.Model(inputs=inputs, outputs=outputs, name='student_efficientnet_lite_b1')

    logger.info(f"Built student model: EfficientNet-Lite B1")
    logger.info(f"  Input size: {input_size}x{input_size}")
    logger.info(f"  Num classes: {num_classes}")
    logger.info(f"  Dropout: {dropout}")
    logger.info(f"  Weights: {weights}")
    logger.info(f"  Total params: {model.count_params():,}")

    return model


def freeze_backbone(model: keras.Model):
    """
    Freeze all backbone layers (for warmup phase).

    Args:
        model: Student model
    """
    # Freeze all layers except head
    for layer in model.layers:
        if layer.name in ['gap', 'dropout', 'logits']:
            layer.trainable = True
        else:
            layer.trainable = False

    trainable_count = sum([1 for layer in model.layers if layer.trainable])
    logger.info(f"Frozen backbone, trainable layers: {trainable_count}")


def unfreeze_top_layers(model: keras.Model, unfreeze_fraction: float = 0.4):
    """
    Unfreeze top N% of backbone layers for fine-tuning.

    Args:
        model: Student model
        unfreeze_fraction: Fraction of layers to unfreeze (0.4 = top 40%)
    """
    # Find backbone
    backbone = None
    for layer in model.layers:
        if 'efficientnet' in layer.name.lower() and hasattr(layer, 'layers'):
            backbone = layer
            break

    if backbone is None:
        # Backbone might be the model itself
        logger.warning("Could not find separate backbone, unfreezing all layers")
        for layer in model.layers:
            layer.trainable = True
        return

    # Count total layers in backbone
    total_layers = len(backbone.layers)
    num_to_unfreeze = int(total_layers * unfreeze_fraction)

    # Freeze all first
    for layer in backbone.layers:
        layer.trainable = False

    # Unfreeze top layers
    for layer in backbone.layers[-num_to_unfreeze:]:
        layer.trainable = True

    # Ensure head is trainable
    for layer in model.layers:
        if layer.name in ['gap', 'dropout', 'logits']:
            layer.trainable = True

    trainable_count = sum([1 for layer in model.layers if layer.trainable])
    logger.info(f"Unfrozen top {unfreeze_fraction:.0%} of backbone ({num_to_unfreeze} layers)")
    logger.info(f"Trainable layers: {trainable_count}")


def compile_student(
    model: keras.Model,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4
):
    """
    Compile student model.

    Args:
        model: Student model
        learning_rate: Learning rate
        weight_decay: Weight decay for AdamW
    """
    # Optimizer: AdamW with weight decay
    optimizer = keras.optimizers.AdamW(
        learning_rate=learning_rate,
        weight_decay=weight_decay
    )

    # Loss: Binary cross-entropy with logits (multi-label)
    loss = keras.losses.BinaryCrossentropy(from_logits=True)

    # Metrics
    metrics = [
        keras.metrics.BinaryAccuracy(name='accuracy'),
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
    ]

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )

    logger.info(f"Compiled student model:")
    logger.info(f"  Optimizer: AdamW (lr={learning_rate}, wd={weight_decay})")
    logger.info(f"  Loss: BinaryCrossentropy(from_logits=True)")
