"""
Teacher model: EfficientNet-B3 for multi-label classification.

This model serves as the teacher in knowledge distillation.
- Larger capacity
- Higher accuracy
- Not optimized for mobile deployment
"""

import tensorflow as tf
from tensorflow import keras
import logging

logger = logging.getLogger(__name__)


def build_teacher_model(
    num_classes: int = 7,
    input_size: int = 300,
    dropout: float = 0.4,
    weights: str = 'imagenet'
) -> keras.Model:
    """
    Build EfficientNet-B3 teacher model.

    Architecture:
    - Backbone: EfficientNet-B3 (pretrained on ImageNet)
    - Head: GlobalAveragePooling + Dropout + Dense(num_classes)
    - Output: Logits (no sigmoid, for stable training with from_logits=True)

    Args:
        num_classes: Number of output classes
        input_size: Input image size (default 300 for B3)
        dropout: Dropout rate in head
        weights: 'imagenet' for pretrained, None for random init

    Returns:
        Keras Model
    """
    # Input
    inputs = keras.Input(shape=(input_size, input_size, 3), name='input_teacher')

    # Backbone: EfficientNet-B3
    # Note: EfficientNetB3 from Keras Applications includes preprocessing
    # It expects input in range [0, 255]
    backbone = keras.applications.EfficientNetB3(
        include_top=False,
        weights=weights,
        input_tensor=inputs,
        pooling=None  # We'll add our own pooling
    )

    # Get backbone output
    x = backbone.output

    # Head
    x = keras.layers.GlobalAveragePooling2D(name='gap')(x)
    x = keras.layers.Dropout(dropout, name='dropout')(x)
    outputs = keras.layers.Dense(num_classes, activation=None, name='logits')(x)

    # Create model
    model = keras.Model(inputs=inputs, outputs=outputs, name='teacher_efficientnetb3')

    logger.info(f"Built teacher model: EfficientNet-B3")
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
        model: Teacher model
    """
    # Find the backbone layer (EfficientNet-B3)
    for layer in model.layers:
        if 'efficientnet' in layer.name.lower():
            layer.trainable = False
            logger.info(f"Frozen backbone: {layer.name}")
        elif layer.name in ['gap', 'dropout', 'logits']:
            layer.trainable = True
        else:
            # This might be the backbone
            layer.trainable = False

    # Alternative: freeze by layer name pattern
    for layer in model.layers:
        if 'block' in layer.name or 'stem' in layer.name:
            layer.trainable = False

    trainable_count = sum([1 for layer in model.layers if layer.trainable])
    logger.info(f"Trainable layers after freeze: {trainable_count}")


def unfreeze_top_layers(model: keras.Model, unfreeze_fraction: float = 0.3):
    """
    Unfreeze top N% of backbone layers for fine-tuning.

    Args:
        model: Teacher model
        unfreeze_fraction: Fraction of layers to unfreeze (0.3 = top 30%)
    """
    # Get backbone
    backbone = None
    for layer in model.layers:
        if 'efficientnetb3' in layer.name.lower():
            backbone = layer
            break

    if backbone is None:
        logger.warning("Could not find EfficientNetB3 backbone, unfreezing all layers")
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


def compile_teacher(
    model: keras.Model,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4
):
    """
    Compile teacher model with appropriate loss and optimizer.

    Args:
        model: Teacher model
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

    logger.info(f"Compiled teacher model:")
    logger.info(f"  Optimizer: AdamW (lr={learning_rate}, wd={weight_decay})")
    logger.info(f"  Loss: BinaryCrossentropy(from_logits=True)")
