"""
Student model: EfficientNet-Lite4 for multi-label classification.

This is the deployment model:
- Optimized for mobile/edge devices
- Uses EfficientNet-Lite4 from efficientnet-lite-keras package
- Input size: 380x380
- Will be quantized to INT8 via QAT

Requires TF_USE_LEGACY_KERAS=1 environment variable.

Preprocessing for EfficientNet-Lite: (x - 127.0) / 128.0
Maps [0..255] to approximately [-1..1]
"""

import tensorflow as tf
from tensorflow import keras
from efficientnet_lite import EfficientNetLiteB0, EfficientNetLiteB1, EfficientNetLiteB2, EfficientNetLiteB3, EfficientNetLiteB4
import logging

logger = logging.getLogger(__name__)

# EfficientNet-Lite variants from efficientnet-lite-keras package
# https://github.com/sebastian-sz/efficientnet-lite-keras
EFFICIENTNET_LITE_MODELS = {
    'lite0': (EfficientNetLiteB0, 224),
    'lite1': (EfficientNetLiteB1, 240),
    'lite2': (EfficientNetLiteB2, 260),
    'lite3': (EfficientNetLiteB3, 280),
    'lite4': (EfficientNetLiteB4, 300),  # Can use up to 380
}


def build_student_model(
    num_classes: int = 7,
    input_size: int = 380,
    dropout: float = 0.3,
    weights: str = 'imagenet',
    variant: str = 'lite4'
) -> keras.Model:
    """
    Build EfficientNet-Lite4 student model.

    Architecture:
    - Backbone: EfficientNet-Lite4 (pretrained on ImageNet)
    - Head: GlobalAveragePooling + Dropout + Dense(num_classes)
    - Output: Logits (no sigmoid) for BinaryCrossentropy(from_logits=True)

    Preprocessing: (x - 127.0) / 128.0 (maps 0..255 to ~-1..1)
    This MUST match the mobile inference engine.

    Args:
        num_classes: Number of output classes
        input_size: Input image size (380 for lite4)
        dropout: Dropout rate in head
        weights: 'imagenet' for pretrained, None for random init
        variant: Which EfficientNet-Lite variant ('lite0' to 'lite4')

    Returns:
        Keras Model (Functional API)
    """
    if variant not in EFFICIENTNET_LITE_MODELS:
        raise ValueError(f"Unknown variant: {variant}. Choose from {list(EFFICIENTNET_LITE_MODELS.keys())}")

    model_class, default_size = EFFICIENTNET_LITE_MODELS[variant]

    logger.info(f"Building EfficientNet-{variant.upper()} with input size {input_size}x{input_size}")

    # Backbone: EfficientNet-Lite from efficientnet-lite-keras
    # Use input_shape instead of input_tensor for TF 2.10 compatibility
    backbone = model_class(
        include_top=False,
        weights=weights,
        input_shape=(input_size, input_size, 3),
        pooling=None
    )

    # Build model using Sequential-like approach
    inputs = backbone.input
    x = backbone.output

    # Head: GAP + Dropout + Dense(logits)
    x = keras.layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    x = keras.layers.Dropout(dropout, name='head_dropout')(x)
    outputs = keras.layers.Dense(num_classes, activation=None, name='logits')(x)

    # Create model
    model = keras.Model(inputs=inputs, outputs=outputs, name=f'student_efficientnet_{variant}')

    logger.info(f"Built student model: EfficientNet-{variant.upper()}")
    logger.info(f"  Input size: {input_size}x{input_size}")
    logger.info(f"  Num classes: {num_classes}")
    logger.info(f"  Dropout: {dropout}")
    logger.info(f"  Weights: {weights}")
    logger.info(f"  Total params: {model.count_params():,}")

    return model


def freeze_backbone(model: keras.Model):
    """
    Freeze all backbone layers (for warmup/head training phase).

    Args:
        model: Student model
    """
    # Find the EfficientNet backbone (it's a nested model)
    backbone = None
    for layer in model.layers:
        if hasattr(layer, 'layers') and len(layer.layers) > 10:
            backbone = layer
            break

    if backbone is not None:
        for layer in backbone.layers:
            layer.trainable = False
        logger.info(f"Frozen backbone: {backbone.name} ({len(backbone.layers)} layers)")
    else:
        # Fallback: freeze all except head layers
        for layer in model.layers:
            if layer.name not in ['global_avg_pool', 'head_dropout', 'logits']:
                layer.trainable = False
        logger.info("Frozen all non-head layers (backbone not found as nested model)")

    # Ensure head is trainable
    for layer in model.layers:
        if layer.name in ['global_avg_pool', 'head_dropout', 'logits']:
            layer.trainable = True

    trainable_count = sum([1 for layer in model.layers if layer.trainable])
    non_trainable = sum([1 for layer in model.layers if not layer.trainable])
    logger.info(f"Trainable layers: {trainable_count}, Non-trainable: {non_trainable}")


def unfreeze_top_layers(model: keras.Model, unfreeze_fraction: float = 0.4):
    """
    Unfreeze top N% of backbone layers for fine-tuning.

    Args:
        model: Student model
        unfreeze_fraction: Fraction of backbone layers to unfreeze (0.4 = top 40%)
    """
    # Find the backbone
    backbone = None
    for layer in model.layers:
        if hasattr(layer, 'layers') and len(layer.layers) > 10:
            backbone = layer
            break

    if backbone is not None:
        total_layers = len(backbone.layers)
        num_to_unfreeze = int(total_layers * unfreeze_fraction)

        # First freeze all backbone layers
        for layer in backbone.layers:
            layer.trainable = False

        # Unfreeze top layers
        for layer in backbone.layers[-num_to_unfreeze:]:
            layer.trainable = True

        logger.info(f"Unfrozen top {unfreeze_fraction:.0%} of backbone ({num_to_unfreeze}/{total_layers} layers)")
    else:
        # Fallback: unfreeze everything
        for layer in model.layers:
            layer.trainable = True
        logger.warning("Could not find backbone, unfreezing all layers")

    # Ensure head is always trainable
    for layer in model.layers:
        if layer.name in ['global_avg_pool', 'head_dropout', 'logits']:
            layer.trainable = True

    trainable_count = sum([1 for layer in model.layers if layer.trainable])
    logger.info(f"Total trainable layers: {trainable_count}")


def compile_student(
    model: keras.Model,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4
):
    """
    Compile student model with AdamW optimizer.

    Args:
        model: Student model
        learning_rate: Learning rate
        weight_decay: Weight decay for AdamW
    """
    # Use experimental.AdamW for TF 2.10 compatibility
    try:
        optimizer = keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=weight_decay
        )
    except AttributeError:
        optimizer = keras.optimizers.experimental.AdamW(
            learning_rate=learning_rate,
            weight_decay=weight_decay
        )

    # Multi-label: BinaryCrossentropy with logits
    loss = keras.losses.BinaryCrossentropy(from_logits=True)

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
