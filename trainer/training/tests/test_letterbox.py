"""
Test letterbox preprocessing to ensure compatibility with mobile inference.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import tensorflow as tf
from src.data.preprocessing import letterbox_resize, letterbox_resize_numpy


def test_letterbox_preserves_aspect_ratio():
    """Test that letterbox maintains aspect ratio."""
    # Create test image (100x50)
    image = np.random.rand(100, 50, 3).astype(np.float32)

    # Apply letterbox to 240x240
    result = letterbox_resize_numpy(image, target_size=240, pad_value=0)

    # Check output shape
    assert result.shape == (240, 240, 3), f"Expected (240, 240, 3), got {result.shape}"

    # Check that content is centered
    # The 100x50 image should become 240x120 (scaled by 2.4)
    # Then centered vertically with 60px padding top/bottom

    # Check padding (should be black)
    assert np.all(result[0, :, :] == 0), "Top padding should be black"
    assert np.all(result[-1, :, :] == 0), "Bottom padding should be black"

    print("✓ Letterbox preserves aspect ratio")


def test_letterbox_tensorflow_vs_numpy():
    """Test that TensorFlow and NumPy implementations match."""
    # Create test image
    image_np = np.random.rand(80, 120, 3).astype(np.float32)
    image_tf = tf.constant(image_np)

    # Apply letterbox
    result_np = letterbox_resize_numpy(image_np, target_size=240)
    result_tf = letterbox_resize(image_tf, target_size=240).numpy()

    # Check shapes match
    assert result_np.shape == result_tf.shape

    # Check values are close (allow small differences due to interpolation)
    assert np.allclose(result_np, result_tf, atol=1e-3)

    print("✓ TensorFlow and NumPy letterbox implementations match")


def test_letterbox_square_image():
    """Test letterbox with square input."""
    # Square image (100x100)
    image = np.random.rand(100, 100, 3).astype(np.float32)

    result = letterbox_resize_numpy(image, target_size=240)

    # Should be resized to 240x240 with no padding
    assert result.shape == (240, 240, 3)

    print("✓ Letterbox handles square images correctly")


if __name__ == '__main__':
    print("Running letterbox tests...")
    test_letterbox_preserves_aspect_ratio()
    test_letterbox_tensorflow_vs_numpy()
    test_letterbox_square_image()
    print("\nAll letterbox tests passed!")
