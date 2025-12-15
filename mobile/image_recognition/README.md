# Image Recognition Plugin

Flutter FFI plugin for high-performance multi-label image recognition using a native C engine.

## Features

- **High Performance**: Uses FFI with persistent isolates and zero-copy data transfer
- **Optimized Architecture**:
  - Persistent isolate worker (no spawning overhead per inference)
  - `TransferableTypedData` for efficient memory transfer
  - Reusable native buffers to minimize allocations
  - Direct JPEG/PNG processing (no Dart-side decoding)
- **Multi-label Classification**: Returns multiple predictions with confidence scores
- **Flexible Initialization**: Support for both asset and file path initialization
- **Detailed Timing**: Optional timing breakdown (preprocess, inference, postprocess)
- **Platform Support**: Android and iOS

## Performance Target

Designed to achieve <200ms inference on mid-range devices through aggressive optimizations.

## Installation

Add to your `pubspec.yaml`:

```yaml
dependencies:
  image_recognition:
    path: path/to/image_recognition
```

Run:

```bash
flutter pub get
```

## Native Library Setup

### Android

Place your compiled native libraries in the following structure:

```
android/src/main/jniLibs/
├── arm64-v8a/libimage_recognition.so
├── armeabi-v7a/libimage_recognition.so
├── x86_64/libimage_recognition.so
└── x86/libimage_recognition.so
```

The `build.gradle` is already configured to package these libraries.

### iOS

Place your `.xcframework` in:

```
ios/Frameworks/image_recognition.xcframework
```

The framework should contain:
- `ios-arm64` (device)
- `ios-arm64-simulator` (Apple Silicon simulator)
- `ios-x86_64-simulator` (Intel Mac simulator)

The `podspec` is configured to link this framework statically.

## Usage

### Basic Example

```dart
import 'package:image_recognition/image_recognition.dart';
import 'dart:typed_data';

// Initialize the engine
final recognizer = ImageRecognition();

await recognizer.initFromAssets(
  modelAssetPath: 'assets/model.tflite',
  configAssetPath: 'assets/config.json',
  options: ImageRecognitionOptions(
    numThreads: 4,
    runWarmup: true,
  ),
);

// Analyze an image
Uint8List imageBytes = await getImageBytes(); // Your JPEG/PNG bytes
final result = await recognizer.analyze(imageBytes);

if (result.ok) {
  for (final prediction in result.predictions) {
    print('${prediction.label}: ${(prediction.confidence * 100).toStringAsFixed(1)}%');
  }

  if (result.isEmpty) {
    print('No labels detected with sufficient confidence');
  }

  // Optional: Check timing
  print('Total time: ${result.totalTimeUs! / 1000}ms');
} else {
  print('Error: ${result.errorMessage}');
}

// Cleanup
await recognizer.dispose();
```

### Initialize from File Paths

```dart
await recognizer.initFromFilePaths(
  modelPath: '/path/to/model.tflite',
  configPath: '/path/to/config.json',
  options: ImageRecognitionOptions(numThreads: 2),
);
```

### Error Handling

```dart
final result = await recognizer.analyze(imageBytes);

if (!result.ok) {
  switch (result.errorCode) {
    case ImageRecognitionError.notInitialized:
      print('Engine not initialized');
      break;
    case ImageRecognitionError.invalidInput:
      print('Invalid image data');
      break;
    case ImageRecognitionError.inferenceFailed:
      print('Inference failed');
      break;
    default:
      print('Unknown error: ${result.errorMessage}');
  }
}
```

## API Reference

### `ImageRecognition`

Main client class for image recognition operations.

#### Methods

- `initFromAssets({required String modelAssetPath, required String configAssetPath, ImageRecognitionOptions? options})`
  Initialize engine from Flutter assets.

- `initFromFilePaths({required String modelPath, required String configPath, ImageRecognitionOptions? options})`
  Initialize engine from file system paths.

- `analyze(Uint8List imageBytes) → Future<ImageRecognitionResult>`
  Analyze compressed image bytes (JPEG/PNG).

- `dispose()`
  Release all resources and shutdown worker isolate.

### `ImageRecognitionResult`

Result of an image analysis.

#### Properties

- `List<LabelPrediction> predictions` - List of detected labels with confidence
- `int errorCode` - Error code (0 = success)
- `String? errorMessage` - Optional error description
- `int? preprocessTimeUs` - Preprocessing time in microseconds
- `int? inferenceTimeUs` - Inference time in microseconds
- `int? postprocessTimeUs` - Postprocessing time in microseconds
- `int? totalTimeUs` - Total processing time
- `String? modelVersion` - Model version string
- `bool ok` - True if successful (errorCode == 0)
- `bool isEmpty` - True if no predictions found

### `LabelPrediction`

Single prediction result.

#### Properties

- `String label` - Predicted label name
- `double confidence` - Confidence score (0.0 to 1.0)

### `ImageRecognitionOptions`

Configuration options for engine initialization.

#### Properties

- `int? numThreads` - Number of threads for inference (null = auto)
- `bool runWarmup` - Run warmup after initialization (default: false)

## Native Engine Contract

The plugin expects a C library implementing the following interface (see `native/include/image_recognition.h`):

```c
int32_t image_rec_init(const char* model_path_utf8, const char* config_path_utf8);
int32_t image_rec_analyze_image_bytes(const uint8_t* bytes, int32_t length, ir_result_t* out_result);
void image_rec_free_result(ir_result_t* result);
void image_rec_shutdown(void);
int32_t image_rec_set_num_threads(int32_t n);
int32_t image_rec_warmup(void);
```

See the header file for full struct definitions and documentation.

## Architecture

### Component Breakdown

```
lib/
├── image_recognition.dart              # Public API exports
└── src/
    ├── api/
    │   └── image_recognition_client.dart  # Main client class
    ├── ffi/
    │   ├── bindings.g.dart                # Generated FFI bindings
    │   ├── dylib_loader.dart              # Platform-specific library loading
    │   └── native_types.dart              # FFI helpers
    ├── isolate/
    │   ├── worker.dart                    # Persistent isolate worker
    │   └── messages.dart                  # Isolate messaging protocol
    ├── models/
    │   ├── image_recognition_result.dart
    │   ├── label_prediction.dart
    │   ├── options.dart
    │   └── errors.dart
    └── util/
        ├── asset_extractor.dart           # Asset → file extraction
        └── logger.dart                    # Optional logging
```

### Performance Optimizations

1. **Persistent Isolate**: Worker isolate lives for the session, avoiding spawn overhead
2. **TransferableTypedData**: Zero-copy transfer of image bytes to worker
3. **Buffer Reuse**: Native buffers are allocated once and reused
4. **Compressed Input**: JPEG/PNG bytes passed directly to C engine (no Dart decoding)
5. **Minimal Copying**: Direct memory access via FFI

## Troubleshooting

### "Engine not initialized"

Make sure to call `initFromAssets()` or `initFromFilePaths()` before `analyze()`.

### "Symbol not found" (iOS)

The native framework may have been stripped. Ensure:
- Symbols have `visibility("default")` and `used` attributes (see header)
- Framework is properly linked in Xcode

### "UnsatisfiedLinkError" (Android)

Library not found. Check:
- `.so` files are in correct `jniLibs/<abi>/` directories
- ABI matches the device/emulator architecture
- `build.gradle` has `jniLibs.srcDirs` configured

### "Failed to load model"

- Verify model and config file paths are correct
- Check files are accessible and not corrupted
- Ensure engine supports the model format

### Slow performance

- Enable warmup: `ImageRecognitionOptions(runWarmup: true)`
- Increase threads: `ImageRecognitionOptions(numThreads: 4)`
- Check device CPU/thermal throttling
- Profile with timing breakdown (`result.totalTimeUs`)

## Example App

See `example/` for a complete demo app with:
- Camera and gallery image selection
- Image preview
- Real-time analysis
- Result display with confidence scores
- Timing breakdown

Run with:

```bash
cd example
flutter run
```

## Development

### Regenerating FFI Bindings

If you modify `native/include/image_recognition.h`:

```bash
dart run ffigen --config ffigen.yaml
```

Note: Requires LLVM/libclang. Install with:
```bash
# Ubuntu/Debian
sudo apt-get install libclang-dev

# macOS
brew install llvm
```

## License

See LICENSE file.

## Contributing

Contributions welcome! Please ensure:
- Code follows Dart/Flutter style guidelines
- Native code follows the C header contract
- Tests pass (when available)
- Performance optimizations are documented
