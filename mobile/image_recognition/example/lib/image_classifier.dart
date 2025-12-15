import 'dart:typed_data';
import 'dart:math' as math;
import 'package:flutter/services.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;

/// Resultado de clasificación de una sola imagen
class ClassificationResult {
  final List<Prediction> predictions;
  final int inferenceTimeMs;

  ClassificationResult({
    required this.predictions,
    required this.inferenceTimeMs,
  });

  bool get isEmpty => predictions.isEmpty;
}

/// Predicción individual con etiqueta y confianza
class Prediction {
  final String label;
  final double confidence;

  Prediction({
    required this.label,
    required this.confidence,
  });
}

/// Clasificador de imágenes multi-etiqueta usando TFLite
class ImageClassifier {
  Interpreter? _interpreter;
  List<String> _labels = [];
  List<double> _thresholds = [];

  // Parámetros del modelo
  static const int inputSize = 240;
  static const int numClasses = 7;

  // Parámetros de cuantización del output
  static const double outputScale = 0.0563;
  static const int outputZeroPoint = 1;

  bool get isInitialized => _interpreter != null;

  /// Inicializa el clasificador cargando el modelo y los metadatos
  Future<void> initialize() async {
    try {
      // 1. Cargar modelo TFLite
      _interpreter = await Interpreter.fromAsset(
        'assets/model_qat_int8.tflite',
        options: InterpreterOptions()..threads = 4,
      );

      // 2. Cargar etiquetas
      final labelsData = await rootBundle.loadString('assets/labels.txt');
      _labels = labelsData
          .split('\n')
          .where((line) => line.trim().isNotEmpty)
          .toList();

      // 3. Cargar thresholds desde inference_config.json
      // Por ahora hardcoded, pero se puede cargar del JSON
      _thresholds = [0.26, 0.31, 0.05, 0.06, 0.52, 0.34, 0.11];

      // 4. Alojar tensores
      _interpreter!.allocateTensors();

      print('ImageClassifier inicializado correctamente');
      print('  Modelo: model_qat_int8.tflite');
      print('  Input shape: ${_interpreter!.getInputTensor(0).shape}');
      print('  Output shape: ${_interpreter!.getOutputTensor(0).shape}');
      print('  Labels: ${_labels.length}');
    } catch (e) {
      print('Error al inicializar ImageClassifier: $e');
      rethrow;
    }
  }

  /// Clasifica una imagen desde bytes
  Future<ClassificationResult> classifyImage(Uint8List imageBytes) async {
    if (!isInitialized) {
      throw StateError('Clasificador no inicializado. Llama a initialize() primero.');
    }

    final stopwatch = Stopwatch()..start();

    try {
      // 1. Decodificar imagen
      final image = img.decodeImage(imageBytes);
      if (image == null) {
        throw Exception('No se pudo decodificar la imagen');
      }

      // 2. Preprocesar: Letterbox a 240x240
      final preprocessed = _letterbox(image, inputSize);

      // 3. Convertir a ByteBuffer uint8 RGB
      final input = _imageToByteBuffer(preprocessed);

      // 4. Ejecutar inferencia
      final output = Uint8List(numClasses);
      _interpreter!.run(input, output);

      // 5. Postprocesar: dequantizar + sigmoid + aplicar thresholds
      final predictions = _postprocess(output);

      stopwatch.stop();

      return ClassificationResult(
        predictions: predictions,
        inferenceTimeMs: stopwatch.elapsedMilliseconds,
      );
    } catch (e) {
      print('Error en clasificación: $e');
      rethrow;
    }
  }

  /// Preprocesamiento: Letterbox con padding negro para mantener aspect ratio
  img.Image _letterbox(img.Image image, int targetSize) {
    final w = image.width;
    final h = image.height;
    final scale = targetSize / math.max(w, h);
    final newW = (w * scale).round();
    final newH = (h * scale).round();

    // Resize manteniendo aspect ratio
    final resized = img.copyResize(
      image,
      width: newW,
      height: newH,
      interpolation: img.Interpolation.linear,
    );

    // Crear canvas con fondo negro
    final output = img.Image(width: targetSize, height: targetSize);
    img.fill(output, color: img.ColorRgb8(0, 0, 0));

    // Centrar la imagen
    final left = ((targetSize - newW) / 2).round();
    final top = ((targetSize - newH) / 2).round();
    img.compositeImage(output, resized, dstX: left, dstY: top);

    return output;
  }

  /// Convierte imagen a ByteBuffer uint8 RGB
  ByteBuffer _imageToByteBuffer(img.Image image) {
    final buffer = Uint8List(1 * inputSize * inputSize * 3);
    int pixelIndex = 0;

    for (int y = 0; y < inputSize; y++) {
      for (int x = 0; x < inputSize; x++) {
        final pixel = image.getPixel(x, y);
        buffer[pixelIndex++] = pixel.r.toInt();
        buffer[pixelIndex++] = pixel.g.toInt();
        buffer[pixelIndex++] = pixel.b.toInt();
      }
    }

    return buffer.buffer;
  }

  /// Postprocesamiento: dequantizar y sigmoid (devuelve TODAS las clases)
  List<Prediction> _postprocess(Uint8List output) {
    final predictions = <Prediction>[];

    for (int i = 0; i < numClasses; i++) {
      // Dequantizar: (int8_value - zero_point) * scale
      final int8Value = output[i].toSigned(8);
      final dequantized = (int8Value - outputZeroPoint) * outputScale;

      // Aplicar sigmoid
      final probability = _sigmoid(dequantized);

      // Agregar TODAS las clases (no solo las que superan threshold)
      predictions.add(Prediction(
        label: _labels[i],
        confidence: probability,
      ));
    }

    // Ordenar por confianza descendente
    predictions.sort((a, b) => b.confidence.compareTo(a.confidence));

    return predictions;
  }

  /// Función sigmoid
  double _sigmoid(double x) {
    return 1.0 / (1.0 + math.exp(-x));
  }

  /// Libera recursos
  void dispose() {
    _interpreter?.close();
    _interpreter = null;
  }
}
