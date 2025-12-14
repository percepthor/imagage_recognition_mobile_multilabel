# Sistema de Reconocimiento de Imágenes Multi-Label para Móviles

Sistema completo end-to-end para clasificación multi-label en dispositivos móviles, con entrenamiento optimizado usando teacher-student distillation y Quantization Aware Training (QAT).

## 🎯 Descripción

Este proyecto proporciona una solución completa para implementar clasificación multi-label en aplicaciones móviles (Android/iOS) con máximo rendimiento y mínimo tamaño de modelo.

### Características Principales

- **Sistema de Entrenamiento Completo**: Pipeline automático con teacher-student distillation
- **Optimización para Móvil**: Modelos cuantizados INT8 con QAT
- **Plugin Flutter Nativo**: Inferencia en C para máximo rendimiento
- **Preprocesamiento Compatible**: Letterbox idéntico entre entrenamiento e inferencia
- **Anti-Overfitting**: Estrategias SOTA para datasets pequeños (~1000 imágenes)

## 📂 Estructura del Proyecto

```
.
├── trainer/                    # Sistema de entrenamiento
│   └── training/
│       ├── src/
│       │   ├── data/          # Procesamiento de datos
│       │   ├── models/        # Teacher & Student models
│       │   ├── train/         # Lógica de entrenamiento
│       │   ├── eval/          # Métricas y evaluación
│       │   └── export/        # Exportación TFLite
│       ├── configs/           # Configuraciones
│       ├── Dockerfile         # Container para entrenamiento
│       └── README.md          # Documentación detallada
│
└── mobile/                     # Plugin Flutter
    └── image_recognition/
        ├── src/               # Motor C de inferencia
        ├── ios/               # Implementación iOS
        ├── android/           # Implementación Android
        └── example/           # App de ejemplo

```

## 🚀 Quick Start

### 1. Entrenar Modelo

```bash
cd trainer/training

# Opción A: Con Docker
docker build -t multilabel-trainer .
docker run --rm \
  -v /path/to/dataset:/data/dataset \
  -v /path/to/output:/out \
  multilabel-trainer \
  python -m src.cli train \
    --data_dir /data/dataset \
    --out_dir /out/run_001 \
    --config /app/configs/default.yaml

# Opción B: Sin Docker
pip install -r requirements.txt
python -m src.cli train \
  --data_dir /path/to/dataset \
  --out_dir /path/to/output \
  --config configs/default.yaml
```

### 2. Integrar en App Móvil

```dart
import 'package:image_recognition/image_recognition.dart';

// Inicializar
final recognizer = ImageRecognition();
await recognizer.initialize(
  modelPath: 'assets/model_qat_int8.tflite',
  configPath: 'assets/inference_config.json',
);

// Reconocer imagen
final results = await recognizer.recognize(imageBytes);
for (var result in results) {
  print('${result.label}: ${result.confidence}');
}
```

## 📊 Arquitectura del Sistema

### Pipeline de Entrenamiento (9 Pasos Automáticos)

1. **Parse & Validate** - Validación completa del dataset
2. **Build Datasets** - TF.data pipelines optimizados
3. **Train Teacher** - EfficientNet-B3 (2 fases)
4. **Train Student** - EfficientNet-Lite B1 con destilación
5. **Apply QAT** - Quantization Aware Training
6. **Optimize Thresholds** - Grid search en validation set
7. **Export TFLite** - Full-integer INT8 quantization
8. **Generate Metadata** - Configuración para inferencia
9. **Final Evaluation** - Métricas en test set

### Modelos

- **Teacher**: EfficientNet-B3 (300x300)
  - Mayor capacidad y precisión
  - Solo para entrenamiento

- **Student**: EfficientNet-Lite B1 (240x240)
  - Optimizado para móvil
  - Cuantizado INT8
  - ~3-4 MB de tamaño

## 🎓 Teacher-Student Distillation

El student aprende tanto de:
- **Hard Targets**: Etiquetas ground truth
- **Soft Targets**: Predicciones del teacher (con temperatura)

Loss combinada:
```
L = α * L_hard + (1-α) * T² * L_soft
```

Donde:
- α = 0.7 (peso para hard loss)
- T = 2.0 (temperatura)

## 📱 Inferencia Móvil

### Motor Nativo en C

```c
// Inicializar
ImageRecContext* ctx = image_rec_init(
    model_path,
    config_path
);

// Inferencia
ImageRecResult* results = NULL;
int num_results;
image_rec_recognize(
    ctx,
    image_data,
    width,
    height,
    channels,
    &results,
    &num_results
);
```

### Características del Motor

- ✅ Letterbox automático (mantiene aspect ratio)
- ✅ Normalización EfficientNet-Lite: `(x-127)/128`
- ✅ Inferencia INT8 con TFLite
- ✅ Umbrales optimizados (global + por clase)
- ✅ Soporte multi-threading
- ✅ Sin dependencias externas (solo TFLite)

## 📈 Optimizaciones

### Anti-Overfitting (Crítico para ~1000 imágenes)

- Transfer learning (pesos ImageNet)
- Freeze→unfreeze progresivo
- Dropout en heads
- AdamW con weight decay
- EarlyStopping + ReduceLROnPlateau
- Data augmentation SOTA
- Teacher-student distillation
- MixUp (opcional)

### Optimización de Tamaño

- Quantization Aware Training (QAT)
- Full-integer INT8 quantization
- Modelo final: ~3-4 MB
- Inferencia: solo CPU

## 📦 Formato de Dataset

```
dataset/
├── images/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
├── train.txt
├── val.txt
└── test.txt
```

Formato de archivos .txt:
```
imagen1.jpg,clase1|clase2|clase3
imagen2.jpg,clase2
imagen3.jpg,              # Sin etiquetas (válido)
```

## 📤 Salidas del Entrenamiento

El sistema genera todos los archivos necesarios:

1. ✅ `model_qat_int8.tflite` - Modelo cuantizado INT8
2. ✅ `labels.txt` - Clases en orden alfabético
3. ✅ `metrics.json` - Métricas completas (F1, precision, recall)
4. ✅ `threshold_recommendation.json` - Umbrales optimizados
5. ✅ `inference_config.json` - Configuración completa para móvil

## 🔧 Configuración

Ver `trainer/training/configs/default.yaml`:

```yaml
seed: 1337
num_classes: 7

teacher:
  input_size: 300
  batch_size: 16
  epochs_head: 10
  epochs_finetune: 30
  lr_head: 1.0e-3
  lr_finetune: 1.0e-4
  dropout: 0.4

student:
  input_size: 240
  batch_size: 32
  epochs_head: 10
  epochs_finetune: 40
  dropout: 0.3

distillation:
  alpha: 0.7
  temperature: 2.0

qat:
  epochs: 10
  lr: 1.0e-5
```

## 📚 Documentación

### Entrenamiento
- [README Completo](trainer/training/README.md) - Guía técnica detallada
- [Sistema Completo](trainer/SISTEMA_COMPLETO.md) - Resumen del sistema
- [Inicio Rápido](trainer/INICIO_RAPIDO.md) - Quick start en 5 minutos

### Mobile
- [Plugin README](mobile/image_recognition/README.md) - Documentación del plugin
- [Requerimientos](mobile/requerimientos) - Especificaciones técnicas

## ⏱️ Tiempos de Entrenamiento

Dataset de ~1000 imágenes:

- **CPU**: 2-3 horas
- **GPU (V100)**: 30-40 minutos
- **GPU (T4)**: 1-1.5 horas

## 🧪 Testing

```bash
# Test de preprocesamiento
cd trainer/training
python tests/test_letterbox.py

# Test del plugin móvil
cd mobile/image_recognition/example
flutter test
flutter run
```

## 📊 Métricas de Rendimiento

### Modelo
- Tamaño: ~3-4 MB (INT8)
- Precisión: F1 macro ~0.85-0.90 (dataset típico)
- Velocidad: ~50-100ms por imagen (móvil mid-range)

### Compatibilidad
- ✅ Android API 21+
- ✅ iOS 12+
- ✅ CPU-only (sin GPU requerida)

## 🤝 Contribuir

Este es un proyecto de la organización Percepthor para sistemas de reconocimiento de imágenes optimizados para móviles.

## 📄 Licencia

Ver archivo LICENSE

## 👨‍💻 Autor

**Felipe Lara**
- Email: felipe@lara.ac
- Organización: Percepthor

## 🙏 Agradecimientos

- TensorFlow & TFLite team
- EfficientNet authors
- Flutter team

---

**Percepthor** - Optimizando IA para el mundo móvil
