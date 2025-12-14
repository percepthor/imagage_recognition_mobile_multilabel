# Sistema de Entrenamiento Multi-Label con Teacher-Student Distillation y QAT

Sistema completo de entrenamiento para clasificación multi-label optimizado para dispositivos móviles.

## Características

- **Teacher-Student Distillation**: EfficientNet-B3 → EfficientNet-Lite B1
- **Quantization Aware Training (QAT)**: Modelo INT8 para móvil
- **Preprocesamiento compatible con móvil**: Letterbox idéntico a inferencia C
- **Optimización de umbrales**: Búsqueda automática de umbrales óptimos
- **Exportación completa**: TFLite INT8 + configuración de inferencia

## Arquitectura del Sistema

```
training/
├── Dockerfile              # Imagen Docker con todas las dependencias
├── requirements.txt        # Dependencias Python
├── configs/
│   └── default.yaml       # Configuración por defecto
├── src/
│   ├── cli.py            # CLI principal
│   ├── data/             # Procesamiento de datos
│   │   ├── parsing.py    # Parsing y validación de dataset
│   │   ├── preprocessing.py  # Letterbox y normalización
│   │   ├── augment.py    # Data augmentation
│   │   └── dataset.py    # TF.data pipelines
│   ├── models/           # Definiciones de modelos
│   │   ├── teacher.py    # EfficientNet-B3
│   │   ├── student.py    # EfficientNet-Lite B1
│   │   └── losses.py     # Pérdidas de destilación
│   ├── train/            # Lógica de entrenamiento (TODO)
│   │   ├── train_teacher.py
│   │   ├── train_student_distill.py
│   │   └── train_student_qat.py
│   ├── eval/             # Evaluación y métricas
│   │   ├── metrics.py    # Métricas multi-label
│   │   └── thresholds.py # Optimización de umbrales
│   └── export/           # Exportación
│       ├── tflite_export.py  # Conversión a TFLite INT8
│       └── metadata.py   # Generación de metadata
└── tests/                # Tests unitarios
    ├── test_letterbox.py
    ├── test_parsing.py
    └── smoke_tflite.py
```

## Formato de Dataset

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
imagen3.jpg,             # Sin etiquetas (válido)
imagen4.jpg              # Sin etiquetas (válido)
```

## Salidas Obligatorias

El sistema genera en `--out_dir`:

1. **model_qat_int8.tflite**: Modelo cuantizado INT8 listo para móvil
2. **labels.txt**: Clases en orden exacto del vector de salida
3. **metrics.json**: Métricas de evaluación
   - precision_macro, recall_macro, f1_macro, f1_micro
   - Métricas por clase
   - No-label rate
4. **threshold_recommendation.json**: Umbrales optimizados
   - global_threshold
   - per_class_thresholds (opcional)
5. **inference_config.json**: Configuración completa para inferencia móvil

## Uso con Docker

### Construcción de la imagen

```bash
cd training
docker build -t multilabel-trainer .
```

### Entrenamiento

```bash
docker run --rm \
  -v /path/to/dataset:/data/dataset \
  -v /path/to/output:/out \
  multilabel-trainer \
  python -m src.cli train \
    --data_dir /data/dataset \
    --out_dir /out/run_001 \
    --config /app/configs/default.yaml
```

### Con GPU

```bash
docker run --rm --gpus all \
  -v /path/to/dataset:/data/dataset \
  -v /path/to/output:/out \
  multilabel-trainer \
  python -m src.cli train \
    --data_dir /data/dataset \
    --out_dir /out/run_001 \
    --config /app/configs/default.yaml
```

## Uso sin Docker

### Instalación

```bash
cd training
pip install -r requirements.txt
```

### Entrenamiento

```bash
python -m src.cli train \
  --data_dir ../dataset \
  --out_dir ../outputs/run_001 \
  --config configs/default.yaml
```

## Pipeline de Entrenamiento

El sistema ejecuta las siguientes fases:

### 1. Parsing y Validación del Dataset
- Lee train/val/test.txt
- Construye vocabulario de clases (orden alfabético)
- Valida existencia de imágenes
- Detecta duplicados entre splits
- Reporta estadísticas

### 2. Entrenamiento del Teacher (EfficientNet-B3)
- **Fase A**: Warmup del head (backbone congelado, 5-10 epochs)
- **Fase B**: Fine-tuning parcial (descongelar último 30%, LR bajo)
- Regularización: Dropout, AdamW, EarlyStopping
- Input: 300x300, [0, 255]

### 3. Entrenamiento del Student con Destilación
- **Fase A**: Warmup del head (5-10 epochs)
- **Fase B**: Fine-tuning con destilación (descongelar último 40%)
- Loss combinada: `L = α * L_hard + (1-α) * T² * L_soft`
  - α = 0.7 (hard loss weight)
  - T = 2.0 (temperature)
- Input: 240x240, [-1, 1]

### 4. Quantization Aware Training (QAT)
- Aplicar QAT a todo el modelo student
- Fine-tune con LR muy bajo (1e-5)
- 5-15 epochs
- Mantener pérdida de destilación activa

### 5. Exportación a TFLite INT8
- Representative dataset: 200 muestras de train
- Full-integer quantization
- Input: uint8 [0, 255]
- Output: int8 (logits cuantizados)

### 6. Optimización de Umbrales
- Grid search en VAL set
- Objetivo: maximizar F1 macro
- Umbrales por clase (opcional, solo si mejora >0.5%)

### 7. Evaluación Final
- Métricas en TEST set
- Con umbrales optimizados

## Configuración

Editar `configs/default.yaml` para ajustar:

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
  lr_head: 1.0e-3
  lr_finetune: 1.0e-4
  dropout: 0.3

distillation:
  alpha: 0.7
  temperature: 2.0

qat:
  epochs: 10
  lr: 1.0e-5

augmentation:
  random_flip_horizontal: true
  random_rotation_factor: 0.03
  random_zoom_factor: 0.10
  color_jitter:
    brightness: 0.15
    contrast: 0.15
    saturation: 0.15

threshold_search:
  grid_min: 0.05
  grid_max: 0.95
  grid_step: 0.01
  objective: f1_macro
```

## Preprocesamiento (Compatibilidad Móvil)

### Letterbox (crítico para compatibilidad)

```python
# Algoritmo EXACTO usado en entrenamiento y móvil:
# 1. s = 240 / max(h, w)
# 2. Resize a (round(h*s), round(w*s)) con bilinear
# 3. Pegar centrado en canvas negro 240x240
# 4. NO crop, solo letterbox
```

### Normalización

- **Student (EfficientNet-Lite)**: `(x - 127.0) / 128.0`  →  [-1, 1]
- **Teacher (EfficientNet-B3)**: Mantener [0, 255] (preprocessing interno)

## Tests

```bash
# Test de letterbox
python tests/test_letterbox.py

# Test de parsing
python tests/test_parsing.py

# Smoke test de TFLite
python tests/smoke_tflite.py
```

## Estado de Implementación

### ✅ Implementado
- [x] Estructura del proyecto
- [x] Parsing y validación de dataset
- [x] Preprocesamiento (letterbox + normalización)
- [x] Data augmentation
- [x] TF.data pipelines
- [x] Definición de modelos (teacher + student)
- [x] Pérdidas de destilación
- [x] Métricas multi-label
- [x] Optimización de umbrales
- [x] Exportación a TFLite INT8
- [x] Generación de metadata
- [x] CLI básico
- [x] Tests de letterbox

### ⚠️ TODO (Requiere Implementación)
- [ ] `src/train/train_teacher.py`: Lógica completa de entrenamiento del teacher
- [ ] `src/train/train_student_distill.py`: Destilación student
- [ ] `src/train/train_student_qat.py`: QAT del student
- [ ] `src/train/callbacks.py`: Callbacks personalizados
- [ ] `src/eval/reports.py`: Generación de reportes visuales
- [ ] Tests completos

## Siguiente Paso: Implementar Módulos de Entrenamiento

Para completar el sistema, implementar:

### 1. `src/train/train_teacher.py`

```python
def train_teacher(config, train_ds, val_ds, out_dir):
    # 1. Construir modelo
    model = build_teacher_model(...)

    # 2. Fase warmup (head only)
    freeze_backbone(model)
    compile_teacher(model, lr=config['teacher']['lr_head'])
    model.fit(train_ds, validation_data=val_ds, epochs=config['teacher']['epochs_head'])

    # 3. Fine-tuning
    unfreeze_top_layers(model, 0.3)
    compile_teacher(model, lr=config['teacher']['lr_finetune'])
    model.fit(train_ds, validation_data=val_ds, epochs=config['teacher']['epochs_finetune'],
              callbacks=[EarlyStopping(...), ReduceLROnPlateau(...)])

    # 4. Guardar
    model.save(out_dir / 'teacher_model.h5')
    return model
```

### 2. `src/train/train_student_distill.py`

Ver especificaciones en el documento de requerimientos.

### 3. `src/train/train_student_qat.py`

```python
import tensorflow_model_optimization as tfmot

def apply_qat(student_model, train_ds, val_ds, config):
    # 1. Aplicar QAT
    qat_model = tfmot.quantization.keras.quantize_model(student_model)

    # 2. Compilar con LR muy bajo
    qat_model.compile(...)

    # 3. Fine-tune
    qat_model.fit(train_ds, validation_data=val_ds, ...)

    return qat_model
```

## Desarrollado por

Felipe Lara - felipe@lara.ac
