# Manual de Entrenamiento - 15 Diciembre 2024

## Resumen del Sistema

Sistema de entrenamiento multi-label con **teacher-student distillation** y **Quantization Aware Training (QAT)** para exportar modelos TFLite INT8 optimizados para mobile.

### Arquitectura de Modelos

| Modelo | Arquitectura | Input Size | Parametros | Uso |
|--------|--------------|------------|------------|-----|
| Teacher | EfficientNet-B3 | 300x300 | ~12M | Entrenamiento (alta precision) |
| Student | EfficientNet-Lite1 | 240x240 | ~5M | Mobile (TFLite INT8) |

### Stack Tecnologico

- **TensorFlow**: 2.10.0-gpu
- **Keras**: 2.10 (legacy, bundled con TF)
- **efficientnet-lite-keras**: Para EfficientNet-Lite
- **tensorflow-model-optimization**: Para QAT
- **Docker**: Contenedor con soporte GPU

---

## Pipeline de Entrenamiento

El pipeline ejecuta 9 pasos automaticamente:

```
1. Parse Dataset        -> Valida estructura y carga labels
2. Build Datasets       -> tf.data pipelines optimizados
3. Train Teacher        -> EfficientNet-B3 (warmup + finetune)
4. Train Student        -> Distillation desde teacher
5. Apply QAT            -> Quantization Aware Training
6. Optimize Thresholds  -> Grid search en validation set
7. Export TFLite INT8   -> Modelo cuantizado
8. Generate Config      -> inference_config.json
9. Final Evaluation     -> Metricas en test set
```

---

## Estructura del Dataset Requerida

```
mi_dataset/
├── images/
│   ├── imagen1.jpg
│   ├── imagen2.png
│   └── ...
├── train_labels.csv    # ~80% datos
├── val_labels.csv      # ~10% datos
├── test_labels.csv     # ~10% datos
└── config.yaml         # Configuracion (opcional)
```

### Formato CSV de Labels

```csv
filepath,labels
imagen1.jpg,clase_a|clase_b
imagen2.jpg,clase_c
imagen3.jpg,clase_a|clase_c|clase_d
```

- `filepath`: Nombre del archivo relativo a `images/`
- `labels`: Clases separadas por `|` (pipe)
- Una imagen puede tener multiples clases (multi-label)

---

## Como Ejecutar el Entrenamiento

### Prerequisitos

1. NVIDIA GPU con drivers instalados
2. Docker con soporte nvidia-container-toolkit
3. Dataset preparado segun estructura anterior

### Paso 1: Clonar/Ubicar el Trainer

```bash
cd /path/to/imagage_recognition_mobile_multilabel/trainer/training
```

### Paso 2: Construir Imagen Docker

```bash
docker compose build
```

### Paso 3: Ejecutar Entrenamiento

```bash
DATA_DIR=/ruta/a/tu/dataset \
OUTPUT_DIR=/ruta/a/tu/dataset/output \
CONFIG_FILE=/ruta/a/tu/dataset/config.yaml \
docker compose up
```

**Variables de entorno:**

| Variable | Descripcion | Default |
|----------|-------------|---------|
| `DATA_DIR` | Ruta al dataset | `./data` |
| `OUTPUT_DIR` | Donde guardar outputs | `./output` |
| `CONFIG_FILE` | Config YAML personalizado | `./configs/default.yaml` |

### Ejemplo Real (Supernova)

```bash
cd /home/verstand/Documents/imagage_recognition_mobile_multilabel/trainer/training

docker compose build

DATA_DIR=/datos/percepthor/dataset_multilabel \
OUTPUT_DIR=/datos/percepthor/dataset_multilabel/output_15dic \
CONFIG_FILE=/datos/percepthor/dataset_multilabel/config.yaml \
docker compose up
```

---

## Configuracion (config.yaml)

Archivo de configuracion completo con todos los parametros:

```yaml
seed: 1337
num_classes: 7  # AJUSTAR al numero de clases de tu dataset

# Multi-GPU (si tienes multiples GPUs)
multi_gpu:
  enabled: true
  strategy: mirrored

# Teacher: EfficientNet-B3
teacher:
  input_size: 300
  batch_size: 16        # x4 si tienes 4 GPUs
  epochs_head: 10       # Warmup solo head
  epochs_finetune: 30   # Fine-tune top layers
  lr_head: 1.0e-3
  lr_finetune: 1.0e-4
  dropout: 0.4
  weight_decay: 1.0e-4
  early_stopping_patience: 10
  reduce_lr_patience: 5
  reduce_lr_factor: 0.5

# Student: EfficientNet-Lite
student:
  variant: lite1        # lite0, lite1, lite2, lite3, lite4
  input_size: 240       # Depende del variant
  batch_size: 16
  epochs_head: 10
  epochs_finetune: 40
  lr_head: 1.0e-3
  lr_finetune: 1.0e-4
  dropout: 0.3
  weight_decay: 1.0e-4
  early_stopping_patience: 10
  reduce_lr_patience: 5
  reduce_lr_factor: 0.5

# Distillation
distillation:
  alpha: 0.7            # 0.7 = 70% hard loss, 30% soft loss
  temperature: 2.0      # Suaviza logits del teacher

# QAT
qat:
  epochs: 10
  lr: 1.0e-5
  early_stopping_patience: 5

# Data Augmentation
augmentation:
  random_flip_horizontal: true
  random_rotation_factor: 0.03
  random_zoom_factor: 0.10
  color_jitter:
    brightness: 0.15
    contrast: 0.15
    saturation: 0.15
  gaussian_noise_stddev: 0.02
  cutout:
    num_patches: 2
    patch_size_ratio: 0.10
  mixup:
    enabled: false
    alpha: 0.2

# Threshold Optimization
threshold_search:
  grid_min: 0.05
  grid_max: 0.95
  grid_step: 0.01
  objective: f1_macro
  per_class_enabled: true
  per_class_improvement_threshold: 0.005

# TFLite Export
tflite_export:
  input_type: uint8     # Input: imagenes [0, 255]
  output_type: int8     # Output: cuantizado
  representative_dataset_size: 200
```

---

## Variantes de EfficientNet-Lite

| Variant | Input Size | TFLite Size | Uso Recomendado |
|---------|------------|-------------|-----------------|
| lite0 | 224x224 | ~4 MB | Dispositivos muy limitados |
| lite1 | 240x240 | ~5 MB | Balance recomendado |
| lite2 | 260x260 | ~6 MB | Mejor precision |
| lite3 | 280x280 | ~7 MB | Alta precision |
| lite4 | 300x300 | ~12 MB | Maxima precision |

Para cambiar variant, editar en config.yaml:
```yaml
student:
  variant: lite1  # Cambiar a lite0, lite2, etc.
  input_size: 240 # Ajustar segun tabla
```

---

## Outputs Generados

Despues del entrenamiento, en `OUTPUT_DIR/`:

```
output/
├── model_qat_int8.tflite         # Modelo para mobile (PRINCIPAL)
├── labels.txt                     # Lista de clases en orden
├── metrics.json                   # Metricas finales (F1, precision, recall)
├── threshold_recommendation.json  # Umbrales optimizados por clase
├── inference_config.json          # Config completo para app mobile
├── teacher_warmup.h5              # Checkpoint teacher fase 1
├── teacher_finetuned.h5           # Checkpoint teacher final
├── student_warmup.h5              # Checkpoint student fase 1
└── student_finetuned.h5           # Checkpoint student final
```

### Archivos Clave para Mobile

1. **model_qat_int8.tflite** - El modelo cuantizado INT8
2. **labels.txt** - Nombres de clases en orden (linea 0 = clase 0)
3. **inference_config.json** - Contiene:
   - Input/output details (shape, scale, zero_point)
   - Thresholds por clase
   - Preprocesamiento requerido

---

## Ajustes Segun Hardware

### Segun Numero de GPUs

| GPUs | Teacher Batch | Student Batch |
|------|---------------|---------------|
| 1 | 16 | 32 |
| 2 | 32 | 64 |
| 4 | 64 | 128 |
| 8 | 128 | 256 |

### Segun Tamano del Dataset

**Dataset pequeno (<1000 imagenes):**
```yaml
teacher:
  epochs_head: 5
  epochs_finetune: 20
  early_stopping_patience: 5
student:
  epochs_head: 5
  epochs_finetune: 25
```

**Dataset grande (>10000 imagenes):**
```yaml
teacher:
  epochs_head: 15
  epochs_finetune: 40
  batch_size: 128
student:
  epochs_head: 15
  epochs_finetune: 50
  batch_size: 256
```

---

## Ejecutar Sin Docker

Si prefieres ejecutar directamente:

```bash
cd /path/to/trainer/training

# Crear virtualenv
python -m venv venv
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar
python -m src.cli train \
  --data_dir /ruta/dataset \
  --out_dir /ruta/output \
  --config configs/default.yaml
```

---

## Troubleshooting

### Error: "Expected X classes, found Y"
- Verificar `num_classes` en config.yaml coincide con las clases del dataset

### Out of Memory (OOM)
- Reducir `batch_size` en teacher y student
- Deshabilitar `multi_gpu` si causa problemas

### QAT Falla
- El sistema usa Post-Training Quantization (PTQ) como fallback
- Modelo sigue siendo INT8 pero puede tener menos precision

### Entrenamiento Muy Lento
- Habilitar `multi_gpu.enabled: true`
- Verificar que Docker tiene acceso a GPUs: `docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi`

### Labels No Coinciden
- Verificar que los CSV usan exactamente los mismos nombres de clase
- El separador de labels es `|` (pipe), no coma

---

## Repetir el Entrenamiento

Para repetir exactamente el mismo entrenamiento:

1. **Usar el mismo seed**: `seed: 1337` en config.yaml
2. **Misma version de TF**: Dockerfile usa `tensorflow:2.10.0-gpu`
3. **Mismo dataset split**: train/val/test CSVs identicos
4. **Misma config**: Copiar el config.yaml usado

```bash
# Ejemplo: repetir entrenamiento del 15 de diciembre
DATA_DIR=/datos/percepthor/dataset_multilabel \
OUTPUT_DIR=/datos/percepthor/dataset_multilabel/output_repeticion \
CONFIG_FILE=/datos/percepthor/dataset_multilabel/config.yaml \
docker compose up
```

---

## Metricas Esperadas

Un entrenamiento exitoso deberia mostrar:

```
PIPELINE COMPLETE!
================================================================================

Mandatory outputs:
  ✓ model_qat_int8.tflite
  ✓ labels.txt
  ✓ metrics.json
  ✓ threshold_recommendation.json
  ✓ inference_config.json

Final test metrics:
  F1 Macro:  0.85+
  F1 Micro:  0.87+
  Precision: 0.83+
  Recall:    0.80+

Ready for mobile deployment!
```

Las metricas varian segun el dataset, pero F1 > 0.80 es un buen indicador.

---

## Contacto

Desarrollado por Felipe Lara para Percepthor.
