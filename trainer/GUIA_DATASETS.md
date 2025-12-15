# Guia para Entrenar con Nuevos Datasets

## Estructura del Dataset

```
mi_dataset/
├── images/
│   ├── imagen1.jpg
│   ├── imagen2.jpg
│   └── ...
├── train_labels.csv    # ~80% de datos
├── val_labels.csv      # ~10% de datos
├── test_labels.csv     # ~10% de datos
└── config.yaml         # Config especifico (opcional)
```

## Formato de Labels CSV

```csv
filepath,labels
imagen1.jpg,clase_a|clase_b
imagen2.jpg,clase_c
imagen3.jpg,clase_a|clase_c|clase_d
```

- `filepath`: nombre del archivo en `images/`
- `labels`: clases separadas por `|` (pipe)
- Multi-label: una imagen puede tener multiples clases

## Config Especifico del Dataset

Crear `config.yaml` en la carpeta del dataset:

```yaml
seed: 1337
num_classes: 7  # Ajustar al numero de clases

multi_gpu:
  enabled: true
  strategy: mirrored

teacher:
  input_size: 300
  batch_size: 64      # 16 x num_gpus
  epochs_head: 10
  epochs_finetune: 30
  lr_head: 1.0e-3
  lr_finetune: 1.0e-4
  dropout: 0.4
  weight_decay: 1.0e-4
  early_stopping_patience: 10
  reduce_lr_patience: 5
  reduce_lr_factor: 0.5

student:
  input_size: 240
  batch_size: 128     # 32 x num_gpus
  epochs_head: 10
  epochs_finetune: 40
  lr_head: 1.0e-3
  lr_finetune: 1.0e-4
  dropout: 0.3
  weight_decay: 1.0e-4
  early_stopping_patience: 10
  reduce_lr_patience: 5
  reduce_lr_factor: 0.5

distillation:
  alpha: 0.7
  temperature: 2.0

qat:
  epochs: 10
  lr: 1.0e-5
  early_stopping_patience: 5

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
  per_class_enabled: true
  per_class_improvement_threshold: 0.005

tflite_export:
  input_type: uint8
  output_type: int8
  representative_dataset_size: 200
```

## Comando de Entrenamiento

```bash
cd /path/to/imagage_recognition_mobile_multilabel/trainer/training

# Con docker-compose
DATA_DIR=/path/to/mi_dataset \
OUTPUT_DIR=/path/to/mi_dataset/output \
CONFIG_FILE=/path/to/mi_dataset/config.yaml \
docker compose up
```

## Ejemplo Completo

```bash
# Dataset en: /datos/mi_proyecto/dataset_v1
# Trainer en: /home/user/imagage_recognition_mobile_multilabel

cd /home/user/imagage_recognition_mobile_multilabel/trainer/training

docker compose build

DATA_DIR=/datos/mi_proyecto/dataset_v1 \
OUTPUT_DIR=/datos/mi_proyecto/dataset_v1/output \
CONFIG_FILE=/datos/mi_proyecto/dataset_v1/config.yaml \
docker compose up
```

## Ajustes Segun Tamano del Dataset

### Dataset pequeno (<1000 imagenes)
```yaml
teacher:
  epochs_head: 5
  epochs_finetune: 20
  early_stopping_patience: 5

student:
  epochs_head: 5
  epochs_finetune: 25
```

### Dataset grande (>10000 imagenes)
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

## Ajustes Segun Numero de GPUs

| GPUs | Teacher Batch | Student Batch |
|------|---------------|---------------|
| 1    | 16            | 32            |
| 2    | 32            | 64            |
| 4    | 64            | 128           |
| 8    | 128           | 256           |

## Verificar Resultados

Los outputs se guardan en `OUTPUT_DIR/`:

```bash
ls -la /path/to/output/

# Archivos principales:
# - model_qat_int8.tflite  -> Modelo para mobile
# - labels.txt             -> Lista de clases
# - metrics.json           -> Metricas de test
# - inference_config.json  -> Config para app
```

## Troubleshooting

### Error: "Expected X classes, found Y"
- Verificar `num_classes` en config.yaml

### Out of Memory
- Reducir batch_size
- Deshabilitar multi_gpu si hay problemas

### Entrenamiento lento
- Habilitar multi_gpu
- Usar GPU con mas VRAM
- Reducir epochs para pruebas rapidas
