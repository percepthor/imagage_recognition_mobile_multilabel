# Sistema de Entrenamiento Multi-Label para Mobile

Sistema completo de entrenamiento con teacher-student distillation, QAT y exportacion a TFLite INT8.

## Arquitectura

```
trainer/training/
├── Dockerfile              # TensorFlow 2.16.1-gpu
├── docker-compose.yml      # Configuracion multi-GPU
├── requirements.txt
├── configs/
│   └── default.yaml        # Config por defecto
└── src/
    ├── cli.py              # Pipeline principal
    ├── data/               # Parsing y datasets
    ├── models/             # Teacher y Student
    ├── train/              # Training con multi-GPU
    ├── eval/               # Metricas y thresholds
    └── export/             # TFLite export
```

## Uso Rapido

### 1. Preparar Dataset

Estructura requerida:
```
tu_dataset/
├── images/
│   ├── img1.jpg
│   └── ...
├── train_labels.csv
├── val_labels.csv
├── test_labels.csv
└── config.yaml          # Opcional: config especifico
```

Formato CSV:
```csv
filepath,labels
imagen1.jpg,clase1|clase2
imagen2.jpg,clase3
imagen3.jpg,clase1
```

### 2. Crear Config (si es necesario)

Copiar y ajustar `configs/default.yaml`:
```yaml
num_classes: 7

multi_gpu:
  enabled: true    # Para usar todas las GPUs

teacher:
  batch_size: 64   # Escalar segun GPUs

student:
  batch_size: 128
```

### 3. Entrenar

```bash
cd trainer/training

# Construir imagen
docker compose build

# Entrenar (ajustar rutas)
DATA_DIR=/ruta/a/tu/dataset \
OUTPUT_DIR=/ruta/a/tu/dataset/output \
CONFIG_FILE=/ruta/a/tu/dataset/config.yaml \
docker compose up
```

## Variables de Entorno

| Variable | Descripcion | Default |
|----------|-------------|---------|
| `DATA_DIR` | Ruta al dataset | `./data` |
| `OUTPUT_DIR` | Ruta para outputs | `./output` |
| `CONFIG_FILE` | Config YAML | `./configs/default.yaml` |

## Pipeline de Entrenamiento

1. **Parse Dataset** - Valida y carga datos
2. **Build Datasets** - tf.data pipelines optimizados
3. **Train Teacher** (EfficientNet-B3)
   - Fase A: Warmup head
   - Fase B: Fine-tune 30% superior
4. **Train Student** (EfficientNet-Lite B1)
   - Destilacion del teacher
   - Loss: `0.7*hard + 0.3*T²*soft`
5. **QAT** - Quantization Aware Training
6. **Optimize Thresholds** - Grid search en validation
7. **Export TFLite INT8** - Cuantizado para mobile
8. **Evaluate** - Metricas en test set

## Outputs

```
output/
├── model_qat_int8.tflite         # Modelo para mobile
├── labels.txt                     # Clases en orden
├── metrics.json                   # Metricas finales
├── threshold_recommendation.json  # Umbrales optimizados
├── inference_config.json          # Config para inferencia
├── teacher_*.h5                   # Checkpoints teacher
└── student_*.h5                   # Checkpoints student
```

## Multi-GPU

El sistema usa `MirroredStrategy` automaticamente cuando:
- `multi_gpu.enabled: true` en config
- Hay GPUs disponibles

Los batch sizes se distribuyen entre GPUs automaticamente.

## Sin Docker

```bash
cd trainer/training
pip install -r requirements.txt

python -m src.cli train \
  --data_dir /ruta/dataset \
  --out_dir /ruta/output \
  --config configs/default.yaml
```
