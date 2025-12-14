# Inicio Rápido - Sistema de Entrenamiento Multi-Label

## 🚀 En 5 Minutos

### 1. Preparar Dataset

Crear esta estructura:

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

Formato de .txt files:
```
imagen1.jpg,clase1|clase2|clase3
imagen2.jpg,clase2
imagen3.jpg,
```

### 2. Configurar

Editar `training/configs/default.yaml`:
- Ajustar `num_classes: 7` a tu número de clases
- Ajustar epochs según tamaño del dataset
- Ajustar batch_size según tu memoria

### 3. Entrenar

#### Opción A: Con Docker (Recomendado)

```bash
cd trainer/training
docker build -t multilabel-trainer .

docker run --rm \
  -v $(pwd)/../dataset:/data/dataset \
  -v $(pwd)/../outputs:/out \
  multilabel-trainer \
  python -m src.cli train \
    --data_dir /data/dataset \
    --out_dir /out/run_001 \
    --config /app/configs/default.yaml
```

#### Opción B: Sin Docker

```bash
cd trainer/training
pip install -r requirements.txt

python -m src.cli train \
  --data_dir ../dataset \
  --out_dir ../outputs/run_001 \
  --config configs/default.yaml
```

### 4. Usar Modelo

Los archivos generados en `outputs/run_001/`:

```
✓ model_qat_int8.tflite        # Copiar a app móvil
✓ labels.txt                    # Copiar a app móvil
✓ inference_config.json         # Usar para configurar app
✓ threshold_recommendation.json # Umbrales optimizados
✓ metrics.json                  # Métricas finales
```

## 🔧 Solución de Problemas

### Error: "File not found"
- Verificar paths absolutos en dataset
- Verificar que todas las imágenes en .txt existan

### Error: "Expected X classes, found Y"
- Ajustar `num_classes` en config.yaml
- Verificar que todas las clases estén en el dataset

### Out of Memory
- Reducir `batch_size` en config.yaml
- Usar Docker con límite de memoria
- Entrenar con GPU

### Entrenamiento muy lento
- Usar GPU (agregar `--gpus all` en docker)
- Reducir `epochs_finetune`
- Reducir tamaño del dataset para pruebas

## 📊 Ejemplo con Dataset Pequeño

```bash
# 1. Crear dataset de prueba (100 imágenes, 3 clases)
mkdir -p test_dataset/images
# ... copiar imágenes ...

# 2. Crear splits
echo "img1.jpg,cat|dog" > test_dataset/train.txt
echo "img2.jpg,bird" >> test_dataset/train.txt
# ...

# 3. Ajustar config para test rápido
# En configs/default.yaml:
#   num_classes: 3
#   teacher.epochs_head: 2
#   teacher.epochs_finetune: 5
#   student.epochs_head: 2
#   student.epochs_finetune: 5
#   qat.epochs: 2

# 4. Entrenar
python -m src.cli train \
  --data_dir test_dataset \
  --out_dir test_output \
  --config configs/default.yaml
```

## ⏱️ Tiempos Esperados

Dataset de 1000 imágenes:

- **CPU**: 2-3 horas
- **GPU (V100)**: 30-40 minutos
- **GPU (T4)**: 1-1.5 horas

## 📞 Ayuda

Ver documentación completa:
- `trainer/training/README.md`
- `trainer/SISTEMA_COMPLETO.md`

---

Felipe Lara - felipe@lara.ac
