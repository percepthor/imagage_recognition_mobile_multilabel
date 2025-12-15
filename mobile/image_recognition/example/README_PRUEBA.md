# Prueba de la Aplicación de Reconocimiento de Imágenes

## Modelo Incluido

✅ **Modelo cargado**: `model_qat_int8.tflite` (4.7 MB)
- **Input**: 240x240 RGB (uint8)
- **Output**: 7 clases (int8 cuantizado)
- **Clases detectadas**: cortadas, fraude, noesproducto, obscuras, puerta_abierta, puerta_cerrada, reflejos

## Instrucciones para Probar

### 1. Conectar Smartphone

**Android:**
```bash
# Habilitar depuración USB en el dispositivo Android
# Conectar por USB
# Verificar conexión:
flutter devices
```

**iOS:**
```bash
# Conectar iPhone/iPad por USB
# Confiar en la computadora desde el dispositivo
# Verificar conexión:
flutter devices
```

### 2. Ejecutar la Aplicación

**Desde el directorio `mobile/image_recognition/example/`:**

```bash
# Ejecutar en modo debug (recomendado para pruebas)
flutter run

# O ejecutar en modo release (más rápido)
flutter run --release
```

### 3. Uso de la Aplicación

1. **Cargar el Modelo**: La app cargará automáticamente el modelo al iniciar
2. **Seleccionar Imagen**:
   - **Cámara**: Tomar foto directamente
   - **Galería**: Seleccionar imagen existente
3. **Analizar**: Presionar el botón "Analizar Imagen"
4. **Ver Resultados**:
   - ✓ **Imagen OK**: No se detectaron problemas
   - ⚠️ **Problemas detectados**: Lista de problemas con porcentaje de confianza

### 4. Tiempos Esperados

- **Carga del modelo**: ~1-2 segundos (solo al iniciar)
- **Inferencia**: ~50-200ms dependiendo del dispositivo
- **Total por imagen**: ~100-300ms

### 5. Permisos Requeridos

La aplicación solicitará permisos automáticamente:
- **Cámara**: Para tomar fotos
- **Galería/Fotos**: Para seleccionar imágenes existentes

### 6. Solución de Problemas

**Error: "No device found"**
```bash
# Verificar que el dispositivo esté conectado
flutter devices

# Si no aparece:
# Android: Verificar que la depuración USB esté habilitada
# iOS: Verificar que el dispositivo confíe en la computadora
```

**Error: "Failed to initialize"**
- Verificar que los archivos en `assets/` existen:
  - `model_qat_int8.tflite`
  - `inference_config.json`
  - `labels.txt`

**Error de permisos**
- Android: Aceptar los permisos cuando la app los solicite
- iOS: Ir a Ajustes > Privacidad y otorgar permisos manualmente si es necesario

### 7. Características Implementadas

✅ **Preprocesamiento Letterbox**: Mantiene aspect ratio con padding negro
✅ **Inferencia INT8**: Modelo cuantizado para máximo rendimiento
✅ **Postprocesamiento**: Dequantización + sigmoid + thresholds optimizados
✅ **Multi-etiqueta**: Detecta múltiples problemas simultáneamente
✅ **UI Completa**: Interfaz Material Design con resultados visuales

### 8. Notas Técnicas

- **Preprocesamiento**: Letterbox automático a 240x240 con padding negro
- **Cuantización**: INT8 para input y output, procesamiento interno int8
- **Thresholds**: Optimizados por clase para maximizar F1 macro
- **Orden RGB**: Asegurado en preprocesamiento
- **Sin normalización**: El modelo espera uint8 [0-255] directamente

## Ejemplos de Uso

### Caso 1: Imagen sin problemas
```
Resultado: ✓ Imagen OK
Tiempo: ~150ms
```

### Caso 2: Imagen con reflejos
```
Resultado: ⚠️ Se detectaron 1 problema(s):
  #1 reflejos (87.3%)
Tiempo: ~142ms
```

### Caso 3: Imagen con múltiples problemas
```
Resultado: ⚠️ Se detectaron 3 problema(s):
  #1 puerta_abierta (94.2%)
  #2 reflejos (76.5%)
  #3 obscuras (68.1%)
Tiempo: ~158ms
```

## Comandos Rápidos

```bash
# Ejecutar en modo debug
cd mobile/image_recognition/example
flutter run

# Ver logs
flutter logs

# Hot reload (durante ejecución)
# Presiona 'r' en la terminal

# Hot restart (durante ejecución)
# Presiona 'R' en la terminal

# Salir
# Presiona 'q' en la terminal
```

## Estado del Proyecto

✅ Modelo entrenado y exportado
✅ Aplicación móvil implementada
✅ Preprocesamiento correcto
✅ Inferencia optimizada
✅ UI completa
✅ Listo para probar

**¡La aplicación está lista para usar! Conecta tu smartphone y ejecuta `flutter run`**
