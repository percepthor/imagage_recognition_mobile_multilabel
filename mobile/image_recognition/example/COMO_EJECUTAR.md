# Cómo Ejecutar la App de Reconocimiento de Imágenes

## Configuración del Toolchain (REPRODUCIBLE)

Esta app requiere un toolchain específico para compilar correctamente con Android SDK 35.

### Toolchain Configurado

- **Android Gradle Plugin**: 8.7.3
- **Gradle**: 8.9
- **JDK**: 17 (OpenJDK)
- **compileSdk**: 35
- **targetSdk**: 35 (flutter.targetSdkVersion)
- **minSdk**: 21
- **Kotlin**: 1.9.22
- **Java source/target**: 17

### Ubicación de las Configuraciones

1. **AGP y Kotlin** → `android/settings.gradle`
   ```gradle
   id "com.android.application" version "8.7.3" apply false
   id "org.jetbrains.kotlin.android" version "1.9.22" apply false
   ```

2. **Gradle** → `android/gradle/wrapper/gradle-wrapper.properties`
   ```properties
   distributionUrl=https\://services.gradle.org/distributions/gradle-8.9-all.zip
   ```

3. **JDK** → `android/gradle.properties`
   ```properties
   org.gradle.java.home=/usr/lib/jvm/java-17-openjdk-amd64
   ```

4. **compileSdk y Java** → `android/app/build.gradle`
   ```gradle
   android {
       compileSdk = 35

       compileOptions {
           sourceCompatibility = JavaVersion.VERSION_17
           targetCompatibility = JavaVersion.VERSION_17
       }

       kotlinOptions {
           jvmTarget = "17"
       }
   }
   ```

## Pasos para Ejecutar

### 1. Conectar dispositivo Android

```bash
# Habilitar depuración USB en el dispositivo
# Conectar por cable USB
flutter devices
```

Debe aparecer tu dispositivo:
```
SM S918B (mobile) • R5CW71VR21F • android-arm64 • Android 16 (API 36)
```

### 2. Limpiar y compilar

**IMPORTANTE**: Si cambias de toolchain o actualizas dependencias, SIEMPRE ejecuta una limpieza profunda:

```bash
cd mobile/image_recognition/example

# Limpieza completa (obligatorio tras cambios de toolchain)
flutter clean
rm -rf android/.gradle
rm -rf android/build
cd android && ./gradlew clean --no-daemon && cd ..
```

### 3. Ejecutar en modo debug

```bash
flutter run -d R5CW71VR21F
```

O en modo release (más rápido):

```bash
flutter run -d R5CW71VR21F --release
```

### 4. Verificar en los logs

Deberías ver:
```
✓ Built build/app/outputs/flutter-apk/app-debug.apk
✓ Installing APK...
✓ Initialized TensorFlow Lite runtime
✓ ImageClassifier inicializado correctamente
✓ Modelo: model_qat_int8.tflite
✓ Input shape: [1, 240, 240, 3]
✓ Output shape: [1, 7]
✓ Labels: 7
```

## Funcionalidad de la App

### Modelo Incluido

- **Archivo**: `assets/model_qat_int8.tflite` (4.7 MB)
- **Arquitectura**: EfficientNet-Lite4 cuantizado INT8
- **Input**: 240x240 RGB (uint8)
- **Output**: 7 clases (int8)
- **Preprocesamiento**: Letterbox con padding negro
- **Thresholds**: Optimizados por clase para maximizar F1 macro

### Clases Detectadas

1. cortadas
2. fraude
3. noesproducto
4. obscuras
5. puerta_abierta
6. puerta_cerrada
7. reflejos

### Uso de la App

1. **Tomar foto**: Presiona el botón "Cámara"
2. **Seleccionar imagen**: Presiona el botón "Galería"
3. **Analizar**: Presiona "Analizar Imagen"
4. **Ver resultados**:
   - ✓ **"Imagen OK"**: No se detectaron problemas
   - ⚠️ **Lista de problemas**: Con porcentaje de confianza

### Permisos Requeridos

La app solicita automáticamente:
- **Cámara**: Para tomar fotos
- **Galería**: Para seleccionar imágenes

Los permisos están configurados en:
- **Android**: `android/app/src/main/AndroidManifest.xml`
- **iOS**: `ios/Runner/Info.plist`

## Troubleshooting

### Error: "jlink", "androidJdkImage", "core-for-system-modules.jar"

**Causa**: Configuración incorrecta de JDK/AGP/Gradle para compilar con SDK 35.

**Solución**:
1. Verificar que estés usando **JDK 17** (no 11, no 21)
2. Verificar **AGP 8.7.3+** y **Gradle 8.9+**
3. Limpiar todas las cachés:
   ```bash
   flutter clean
   rm -rf ~/.gradle/caches
   rm -rf android/.gradle
   ```

### Error: "Flutter plugin android_lifecycle requires SDK 35"

**Causa**: `image_picker` requiere SDK 35, pero tu `compileSdk` está en 34 o menos.

**Solución**: Actualizar a `compileSdk = 35` en `android/app/build.gradle`

### Error: "Java version mismatch"

**Causa**: Gradle está usando un JDK diferente al configurado.

**Solución**: Forzar JDK 17 en `android/gradle.properties`:
```properties
org.gradle.java.home=/usr/lib/jvm/java-17-openjdk-amd64
```

### La app compila pero falla al tomar foto

**Causas posibles**:
1. Permisos no otorgados
2. FileProvider no configurado

**Solución**:
1. Verificar permisos en `AndroidManifest.xml`:
   ```xml
   <uses-permission android:name="android.permission.CAMERA"/>
   <uses-permission android:name="android.permission.READ_MEDIA_IMAGES"/>
   ```
2. Aceptar permisos cuando la app los solicite

### La inferencia es muy lenta

**Causa**: Modo debug tiene overhead significativo.

**Solución**: Ejecutar en modo release:
```bash
flutter run --release
```

Tiempos esperados:
- **Debug**: ~200-400ms
- **Release**: ~50-150ms (dependiendo del dispositivo)

## Verificación del Modelo

Para verificar que el modelo cargó correctamente, busca en los logs:

```
I/flutter: ImageClassifier inicializado correctamente
I/flutter:   Modelo: model_qat_int8.tflite
I/flutter:   Input shape: [1, 240, 240, 3]
I/flutter:   Output shape: [1, 7]
I/flutter:   Labels: 7
```

## Arquitectura del Código

### Archivos Principales

- **`lib/main.dart`**: UI principal con botones de cámara/galería
- **`lib/image_classifier.dart`**: Lógica de clasificación (letterbox + TFLite + postprocesamiento)
- **`assets/model_qat_int8.tflite`**: Modelo cuantizado INT8
- **`assets/inference_config.json`**: Configuración de inferencia
- **`assets/labels.txt`**: Nombres de las clases

### Pipeline de Inferencia

1. **Cargar imagen** (desde cámara o galería)
2. **Letterbox** a 240x240 con padding negro (mantiene aspect ratio)
3. **Convertir a uint8 RGB** (sin normalización)
4. **Ejecutar inferencia** con TFLite
5. **Dequantizar output** de int8 a float32
6. **Aplicar sigmoid** para obtener probabilidades
7. **Aplicar thresholds** optimizados por clase
8. **Mostrar resultados** ordenados por confianza

## Desarrollo Reproducible

Para asegurar builds reproducibles:

1. **Siempre** usar las versiones exactas del toolchain especificadas arriba
2. **Siempre** limpiar cachés tras cambios de toolchain
3. **Nunca** modificar `compileSdk` sin actualizar AGP/Gradle
4. **Verificar** que `org.gradle.java.home` apunta a JDK 17

## Recursos

- [Flutter Docs](https://docs.flutter.dev/)
- [TensorFlow Lite for Flutter](https://pub.dev/packages/tflite_flutter)
- [Android Gradle Plugin Compatibility](https://developer.android.com/build/releases/gradle-plugin)
- [Gradle-JDK Compatibility](https://docs.gradle.org/current/userguide/compatibility.html)
