#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Export visible + "used" para evitar stripping en iOS/LTO
// Flutter recomienda marcar símbolos usados por Dart.
#if defined(_WIN32)
  #define IR_API __declspec(dllexport)
#else
  #define IR_API __attribute__((visibility("default"))) __attribute__((used))
#endif

typedef struct {
  const char* label_utf8;   // puntero a string UTF-8 null-terminated
  float confidence;         // 0..1
} ir_label_prediction_t;

typedef struct {
  int32_t error_code;       // 0 = OK
  const char* error_message_utf8; // nullable; UTF-8
  int32_t num_predictions;  // 0..N
  ir_label_prediction_t* predictions; // array length num_predictions

  // Opcional: tiempos (microsegundos) y metadata
  int64_t preprocess_time_us;
  int64_t inference_time_us;
  int64_t postprocess_time_us;
  const char* model_version_utf8; // nullable
} ir_result_t;

// Inicializa motor (carga modelo y config)
IR_API int32_t image_rec_init(const char* model_path_utf8,
                              const char* config_path_utf8);

// Análisis desde bytes comprimidos (JPG/PNG/etc). Recomendado por rendimiento/memoria.
IR_API int32_t image_rec_analyze_image_bytes(const uint8_t* bytes,
                                             int32_t length,
                                             ir_result_t* out_result);

// Libera SOLO memoria interna apuntada por out_result (predictions, error_message, etc).
// NO debe free() el puntero out_result en sí (lo maneja el caller).
IR_API void image_rec_free_result(ir_result_t* result);

// Cierra motor, libera caches/interpreters, etc.
IR_API void image_rec_shutdown(void);

// Opcionales (si existen, el wrapper puede llamarlas)
IR_API int32_t image_rec_set_num_threads(int32_t n);
IR_API int32_t image_rec_warmup(void);

#ifdef __cplusplus
}
#endif
