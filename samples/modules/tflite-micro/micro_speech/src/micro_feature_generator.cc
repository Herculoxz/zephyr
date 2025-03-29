#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>
#include <zephyr/sys/util.h>
#include <zephyr/init.h>
#include "micro_feature_generator.h"

#include <cmath>
#include <cstring>
#include "/home/abhinav/zephyrproject/zephyr/samples/modules/tflite-micro/testing/models/audio_preprocessor_int8_model_data.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "micro_model_settings.h"

LOG_MODULE_REGISTER(tfmicro_speech, LOG_LEVEL_DBG);

namespace {

// Memory for TFLM model execution (adjusted for Zephyr)
constexpr size_t kArenaSize = 16 * 1024;
K_HEAP_DEFINE(tflm_arena, kArenaSize);  // Use Zephyr heap instead of static buffer

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;

constexpr int kAudioSampleDurationCount =
    kFeatureDurationMs * kAudioSampleFrequency / 1000;
constexpr int kAudioSampleStrideCount =
    kFeatureStrideMs * kAudioSampleFrequency / 1000;

using AudioPreprocessorOpResolver = tflite::MicroMutableOpResolver<18>;

}  // namespace

TfLiteStatus RegisterOps(AudioPreprocessorOpResolver& op_resolver) {
  TF_LITE_ENSURE_STATUS(op_resolver.AddReshape());
  TF_LITE_ENSURE_STATUS(op_resolver.AddCast());
  TF_LITE_ENSURE_STATUS(op_resolver.AddStridedSlice());
  TF_LITE_ENSURE_STATUS(op_resolver.AddConcatenation());
  TF_LITE_ENSURE_STATUS(op_resolver.AddMul());
  TF_LITE_ENSURE_STATUS(op_resolver.AddAdd());
  TF_LITE_ENSURE_STATUS(op_resolver.AddDiv());
  TF_LITE_ENSURE_STATUS(op_resolver.AddMinimum());
  TF_LITE_ENSURE_STATUS(op_resolver.AddMaximum());
  TF_LITE_ENSURE_STATUS(op_resolver.AddWindow());
  TF_LITE_ENSURE_STATUS(op_resolver.AddFftAutoScale());
  TF_LITE_ENSURE_STATUS(op_resolver.AddRfft());
  TF_LITE_ENSURE_STATUS(op_resolver.AddEnergy());
  TF_LITE_ENSURE_STATUS(op_resolver.AddFilterBank());
  TF_LITE_ENSURE_STATUS(op_resolver.AddFilterBankSquareRoot());
  TF_LITE_ENSURE_STATUS(op_resolver.AddFilterBankSpectralSubtraction());
  TF_LITE_ENSURE_STATUS(op_resolver.AddPCAN());
  TF_LITE_ENSURE_STATUS(op_resolver.AddFilterBankLog());
  return kTfLiteOk;
}

TfLiteStatus InitializeMicroFeatures() {
  LOG_INF("Initializing Micro Features");

  model = tflite::GetModel(g_audio_preprocessor_int8_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    LOG_ERR("Model schema version mismatch: %d != %d", model->version(), TFLITE_SCHEMA_VERSION);
    return kTfLiteError;
  }

  static AudioPreprocessorOpResolver op_resolver;
  RegisterOps(op_resolver);

  uint8_t* arena_ptr = (uint8_t*)k_heap_alloc(&tflm_arena, kArenaSize, K_NO_WAIT);
  if (!arena_ptr) {
    LOG_ERR("Failed to allocate memory for TensorFlow Lite Micro arena");
    return kTfLiteError;
  }

  static tflite::MicroInterpreter static_interpreter(model, op_resolver, arena_ptr, kArenaSize);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    LOG_ERR("AllocateTensors failed for Feature provider model");
    return kTfLiteError;
  }

  LOG_INF("Micro Features Initialized");
  return kTfLiteOk;
}

TfLiteStatus GenerateSingleFeature(const int16_t* audio_data,
                                   const int audio_data_size,
                                   int8_t* feature_output) {
  TfLiteTensor* input = interpreter->input(0);
  TfLiteTensor* output = interpreter->output(0);
  std::copy_n(audio_data, audio_data_size, tflite::GetTensorData<int16_t>(input));

  if (interpreter->Invoke() != kTfLiteOk) {
    LOG_ERR("Feature generator model invocation failed");
    return kTfLiteError;
  }

  std::copy_n(tflite::GetTensorData<int8_t>(output), kFeatureSize, feature_output);

  return kTfLiteOk;
}

TfLiteStatus GenerateFeatures(const int16_t* audio_data,
                              const size_t audio_data_size,
                              Features* features_output) {
  size_t remaining_samples = audio_data_size;
  size_t feature_index = 0;
  
  while (remaining_samples >= kAudioSampleDurationCount &&
         feature_index < kFeatureCount) {
    TF_LITE_ENSURE_STATUS(
        GenerateSingleFeature(audio_data, kAudioSampleDurationCount,
                              (*features_output)[feature_index]));
    feature_index++;
    audio_data += kAudioSampleStrideCount;
    remaining_samples -= kAudioSampleStrideCount;
  }

  return kTfLiteOk;
}
