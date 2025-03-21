/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <main_functions.cpp>
#include "tensorflow/lite/core/c/common.h"
#include "micro_model_settings.h"
#include "audio_preprocessor_int8_model_data.h"
#include "micro_speech_quantized_model_data.h"
#include "no_1000ms_audio_data.h"
#include "no_30ms_audio_data.h"
#include "noise_1000ms_audio_data.h"
#include "silence_1000ms_audio_data.h"
#include "yes_1000ms_audio_data.h"
#include "yes_30ms_audio_data.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/testing/micro_test.h"
onstexpr size_t kArenaSize = 28584;
alignas(16) uint8_t g_arena[kArenaSize];

using Features = int8_t[kFeatureCount][kFeatureSize];
Features g_features;

// Audio-related constants
constexpr int kAudioSampleDurationCount = kFeatureDurationMs * kAudioSampleFrequency / 1000;
constexpr int kAudioSampleStrideCount   = kFeatureStrideMs * kAudioSampleFrequency / 1000;

// Declare global interpreters
tflite::MicroInterpreter* audio_preproc_interpreter = nullptr;
tflite::MicroInterpreter* inference_interpreter     = nullptr;

// ------------------------ Function Prototypes -------------------------------
void setup();
void loop();

bool initializeAudioPreprocessorModel();
bool initializeInferenceModel();
bool runFeatureExtraction(const int16_t* audio_data, size_t audio_data_size);
bool runInference(const char* expected_label);

// ------------------------ Register All Ops ----------------------------------
using AllOpsResolver = tflite::MicroMutableOpResolver<22>;

TfLiteStatus RegisterAllOps(AllOpsResolver& op_resolver) {
  // MicroSpeech ops
  TF_LITE_ENSURE_STATUS(op_resolver.AddReshape());
  TF_LITE_ENSURE_STATUS(op_resolver.AddFullyConnected());
  TF_LITE_ENSURE_STATUS(op_resolver.AddDepthwiseConv2D());
  TF_LITE_ENSURE_STATUS(op_resolver.AddSoftmax());

  // Audio Preprocessing ops
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

// ------------------------ Model Initialization -------------------------------

// Setup function for audio preprocessor model
bool initializeAudioPreprocessorModel() {
  const tflite::Model* model = tflite::GetModel(g_audio_preprocessor_int8_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) return false;

  static AllOpsResolver resolver;
  if (RegisterAllOps(resolver) != kTfLiteOk) return false;

  static tflite::MicroInterpreter local_interpreter(model, resolver, g_arena, kArenaSize);
  if (local_interpreter.AllocateTensors() != kTfLiteOk) return false;

  audio_preproc_interpreter = &local_interpreter;
  MicroPrintf("AudioPreprocessor arena used: %u", audio_preproc_interpreter->arena_used_bytes());
  return true;
}

// Setup function for inference model
bool initializeInferenceModel() {
  const tflite::Model* model = tflite::GetModel(g_micro_speech_quantized_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) return false;

  static AllOpsResolver resolver;
  if (RegisterAllOps(resolver) != kTfLiteOk) return false;

  static tflite::MicroInterpreter local_interpreter(model, resolver, g_arena, kArenaSize);
  if (local_interpreter.AllocateTensors() != kTfLiteOk) return false;

  inference_interpreter = &local_interpreter;
  MicroPrintf("MicroSpeech arena used: %u", inference_interpreter->arena_used_bytes());
  return true;
}

// ------------------------ Feature Generation -------------------------------

// Extracts one feature vector from audio
TfLiteStatus GenerateSingleFeature(const int16_t* audio_data,
                                   const int audio_data_size,
                                   int8_t* feature_output) {
  TfLiteTensor* input  = audio_preproc_interpreter->input(0);
  TfLiteTensor* output = audio_preproc_interpreter->output(0);

  if (!input || !output ||
      audio_data_size != kAudioSampleDurationCount ||
      input->dims->data[input->dims->size - 1] != kAudioSampleDurationCount ||
      output->dims->data[output->dims->size - 1] != kFeatureSize)
    return kTfLiteError;

  std::copy_n(audio_data, audio_data_size, tflite::GetTensorData<int16_t>(input));
  if (audio_preproc_interpreter->Invoke() != kTfLiteOk) return kTfLiteError;
  std::copy_n(tflite::GetTensorData<int8_t>(output), kFeatureSize, feature_output);
  return kTfLiteOk;
}

// Full feature extraction process for a given audio
bool runFeatureExtraction(const int16_t* audio_data, size_t audio_data_size) {
  size_t remaining = audio_data_size;
  size_t index = 0;

  while (remaining >= kAudioSampleDurationCount && index < kFeatureCount) {
    if (GenerateSingleFeature(audio_data, kAudioSampleDurationCount, g_features[index]) != kTfLiteOk)
      return false;

    audio_data += kAudioSampleStrideCount;
    remaining  -= kAudioSampleStrideCount;
    index++;
  }
  return true;
}

// ------------------------ Model Inference ---------------------------------

bool runInference(const char* expected_label) {
  TfLiteTensor* input  = inference_interpreter->input(0);
  TfLiteTensor* output = inference_interpreter->output(0);

  if (!input || !output ||
      input->dims->data[input->dims->size - 1] != kFeatureElementCount ||
      output->dims->data[output->dims->size - 1] != kCategoryCount)
    return false;

  float output_scale = output->params.scale;
  int output_zero_point = output->params.zero_point;

  std::copy_n(&g_features[0][0], kFeatureElementCount, tflite::GetTensorData<int8_t>(input));
  if (inference_interpreter->Invoke() != kTfLiteOk) return false;

  float predictions[kCategoryCount];
  MicroPrintf("Predictions for <%s>", expected_label);

  for (int i = 0; i < kCategoryCount; i++) {
    predictions[i] = (tflite::GetTensorData<int8_t>(output)[i] - output_zero_point) * output_scale;
    MicroPrintf("  %.4f %s", static_cast<double>(predictions[i]), kCategoryLabels[i]);
  }

  int pred_index = std::distance(std::begin(predictions),
                                 std::max_element(std::begin(predictions), std::end(predictions)));
  return strcmp(expected_label, kCategoryLabels[pred_index]) == 0;
}

// ------------------------ Arduino-style Main --------------------------------

void setup() {
  // Initialize Serial / UART if needed here
  MicroPrintf("Starting MicroSpeech Inference...\n");

  if (!initializeAudioPreprocessorModel()) {
    MicroPrintf("Error initializing Audio Preprocessor Model");
    return;
  }

  if (!initializeInferenceModel()) {
    MicroPrintf("Error initializing Inference Model");
    return;
  }
}

void loop() {
  // Test audio sample repeatedly or continuously in a real device
  const char* expected = "yes";

  if (!runFeatureExtraction(g_yes_1000ms_audio_data, g_yes_1000ms_audio_data_size)) {
    MicroPrintf("Feature extraction failed!");
    return;
  }

  if (!runInference(expected)) {
    MicroPrintf(" Prediction mismatch or failed");
  } else {
    MicroPrintf(" Prediction matched expected label: %s", expected);
  }

  delay(5000);  // Run every few seconds (Arduino-style)
}
