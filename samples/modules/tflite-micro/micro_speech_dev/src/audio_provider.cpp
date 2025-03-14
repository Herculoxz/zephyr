/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include <zephyr.h>
#include <device.h>
#include <drivers/pdm.h>
#include <sys/__assert.h>
#include <sys/util.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#include "audio_provider.h"
#include "micro_features_micro_model_settings.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "test_over_serial/test_over_serial.h"

using namespace test_over_serial;

namespace {
bool g_is_audio_initialized = false;
// An internal buffer able to fit 16x our sample size
constexpr int kAudioCaptureBufferSize = DEFAULT_PDM_BUFFER_SIZE * 16;
int16_t g_audio_capture_buffer[kAudioCaptureBufferSize];
// A buffer that holds our output
int16_t g_audio_output_buffer[kMaxAudioSampleSize];
// Mark as atomic so we can check in a while loop to see if
// any samples have arrived yet.
atomic_t g_latest_audio_timestamp;
// test_over_serial sample index
uint32_t g_test_sample_index;
// test_over_serial silence insertion flag
bool g_test_insert_silence = true;

// PDM microphone device
const struct device *pdm_mic = DEVICE_DT_GET(DT_NODELABEL(pdm0));
}  // namespace

void CaptureSamples() {
  // This is how many bytes of new data we have each time this is called
  const int number_of_samples = DEFAULT_PDM_BUFFER_SIZE / 2;
  // Calculate what timestamp the last audio sample represents
  const int32_t time_in_ms =
      atomic_get(&g_latest_audio_timestamp) +
      (number_of_samples / (kAudioSampleFrequency / 1000));
  // Determine the index, in the history of all samples, of the last sample
  const int32_t start_sample_offset =
      atomic_get(&g_latest_audio_timestamp) * (kAudioSampleFrequency / 1000);
  // Determine the index of this sample in our ring buffer
  const int capture_index = start_sample_offset % kAudioCaptureBufferSize;
  // Read the data to the correct place in our buffer
  int num_read = pdm_read(pdm_mic, g_audio_capture_buffer + capture_index,
                          DEFAULT_PDM_BUFFER_SIZE);
  if (num_read != DEFAULT_PDM_BUFFER_SIZE) {
    MicroPrintf("### short read (%d/%d) @%dms", num_read,
                DEFAULT_PDM_BUFFER_SIZE, time_in_ms);
    while (true) {
      // NORETURN
    }
  }
  // This is how we let the outside world know that new audio data has arrived.
  atomic_set(&g_latest_audio_timestamp, time_in_ms);
}

TfLiteStatus InitAudioRecording() {
  if (!g_is_audio_initialized) {
    // Initialize the PDM microphone
    if (!device_is_ready(pdm_mic)) {
      MicroPrintf("PDM microphone device not ready");
      return kTfLiteError;
    }

    // Configure the PDM microphone
    struct pdm_config cfg = {
        .clock_freq = kAudioSampleFrequency,
        .mode = PDM_MODE_MONO,
        .gain = 13,  // gain: -20db (min) + 6.5db (13) + 3.2db (builtin) = -10.3db
    };
    pdm_configure(pdm_mic, &cfg);

    // Start the PDM microphone
    pdm_start(pdm_mic);

    // Block until we have our first audio sample
    while (!atomic_get(&g_latest_audio_timestamp)) {
    }

    g_is_audio_initialized = true;
  }

  return kTfLiteOk;
}

TfLiteStatus GetAudioSamples(int start_ms, int duration_ms,
                             int* audio_samples_size, int16_t** audio_samples) {
  // Determine the index, in the history of all samples, of the first
  // sample we want
  const int start_offset = start_ms * (kAudioSampleFrequency / 1000);
  // Determine how many samples we want in total
  const int duration_sample_count =
      duration_ms * (kAudioSampleFrequency / 1000);
  for (int i = 0; i < duration_sample_count; ++i) {
    // For each sample, transform its index in the history of all samples into
    // its index in g_audio_capture_buffer
    const int capture_index = (start_offset + i) % kAudioCaptureBufferSize;
    // Write the sample to the output buffer
    g_audio_output_buffer[i] = g_audio_capture_buffer[capture_index];
  }

  // Set pointers to provide access to the audio
  *audio_samples_size = duration_sample_count;
  *audio_samples = g_audio_output_buffer;

  return kTfLiteOk;
}

namespace {

void InsertSilence(const size_t len, int16_t value) {
  for (size_t i = 0; i < len; i++) {
    const size_t index = (g_test_sample_index + i) % kAudioCaptureBufferSize;
    g_audio_capture_buffer[index] = value;
  }
  g_test_sample_index += len;
}

int32_t ProcessTestInput(TestOverSerial& test) {
  constexpr size_t samples_16ms = ((kAudioSampleFrequency / 1000) * 16);

  InputHandler handler = [](const InputBuffer* const input) {
    if (0 == input->offset) {
      // don't insert silence
      g_test_insert_silence = false;
    }

    for (size_t i = 0; i < input->length; i++) {
      const size_t index = (g_test_sample_index + i) % kAudioCaptureBufferSize;
      g_audio_capture_buffer[index] = input->data.int16[i];
    }
    g_test_sample_index += input->length;

    if (input->total == (input->offset + input->length)) {
      // allow silence insertion again
      g_test_insert_silence = true;
    }
    return true;
  };

  test.ProcessInput(&handler);

  if (g_test_insert_silence) {
    // add 16ms of silence just like the PDM interface
    InsertSilence(samples_16ms, 0);
  }

  // Round the timestamp to a multiple of 64ms,
  // This emulates the PDM interface during inference processing.
  atomic_set(&g_latest_audio_timestamp,
             (g_test_sample_index / (samples_16ms * 4)) * 64);
  return atomic_get(&g_latest_audio_timestamp);
}

}  // namespace

int32_t LatestAudioTimestamp() {
  TestOverSerial& test = TestOverSerial::Instance(kAUDIO_PCM_16KHZ_MONO_S16);
  if (!test.IsTestMode()) {
    // check serial port for test mode command
    test.ProcessInput(nullptr);
  }
  if (test.IsTestMode()) {
    if (g_is_audio_initialized) {
      // stop capture from hardware
      pdm_stop(pdm_mic);
      g_is_audio_initialized = false;
      g_test_sample_index =
          atomic_get(&g_latest_audio_timestamp) * (kAudioSampleFrequency / 1000);
    }
    return ProcessTestInput(test);
  } else {
    // CaptureSamples() updated the timestamp
    return atomic_get(&g_latest_audio_timestamp);
  }
  // NOTREACHED
}