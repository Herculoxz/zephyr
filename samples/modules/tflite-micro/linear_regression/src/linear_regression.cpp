/* 
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

#include <zephyr/kernel.h>
#include <zephyr/device.h>
#include <zephyr/drivers/gpio.h>
#include <zephyr/sys/printk.h>
#include <zephyr/random/random.h>

// TFLite Micro headers
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

#include "linear_regression_model_data.h"

// Constants
#define TENSOR_ARENA_SIZE 2048
static uint8_t tensor_arena[TENSOR_ARENA_SIZE];

static tflite::ErrorReporter* error_reporter = nullptr;
static const tflite::Model* model = nullptr;
static tflite::MicroInterpreter* interpreter = nullptr;
static TfLiteTensor* input = nullptr;
static TfLiteTensor* output = nullptr;

float randFloat(float min, float max) {
    return ((max - min) * ((float)sys_rand32_get() / UINT32_MAX)) + min;
}

void init_model() {
    static tflite::MicroErrorReporter micro_error_reporter;
    error_reporter = &micro_error_reporter;

    model = tflite::GetModel(g_linear_regresion_model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        error_reporter->Report("Model version mismatch.");
        return;
    }

    // REPLACED AllOpsResolver with MicroMutableOpResolver
    static tflite::MicroMutableOpResolver<2> resolver;
    resolver.AddFullyConnected();
    resolver.AddAdd();  // Add other ops if your model uses more (e.g., Mul, Relu, etc.)

    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, TENSOR_ARENA_SIZE,
        /* resource_variables */ nullptr,
        /* profiler */ nullptr
    );
    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        error_reporter->Report("AllocateTensors failed.");
        return;
    }

    input = interpreter->input(0);
    output = interpreter->output(0);
}

void run_inference() {
    for (int i = 0; i < 100; ++i) {
        float x_val = randFloat(0.0f, 1.0f);
        input->data.f[0] = x_val;

        TfLiteStatus status = interpreter->Invoke();
        if (status != kTfLiteOk) {
            error_reporter->Report("Inference failed.");
            return;
        }

        float y_val = output->data.f[0];
        printk("Input: %.3f -> Output: %.3f\n", (double)x_val, (double)y_val);
    }
}

int main(void) {
    printk("Zephyr TFLite Linear Regression Demo\n");

    init_model();

    while (true) {
        run_inference();
        printk("Waiting 5s before next round of inference...\n");
        k_sleep(K_SECONDS(5));
    }
}
