#include "main_functions.h"

#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>
#include "micro_model_settings.h"    // Model-specific constants
#include "model.h"                  // Pre-trained Micro Speech model (g_model)
#include "command_responder.h"      // For RespondToCommand
#include "yes_micro_features_data.h" // Preprocessed "yes" MFCC features
#include <tensorflow/lite/micro/micro_log.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/system_setup.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <zephyr/kernel.h>          // For k_msleep()

/* Globals, used for compatibility with Arduino-style sketches. */
namespace {
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;
  int inference_count = 0;
  bool setup_successful = false;  // Flag to track setup success

  // Tensor arena size (using the constant from micro_model_settings.h)
  uint8_t tensor_arena[kTensorArenaSize];
}  // namespace

/* Setup function - runs once at startup */
void setup(void) {
  MicroPrintf("Entering setup()");

  // Initialize hardware (if needed, platform-specific)
  tflite::InitializeTarget();
  MicroPrintf("Target initialized");

  // Log the schema version
  MicroPrintf("Zephyr TFLITE_SCHEMA_VERSION: %d", TFLITE_SCHEMA_VERSION);

  // Log kInputSize
  MicroPrintf("kInputSize: %d", kInputSize);

  // Load the pre-trained Micro Speech model
  model = tflite::GetModel(g_model);
  if (model == nullptr) {
    MicroPrintf("Failed to load model: g_model is nullptr");
    return;
  }
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf("Model schema version %d does not match supported version %d.",
                model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }
  MicroPrintf("Model loaded, version: %d, size: %d bytes", model->version(), g_model_len);

  // Set up the operation resolver with operations used by Micro Speech
  static tflite::MicroMutableOpResolver<4> resolver;  // Increased to 10

  resolver.AddDepthwiseConv2D();
  MicroPrintf("Added DepthwiseConv2D operation");
  resolver.AddFullyConnected();
  MicroPrintf("Added FullyConnected operation");
  resolver.AddSoftmax();
  MicroPrintf("Added Softmax operation");
  resolver.AddReshape();
  MicroPrintf("Added Reshape operation");
 
 
  // Build the interpreter
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;
  MicroPrintf("Interpreter built");

  // Log the number of inputs and outputs
  MicroPrintf("Model has %d inputs, %d outputs", interpreter->inputs_size(), interpreter->outputs_size());

  // Allocate memory for the model's tensors
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    MicroPrintf("AllocateTensors() failed with status: %d", allocate_status);
    return;
  }
  MicroPrintf("Tensors allocated successfully");

  // Get pointers to input and output tensors
  input = interpreter->input(0);
  if (input == nullptr) {
    MicroPrintf("Failed to get input tensor");
    return;
  }
  output = interpreter->output(0);
  if (output == nullptr) {
    MicroPrintf("Failed to get output tensor");
    return;
  }
  MicroPrintf("Input/output tensors obtained. Input size: %d bytes, Output size: %d bytes",
              input->bytes, output->bytes);

  // Explicitly check for input->bytes == 0
  if (input->bytes == 0) {
    MicroPrintf("Error: Input tensor size is 0, cannot proceed");
    return;
  }

  // Log input tensor details
  MicroPrintf("Input tensor type: %d, dimensions: %d", input->type, input->dims->size);
  for (int i = 0; i < input->dims->size; i++) {
    MicroPrintf("Input dimension %d: %d", i, input->dims->data[i]);
  }

  // Verify input tensor size matches expected size
  if (input->bytes != kInputSize) {
    MicroPrintf("Input tensor size (%d) does not match expected size (%d)",
                input->bytes, kInputSize);
    return;
  }
  MicroPrintf("Input size verified");

  // Initialize inference counter
  inference_count = 0;

  // Mark setup as successful
  setup_successful = true;
  MicroPrintf("Micro Speech setup complete");
}

/* Loop function - runs once */
void loop(void) {
  if (!setup_successful) {
    MicroPrintf("Setup failed, skipping loop");
    return;
  }

  // Compute the size of the MFCC features
  const int sample_data_size = g_yes_micro_f2e59fea_nohash_1_width * g_yes_micro_f2e59fea_nohash_1_height;

  // Load the preprocessed "yes" MFCC features into the input tensor
  if (sample_data_size != input->bytes) {
    MicroPrintf("Sample data size (%d) does not match input tensor size (%d)",
                sample_data_size, input->bytes);
    return;
  }

  // Copy the sample data into the input tensor
  for (int i = 0; i < input->bytes; i++) {
    input->data.int8[i] = g_yes_micro_f2e59fea_nohash_1_data[i];
  }

  // Run inference
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    MicroPrintf("Inference failed with status: %d", invoke_status);
    return;
  }

  // Get the output scores (e.g., for "silence," "unknown," "yes," "no")
  int8_t* output_data = output->data.int8;
  float silence_score = (output_data[0] - output->params.zero_point) * output->params.scale;
  float unknown_score = (output_data[1] - output->params.zero_point) * output->params.scale;
  float yes_score = (output_data[2] - output->params.zero_point) * output->params.scale;
  float no_score = (output_data[3] - output->params.zero_point) * output->params.scale;

  // Pass results to the command responder
  RespondToCommand(silence_score, unknown_score, yes_score, no_score);

  // Increment inference counter
  inference_count++;

  MicroPrintf("Inference %d complete", inference_count);
}