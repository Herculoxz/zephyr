#include "command_responder.h"
#include <tensorflow/lite/micro/micro_log.h>
#include "micro_model_settings.h"

// Respond to the inference results by printing the scores
void RespondToCommand(float silence_score, float unknown_score, float yes_score, float no_score) {
  // Print the scores for each label
  MicroPrintf("Scores:");
  MicroPrintf("  %s: %f", kLabels[0], static_cast<double>(silence_score));
  MicroPrintf("  %s: %f", kLabels[1], static_cast<double>(unknown_score));
  MicroPrintf("  %s: %f", kLabels[2], static_cast<double>(yes_score));
  MicroPrintf("  %s: %f", kLabels[3], static_cast<double>(no_score));

  // Optional: Determine the predicted label (highest score)
  float scores[kLabelCount] = {silence_score, unknown_score, yes_score, no_score};
  int predicted_label = 0;
  for (int i = 1; i < kLabelCount; i++) {
    if (scores[i] > scores[predicted_label]) {
      predicted_label = i;
    }
  }
  MicroPrintf("Predicted label: %s (score: %f)", kLabels[predicted_label], static_cast<double>(scores[predicted_label]));
}