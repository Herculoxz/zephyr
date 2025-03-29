#include "recognize_commands.h"
#include <limits.h>  // Fixed incorrect header

RecognizeCommands::RecognizeCommands(tflite::ErrorReporter* error_reporter,
  int32_t average_window_duration_ms,
  float detection_threshold,  // Fixed to match header
  int32_t suppression_ms,
  int32_t minimum_count)
: error_reporter_(error_reporter),
  average_window_duration_ms_(average_window_duration_ms),
  detection_threshold_(detection_threshold),
  suppression_ms_(suppression_ms),
  minimum_count_(minimum_count),
  previous_results_(error_reporter) {  // Explicitly initialize queue
    previous_top_label_ = "silence";
    previous_top_label_time_ = INT32_MIN;  // Corrected for int32_t
}

TfLiteStatus RecognizeCommands::ProcessLatestResults(
    const TfLiteTensor* latest_results, int32_t current_time_ms,
    const char** found_command, uint8_t* score, bool* is_new_command) {

  if ((latest_results->dims->size != 2) ||
      (latest_results->dims->data[0] != 1) ||
      (latest_results->dims->data[1] != kCategoryCount)) {
    TF_LITE_REPORT_ERROR(
        error_reporter_,
        "The results for recognition should contain %d elements, but there are "
        "%d in an %d-dimensional shape",
        kCategoryCount, latest_results->dims->data[1],
        latest_results->dims->size);
    return kTfLiteError;
  }

  if (latest_results->type != kTfLiteInt8) {
    TF_LITE_REPORT_ERROR(
        error_reporter_,
        "The results for recognition should be int8_t elements, but are %d",
        latest_results->type);
    return kTfLiteError;
  }

  if ((!previous_results_.empty()) &&
      (current_time_ms < previous_results_.front().time_)) {
    TF_LITE_REPORT_ERROR(
        error_reporter_,
        "Results must be fed in increasing time order, but received a "
        "timestamp of %d that was earlier than the previous one of %d",
        current_time_ms, previous_results_.front().time_);
    return kTfLiteError;
  }

  previous_results_.push_back({current_time_ms, latest_results->data.int8});

  int64_t time_limit = current_time_ms - average_window_duration_ms_;
  while ((!previous_results_.empty()) &&
         previous_results_.front().time_ < time_limit) {
    previous_results_.pop_front();
  }

  if ((previous_results_.size() < minimum_count_) ||
      ((current_time_ms - previous_results_.front().time_) < (average_window_duration_ms_ / 4))) {
    *found_command = previous_top_label_;
    *score = 0;
    *is_new_command = false;
    return kTfLiteOk;
  }

  int32_t average_scores[kCategoryCount] = {0};
  for (int offset = 0; offset < previous_results_.size(); ++offset) {
    const auto& previous_result = previous_results_.from_front(offset);
    for (int i = 0; i < kCategoryCount; ++i) {
      average_scores[i] += previous_result.scores[i] + 128;
    }
  }
  for (int i = 0; i < kCategoryCount; ++i) {
    average_scores[i] /= previous_results_.size();
  }

  int current_top_index = 0;
  int32_t current_top_score = 0;
  for (int i = 0; i < kCategoryCount; ++i) {
    if (average_scores[i] > current_top_score) {
      current_top_score = average_scores[i];
      current_top_index = i;
    }
  }
  *found_command = kCategoryLabels[current_top_index];
  *score = current_top_score;
  *is_new_command = (current_top_score > detection_threshold_);

  return kTfLiteOk;
}
