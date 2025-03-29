#ifndef TENSORFLOW_LITE_MICRO_EXAMPLES_MICRO_SPEECH_RECOGNIZE_COMMANDS_H_
#define TENSORFLOW_LITE_MICRO_EXAMPLES_MICRO_SPEECH_RECOGNIZE_COMMANDS_H_

#include <cstdint>
#include <limits.h>  // Correct header for INT32_MIN
#include "tensorflow/lite/c/common.h"
#include "micro_model_settings.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"

class PreviousResultsQueue {
 public:
  PreviousResultsQueue(tflite::ErrorReporter* error_reporter)
      : error_reporter_(error_reporter), front_index_(0), size_(0) {}

  struct Result {
    Result() : time_(0), scores() {}
    Result(int32_t time, const int8_t* input_scores) : time_(time) {
      for (int i = 0; i < kCategoryCount; ++i) {
        scores[i] = input_scores[i];
      }
    }
    int32_t time_;
    int8_t scores[kCategoryCount];
  };

  int size() { return size_; }
  bool empty() { return size_ == 0; }
  Result& front() { return results_[front_index_]; }
  Result& back() {
    int back_index = front_index_ + (size_ - 1);
    if (back_index >= kMaxResults) {
      back_index -= kMaxResults;
    }
    return results_[back_index];
  }

  void push_back(const Result& entry) {
    if (size_ >= kMaxResults) {
      TF_LITE_REPORT_ERROR(
          error_reporter_,
          "Couldn't push_back latest result, too many already!");
      return;
    }
    size_ += 1;
    back() = entry;
  }

  void pop_front() {  // Changed return type to void
    if (size_ <= 0) {
      TF_LITE_REPORT_ERROR(error_reporter_,
                           "Couldn't pop_front result, none present!");
      return;
    }
    front_index_ += 1;
    if (front_index_ >= kMaxResults) {
      front_index_ = 0;
    }
    size_ -= 1;
  }

  Result& from_front(int offset) {
    if ((offset < 0) || (offset >= size_)) {
      TF_LITE_REPORT_ERROR(error_reporter_,
                           "Attempt to read beyond the end of the queue!");
      offset = size_ - 1;
    }
    int index = front_index_ + offset;
    if (index >= kMaxResults) {
      index -= kMaxResults;
    }
    return results_[index];
  }

 private:
  tflite::ErrorReporter* error_reporter_;
  static constexpr int kMaxResults = 50;
  Result results_[kMaxResults];
  int front_index_;
  int size_;
};

class RecognizeCommands {
 public:
  RecognizeCommands(tflite::ErrorReporter* error_reporter,
    int32_t average_window_duration_ms,
    float detection_threshold,  // Fixed type (was uint8_t)
    int32_t suppression_ms,
    int32_t minimum_count);

  TfLiteStatus ProcessLatestResults(const TfLiteTensor* latest_results,
                                    int32_t current_time_ms,
                                    const char** found_command, 
                                    uint8_t* score,
                                    bool* is_new_command);

 private:
  tflite::ErrorReporter* error_reporter_;
  int32_t average_window_duration_ms_;
  float detection_threshold_;
  int32_t suppression_ms_;
  int32_t minimum_count_;

  PreviousResultsQueue previous_results_;
  const char* previous_top_label_;
  int32_t previous_top_label_time_;
};

#endif  // TENSORFLOW_LITE_MICRO_EXAMPLES_MICRO_SPEECH_RECOGNIZE_COMMANDS_H_
