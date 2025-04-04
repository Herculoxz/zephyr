#ifndef MICRO_MODEL_SETTINGS_H_
#define MICRO_MODEL_SETTINGS_H_

// Number of labels the model can predict (silence, unknown, yes, no)
constexpr int kLabelCount = 4;

// Labels for the output classes
constexpr char kSilenceLabel[] = "silence";
constexpr char kUnknownLabel[] = "unknown";
constexpr char kYesLabel[] = "yes";
constexpr char kNoLabel[] = "no";

// Array of label names for easy access
constexpr const char* kLabels[kLabelCount] = {
    kSilenceLabel,
    kUnknownLabel,
    kYesLabel,
    kNoLabel
};

// Input tensor dimensions (adjust based on your model and yes_30ms_sample_data.cc)
// Assuming 49 time steps x 40 MFCC coefficients, flattened
constexpr int kInputSize = 49 * 40;  // 1960 elements

// Tensor arena size (already defined in main_functions.cpp, but can be centralized here)
constexpr int kTensorArenaSize = 100 * 1024;  // 10 KB, adjust as needed

#endif  // MICRO_MODEL_SETTINGS_H_