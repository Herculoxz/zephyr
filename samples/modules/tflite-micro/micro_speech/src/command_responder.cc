#include "command_responder.h"
#include "tensorflow/lite/micro/micro_log.h"

// The default implementation writes out the name of the recognized command
// to the error console. Real applications will want to take some custom
// action instead, and should implement their own versions of this function.
void RespondToCommand(int32_t current_time, const char* found_command,
                      float score, bool is_new_command) {
  if (is_new_command) {
    MicroPrintf("Heard %s (%.4f) @%dms", found_command, score, current_time);
  }
}