#include "main_functions.h"
#include <tensorflow/lite/micro/micro_log.h>

int main() {
  MicroPrintf("Starting Micro Speech application");
  setup();
  MicroPrintf("Setup completed, running loop once");
  loop();  // Run loop only once
  MicroPrintf("Application finished");
  return 0;
}