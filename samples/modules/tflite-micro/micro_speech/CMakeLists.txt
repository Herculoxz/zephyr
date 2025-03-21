# cmake_minimum_required(VERSION 3.20.0)

# find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
# project(tensorflow_micro_speech)

# # These samples use local static initialization. Since Zephyr doesn't support the
# # C++ ABI for thread-safe initialization of local statics and the constructors don't
# # appear to require thread safety, we turn it off in the C++ compiler.
# set(NO_THREADSAFE_STATICS $<TARGET_PROPERTY:compiler-cpp,no_threadsafe_statics>)
# zephyr_compile_options($<$<COMPILE_LANGUAGE:CXX>:${NO_THREADSAFE_STATICS}>)

# file(GLOB app_sources src/*)
# target_sources(app PRIVATE ${app_sources})
#---------------------------
# cmake_minimum_required(VERSION 3.20.0)

# find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
# project(tensorflow_micro_speech)

# # Disable thread-safe statics for C++
# set(NO_THREADSAFE_STATICS $<TARGET_PROPERTY:compiler-cpp,no_threadsafe_statics>)
# zephyr_compile_options($<$<COMPILE_LANGUAGE:CXX>:${NO_THREADSAFE_STATICS}>)

# target_sources(app PRIVATE src/main_functions.cpp)
# #---------------------------
cmake_minimum_required(VERSION 3.20.0)

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(tensorflow_micro_speech)

# These samples use local static initialization. Since Zephyr doesn't support the
# C++ ABI for thread-safe initialization of local statics and the constructors don't
# appear to require thread safety, we turn it off in the C++ compiler.
set(NO_THREADSAFE_STATICS $<TARGET_PROPERTY:compiler-cpp,no_threadsafe_statics>)
zephyr_compile_options($<$<COMPILE_LANGUAGE:CXX>:${NO_THREADSAFE_STATICS}>)

# Add your source files (only .c and .cpp files)
file(GLOB app_sources src/*.c src/*.cpp)
if(NOT app_sources)
  message(FATAL_ERROR "No source files found in the 'src/' directory.")
endif()

# Add tflite-micro signal kernels source files
target_sources(app PRIVATE
  ${ZEPHYR_BASE}/../optional/modules/lib/tflite-micro/signal/micro/kernels/framer.cc
  ${ZEPHYR_BASE}/../optional/modules/lib/tflite-micro/signal/micro/kernels/pcan.cc
  ${ZEPHYR_BASE}/../optional/modules/lib/tflite-micro/signal/micro/kernels/pcan_flexbuffers_generated_data.cc
  ${ZEPHYR_BASE}/../optional/modules/lib/tflite-micro/signal/micro/kernels/filter_bank.cc
  ${ZEPHYR_BASE}/../optional/modules/lib/tflite-micro/signal/micro/kernels/filter_bank_flexbuffers_generated_data.cc
)


message(STATUS "App sources: ${app_sources}")
message(STATUS "TFLM signal sources: ${tflm_signal_sources}")
