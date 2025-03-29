# cmake_minimum_required(VERSION 3.20.0)

# find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
# project(tensorflow_micro_speech)

# # Disable thread-safe statics for local variables in C++
# zephyr_compile_options(-fno-threadsafe-statics)

# # Include the necessary directories
# zephyr_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
# zephyr_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/signals)

# # Add all source files from `src`
# file(GLOB app_sources src/*.cc src/*.c)

# target_sources(app PRIVATE
#   ${ZEPHYR_BASE}../optional/modules/lib/tflite-micro/signal/micro/kernels/framer.cc
#   ${ZEPHYR_BASE}../optional/modules/lib/tflite-micro/signal/micro/kernels/pcan.cc
#   ${ZEPHYR_BASE}../optional/modules/lib/tflite-micro/signal/micro/kernels/pcan_flexbuffers_generated_data.cc
#   ${ZEPHYR_BASE}../optional/modules/lib/tflite-micro/signal/micro/kernels/filter_bank.cc
#   ${ZEPHYR_BASE}../optional/modules/lib/tflite-micro/signal/micro/kernels/filter_bank_flexbuffers_generated_data.cc
#   ${ZEPHYR_BASE}../optional/modules/lib/tflite-micro/signal/micro/kernels/energy.cc 
#   ${ZEPHYR_BASE}../optional/modules/lib/tflite-micro/signal/micro/kernels/filter_bank_square_root.cc 
#   ${ZEPHYR_BASE}../optional/modules/lib/tflite-micro/signal/micro/kernels/filter_bank_spectral_subtraction.cc 
#   ${ZEPHYR_BASE}../optional/modules/lib/tflite-micro/signal/micro/kernels/filter_bank_log.cc 
#   ${ZEPHYR_BASE}../optional/modules/lib/tflite-micro/signal/micro/kernels/filter_bank_square_root_common.cc
# )

# # Add all source files from `signals`
# file(GLOB SIGNAL_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/signals/*.cpp)

# # Add sources to the Zephyr application target
# target_sources(app PRIVATE ${app_sources} ${SIGNAL_SOURCES})

#---------------------------
# cmake_minimum_required(VERSION 3.20.0)

# find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
# project(tensorflow_micro_speech)

# # Disable thread-safe statics for C++
# set(NO_THREADSAFE_STATICS $<TARGET_PROPERTY:compiler-cpp,no_threadsafe_statics>)
# zephyr_compile_options($<$<COMPILE_LANGUAGE:CXX>:${NO_THREADSAFE_STATICS}>)

# target_sources(app PRIVATE src/main_functions.cpp)
# #---------------------------
# cmake_minimum_required(VERSION 3.20.0)

# find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
# project(tensorflow_micro_speech)

# # Ensure C and C++ are enabled
# enable_language(C CXX)

# # Disable thread-safe statics (Zephyr-specific)
# set(NO_THREADSAFE_STATICS $<TARGET_PROPERTY:compiler-cpp,no_threadsafe_statics>)
# zephyr_compile_options($<$<COMPILE_LANGUAGE:CXX>:${NO_THREADSAFE_STATICS}>)

# # Add source files from `src/` and `signals/` directories
# file(GLOB_RECURSE app_sources 
#   ${CMAKE_CURRENT_SOURCE_DIR}/src/*.c 
#   ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc 
#   ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp 
#   ${CMAKE_CURRENT_SOURCE_DIR}/signals/*.c 
#   ${CMAKE_CURRENT_SOURCE_DIR}/signals/*.cc 
#   ${CMAKE_CURRENT_SOURCE_DIR}/signals/*.cpp
# )

# # Debugging: Print found source files
# message(STATUS "Source files found: ${app_sources}")

# # If no sources are found, stop with an error
# if(NOT app_sources)
#   message(FATAL_ERROR "No source files found in the 'src/' or 'signals/' directory.")
# endif()

# # Add sources to Zephyr app
# target_sources(app PRIVATE ${app_sources})

# # Add TensorFlow Lite Micro signal processing kernels
# target_sources(app PRIVATE
#   ${ZEPHYR_BASE}/modules/lib/tflite-micro/signal/micro/kernels/framer.cc
#   ${ZEPHYR_BASE}/modules/lib/tflite-micro/signal/micro/kernels/pcan.cc
#   ${ZEPHYR_BASE}/modules/lib/tflite-micro/signal/micro/kernels/pcan_flexbuffers_generated_data.cc
#   ${ZEPHYR_BASE}/modules/lib/tflite-micro/signal/micro/kernels/filter_bank.cc
#   ${ZEPHYR_BASE}/modules/lib/tflite-micro/signal/micro/kernels/filter_bank_flexbuffers_generated_data.cc
#   ${ZEPHYR_BASE}/modules/lib/tflite-micro/signal/micro/kernels/energy.cc 
#   ${ZEPHYR_BASE}/modules/lib/tflite-micro/signal/micro/kernels/filter_bank_square_root.cc 
#   ${ZEPHYR_BASE}/modules/lib/tflite-micro/signal/micro/kernels/filter_bank_spectral_subtraction.cc 
#   ${ZEPHYR_BASE}/modules/lib/tflite-micro/signal/micro/kernels/filter_bank_log.cc 
#   ${ZEPHYR_BASE}/modules/lib/tflite-micro/signal/micro/kernels/filter_bank_square_root_common.cc
# )

# # Include directories for headers
# target_include_directories(app PRIVATE
#   ${CMAKE_CURRENT_SOURCE_DIR}/src
#   ${CMAKE_CURRENT_SOURCE_DIR}/signals
#   ${ZEPHYR_BASE}/modules/lib/tflite-micro
#   ${ZEPHYR_BASE}/include
# )

# # Ensure Zephyr drivers are included
# zephyr_library_include_directories(${ZEPHYR_BASE}/drivers)

# message(STATUS "App sources: ${app_sources}")
cmake_minimum_required(VERSION 3.20.0)

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(tensorflow_hello_world)

# Include necessary directories
include_directories(${PROJECT_SOURCE_DIR}/testing/microfrontend/lib)

# Add main source files
set(SRC_FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main_functions.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/audio_provider.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/command_responder.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/feature_provider.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/micro_feature_generator.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/model.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/recognize_commands.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/micro_feature_generator.cc 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/feature_provider.cc
)

target_sources(app PRIVATE ${SRC_FILES})

# ✅ Fixing paths to TensorFlow Lite Micro
set(TFLM_DIR ${ZEPHYR_BASE}/../optional/modules/lib/tflite-micro/signal)

target_sources(app PRIVATE
  ${TFLM_DIR}/micro/kernels/delay.cc
  ${TFLM_DIR}/micro/kernels/delay_flexbuffers_generated_data.cc
  ${TFLM_DIR}/micro/kernels/energy.cc
  ${TFLM_DIR}/micro/kernels/energy_flexbuffers_generated_data.cc
  ${TFLM_DIR}/src/fft_auto_scale.cc
  ${TFLM_DIR}/micro/kernels/fft_flexbuffers_generated_data.cc
  ${TFLM_DIR}/micro/kernels/filter_bank.cc
  ${TFLM_DIR}/micro/kernels/filter_bank_flexbuffers_generated_data.cc
  ${TFLM_DIR}/micro/kernels/filter_bank_spectral_subtraction.cc
  ${TFLM_DIR}/micro/kernels/filter_bank_spectral_subtraction_flexbuffers_generated_data.cc
  ${TFLM_DIR}/micro/kernels/filter_bank_square_root.cc
 # ${TFLM_DIR}/micro/kernels/filter_bank_square_root_flexbuffers_generated_data.cc
  ${TFLM_DIR}/micro/kernels/filter_bank_log.cc
  ${TFLM_DIR}/micro/kernels/filter_bank_log_flexbuffers_generated_data.cc
  ${TFLM_DIR}/micro/kernels/pcan.cc
  ${TFLM_DIR}/micro/kernels/pcan_flexbuffers_generated_data.cc
 # ${TFLM_DIR}/audio_utils.cc
)
