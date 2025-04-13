# TensorFlow Lite Micro Speech Sample (inference only)

## Overview

This sample TensorFlow application demonstrates speech recognition using TensorFlow Lite Micro. The model included with the sample is trained to recognize simple spoken commands, such as `yes` and `no` from audio input. It processes audio data in real-time, extracting features and performing inference to classify spoken words.

The sample provides a complete end-to-end workflow, including training a model, converting it for use with TensorFlow Lite Micro, and running inference on a microcontroller. It showcases how to handle audio input and integrate a lightweight speech recognition model in resource-constrained environments.


> **Note**: This README and sample are based on [the TensorFlow Micro Speech sample](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/examples/micro_speech).

## Building and Running

> The sample requires a pre processed audio file to run inference of the corresponding keyword. 

### impementing `no` keyword : 
1. Make sure you have the pre processed features of the audio file you want to feed into the model. i.e `no_micro_features_data.cc` and `no_micro_features_data.h`

2. Include the `g_no_micro_f2e59fea_nohash_1_data[]` in **main_functions.cc** in the loop function to calculate score of the prediction. 
```
  // Compute the size of the MFCC features
const int sample_data_size = g_no_micro_f2e59fea_nohash_1_width * g_no_micro_f2e59fea_nohash_1_height;
```
3. Re build the model using the following command :
```
west build -b qemu_xtensa
```
--- 
**To only run the `yes` keyword inference , you can directly build his project without any changes in the code base  as per your board config.**

> We can also use esp32_devkit_wroom_procpu or nxp board config (rimage and remoteproc is needed in this case)

### Adding the tflite-micro Module

Add the `tflite-micro` module to your West manifest and pull it:

```bash
west config manifest.project-filter -- +tflite-micro
west update