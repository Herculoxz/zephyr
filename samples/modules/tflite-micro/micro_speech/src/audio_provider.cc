// #include <zephyr/kernel.h>
// #include <zephyr/device.h>
// #include <zephyr/drivers/i2s.h>
// #include <zephyr/sys/ring_buffer.h>
// #include <zephyr/logging/log.h>
// #include </home/abhinav/zephyrproject/zephyr/samples/modules/tflite-micro/testing/src/audio_provider.h>

// LOG_MODULE_REGISTER(audio_provider, LOG_LEVEL_INF);

// #define I2S_DEV DT_LABEL(DT_ALIAS(i2s0))  // Use the first I2S alias

// #define AUDIO_BUFFER_SIZE 40000
// #define I2S_BYTES_TO_READ 3200

// static struct ring_buf audio_ring_buffer;
// K_THREAD_STACK_DEFINE(audio_thread_stack, 4096);
// struct k_thread audio_thread_data;

// volatile int32_t g_latest_audio_timestamp = 0;
// int16_t g_audio_output_buffer[AUDIO_BUFFER_SIZE];
// bool g_is_audio_initialized = false;

// static void capture_audio_samples(void *arg, void *b, void *c) {
//     const struct device *i2s_dev = device_get_binding(I2S_DEV);
//     if (!i2s_dev) {
//         LOG_ERR("I2S device not found!");
//         return;
//     }

//     static uint8_t i2s_read_buffer[I2S_BYTES_TO_READ];

//     while (1) {
//         size_t bytes_read = 0;
//         int ret = i2s_read(i2s_dev, i2s_read_buffer, I2S_BYTES_TO_READ, &bytes_read);
//         if (ret != 0) {
//             LOG_ERR("I2S read error: %d", ret);
//             continue;
//         }

//         if (bytes_read > 0) {
//             ring_buf_put(&audio_ring_buffer, i2s_read_buffer, bytes_read);
//             g_latest_audio_timestamp += ((1000 * (bytes_read / 2)) / 16000); // Assuming 16KHz sample rate
//         }
//     }
// }

// int init_audio_recording(void) {
//     ring_buf_init(&audio_ring_buffer, sizeof(g_audio_output_buffer), (uint8_t *)g_audio_output_buffer);
    
//     k_thread_create(&audio_thread_data, audio_thread_stack, K_THREAD_STACK_SIZEOF(audio_thread_stack),
//                     capture_audio_samples, NULL, NULL, NULL, 7, 0, K_NO_WAIT);
    
//     while (!g_latest_audio_timestamp) {
//         k_sleep(K_MSEC(1));
//     }
//     LOG_INF("Audio recording started");
//     return 0;
// }

// int get_audio_samples(int *audio_samples_size, int16_t **audio_samples) {
//     if (!g_is_audio_initialized) {
//         int status = init_audio_recording();
//         if (status != 0) {
//             return status;
//         }
//         g_is_audio_initialized = true;
//     }

//     int bytes_read = ring_buf_get(&audio_ring_buffer, (uint8_t *)(g_audio_output_buffer), 16000);
//     if (bytes_read < 0) {
//         LOG_INF("Couldn't read data in time");
//         bytes_read = 0;
//     }
    
//     *audio_samples_size = bytes_read;
//     *audio_samples = g_audio_output_buffer;
//     return 0;
// }

// int32_t latest_audio_timestamp() {
//     return g_latest_audio_timestamp;
// }
#include <zephyr/kernel.h>
#include <zephyr/sys/ring_buffer.h>
#include <zephyr/logging/log.h>
#include </home/abhinav/zephyrproject/zephyr/samples/modules/tflite-micro/testing/src/audio_provider.h>

LOG_MODULE_REGISTER(audio_provider, LOG_LEVEL_INF);

#define AUDIO_BUFFER_SIZE 40000
#define SIMULATED_BYTES_TO_READ 3200  // Simulating 3200 bytes per read

static struct ring_buf audio_ring_buffer;
K_THREAD_STACK_DEFINE(audio_thread_stack, 4096);
struct k_thread audio_thread_data;

volatile int32_t g_latest_audio_timestamp = 0;
int16_t g_audio_output_buffer[AUDIO_BUFFER_SIZE];
bool g_is_audio_initialized = false;

// Simulated audio capture function
static void capture_audio_samples(void *arg, void *b, void *c) {
    static int16_t simulated_audio_buffer[SIMULATED_BYTES_TO_READ];

    while (1) {
        k_sleep(K_MSEC(100)); // Simulate delay (adjust based on real use case)

        // Generate dummy audio data (Replace this with real processing if needed)
        for (int i = 0; i < SIMULATED_BYTES_TO_READ; i++) {
            simulated_audio_buffer[i] = (i % 256);  // Fake waveform data
        }

        ring_buf_put(&audio_ring_buffer, (uint8_t*)simulated_audio_buffer, SIMULATED_BYTES_TO_READ);
        g_latest_audio_timestamp += ((1000 * (SIMULATED_BYTES_TO_READ / 2)) / 16000);  // Simulating 16KHz sample rate
        
        LOG_INF("Captured %d bytes of simulated audio", SIMULATED_BYTES_TO_READ);
    }
}

// Initialize audio recording (simulated)
int init_audio_recording(void) {
    ring_buf_init(&audio_ring_buffer, sizeof(g_audio_output_buffer), reinterpret_cast<uint8_t*>(g_audio_output_buffer));

    k_thread_create(&audio_thread_data, audio_thread_stack, K_THREAD_STACK_SIZEOF(audio_thread_stack),
                    capture_audio_samples, NULL, NULL, NULL, 7, 0, K_NO_WAIT);
    
    while (!g_latest_audio_timestamp) {
        k_sleep(K_MSEC(1));
    }
    LOG_INF("Simulated audio recording started");
    return 0;
}

// Get audio samples (simulated)
int GetAudioSamples(int *audio_samples_size, int16_t **audio_samples) {
    if (!g_is_audio_initialized) {
        int status = init_audio_recording();
        if (status != 0) {
            return status;
        }
        g_is_audio_initialized = true;
    }

    int bytes_read = ring_buf_get(&audio_ring_buffer, (uint8_t *)(g_audio_output_buffer), 16000);
    if (bytes_read < 0) {
        LOG_INF("Couldn't read simulated data in time");
        bytes_read = 0;
    }

    // Convert int16_t to int8_t (if necessary)
    for (int i = 0; i < bytes_read / 2; i++) {
        ((int8_t*)g_audio_output_buffer)[i] = g_audio_output_buffer[i] >> 8; // Simple quantization
    }

    *audio_samples_size = bytes_read / 2;  // Convert bytes to number of int8_t samples
    *audio_samples = (int16_t*)g_audio_output_buffer;

    LOG_INF("Returning %d int8_t samples", *audio_samples_size);
    return 0;
}

// Get latest audio timestamp (simulated)
int32_t LatestAudioTimestamp() {
    return g_latest_audio_timestamp;
}
