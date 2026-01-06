#define MINIAUDIO_IMPLEMENTATION
#include "../libs/miniaudio.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <map>
#include <mutex>
#include <thread>
#include <atomic>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <onnxruntime_cxx_api.h>
#include "json.hpp"

using json = nlohmann::json;

// --- CONFIGURATION ---
const int SAMPLE_RATE = 16000;
// We wait for 3 seconds of audio, process it, print it, then clear it.
// This prevents "Hello Hello Hello" loops and ensures the model has enough context.
const int WINDOW_DURATION_SEC = 3; 
const size_t WINDOW_SIZE = SAMPLE_RATE * WINDOW_DURATION_SEC;

// --- GLOBAL VARIABLES ---
std::vector<float> g_audioBuffer; 
std::mutex g_bufferMutex;         
bool g_isRunning = true;
// Track if we are actually receiving signal or just silence/zeros
std::atomic<float> g_maxAmplitude{0.0f}; 

// --- VOCAB LOADER ---
std::map<int, std::string> LoadVocab(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        std::cerr << "Error: Could not open vocab file at " << path << std::endl;
        exit(1);
    }
    json data = json::parse(f);
    std::map<int, std::string> id_to_token;
    for (auto& element : data.items()) {
        id_to_token[element.value()] = element.key();
    }
    return id_to_token;
}

// --- MICROPHONE CALLBACK ---
void data_callback(ma_device* pDevice, void* pOutput, const void* pInput, ma_uint32 frameCount) {
    const float* inputFloats = (const float*)pInput;
    
    // Check if we are getting zeros (Permission issue or hardware mute)
    float localMax = 0.0f;
    for (ma_uint32 i = 0; i < frameCount; ++i) {
        float val = std::abs(inputFloats[i]);
        if (val > localMax) localMax = val;
    }
    float currentMax = g_maxAmplitude.load();
    if (localMax > currentMax) g_maxAmplitude.store(localMax);

    std::lock_guard<std::mutex> lock(g_bufferMutex);
    g_audioBuffer.insert(g_audioBuffer.end(), inputFloats, inputFloats + frameCount);
}

// --- INFERENCE ENGINE ---
std::string RunInference(std::vector<float>& samples, Ort::Session& session, std::map<int, std::string>& vocab) {
    // 1. Normalization
    double sum = 0.0;
    for (float x : samples) sum += x;
    double mean = sum / samples.size();

    double sq_sum = 0.0;
    for (float x : samples) {
        double diff = x - mean;
        sq_sum += diff * diff;
    }
    double stdev = std::sqrt(sq_sum / samples.size());
    double epsilon = 1e-5;

    std::vector<float> norm_samples = samples;
    for (auto& x : norm_samples) {
        x = (float)((x - mean) / (stdev + epsilon));
    }

    // 2. Run ONNX Model
    size_t batch_size = 1;
    std::vector<int64_t> input_dims = { (int64_t)batch_size, (int64_t)norm_samples.size() };
    
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, norm_samples.data(), norm_samples.size(), input_dims.data(), input_dims.size()
    );

    Ort::AllocatorWithDefaultOptions allocator;
    auto input_name_ptr = session.GetInputNameAllocated(0, allocator);
    auto output_name_ptr = session.GetOutputNameAllocated(0, allocator);
    const char* input_names[] = { input_name_ptr.get() };
    const char* output_names[] = { output_name_ptr.get() };

    auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);

    float* floatarr = output_tensors[0].GetTensorMutableData<float>();
    auto shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    int seq_len = shape[1];
    int vocab_size = shape[2];

    std::string text = "";
    int previous_index = -1; 

    // 3. Decode Tokens
    for (int t = 0; t < seq_len; t++) {
        float* logits = floatarr + (t * vocab_size);
        int max_index = 0; 
        float max_val = logits[0];
        for (int v = 1; v < vocab_size; v++) {
            if (logits[v] > max_val) { max_val = logits[v]; max_index = v; }
        }

        // Greedy Decode with specific cleanup
        if (max_index != previous_index) {
            std::string token = vocab[max_index];
            // Filter out common special tokens aggressively
            bool is_special = (token == "[PAD]" || token == "<pad>" || token == "<s>" || token == "</s>" || token == "<unk>" || token == "|");
            
            if (!is_special) {
                text += token;
            } else if (token == "|") {
                text += " ";
            }
        }
        previous_index = max_index;
    }
    return text;
}

// --- MAIN ---
int main() {
    // FORCE FLUSH to ensure text appears
    std::cout.setf(std::ios::unitbuf);

    std::string vocab_path = "../vocab/vocab.json";
    std::string model_path = "../onnx_output/model.onnx";

    Ort::Env env(ORT_LOGGING_LEVEL_ERROR, "LiveRec"); 
    Ort::SessionOptions session_options;
    Ort::Session session(env, model_path.c_str(), session_options);

    auto vocab = LoadVocab(vocab_path);

    ma_device_config config = ma_device_config_init(ma_device_type_capture);
    config.capture.format = ma_format_f32; 
    config.capture.channels = 1;           
    config.sampleRate = SAMPLE_RATE;       
    config.dataCallback = data_callback;   
    config.capture.shareMode = ma_share_mode_shared; 

    ma_device device;
    if (ma_device_init(NULL, &config, &device) != MA_SUCCESS) {
        std::cerr << "FAILED to init microphone!" << std::endl;
        return -1;
    }
    ma_device_start(&device);

    std::cout << ">>> SYSTEM LIVE: Speak now." << std::endl;

    // Check for "Dead Microphone" (Permissions Issue)
    std::this_thread::sleep_for(std::chrono::seconds(1));
    if (g_maxAmplitude.load() == 0.0f) {
        std::cout << "\n[!] WARNING: No audio signal detected." << std::endl;
        std::cout << "[!] CHECK MACOS PRIVACY SETTINGS -> MICROPHONE." << std::endl;
    }

    while (g_isRunning) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        std::vector<float> current_window;
        bool should_process = false;

        {
            std::lock_guard<std::mutex> lock(g_bufferMutex);
            if (g_audioBuffer.size() >= WINDOW_SIZE) {
                // Copy the buffer
                current_window = g_audioBuffer;
                // CLEAR the buffer completely (No overlapping, prevents duplication)
                g_audioBuffer.clear();
                should_process = true;
            }
        }

        if (should_process) {
            std::string result = RunInference(current_window, session, vocab);
            
            if (!result.empty()) {
                // "TR:" is our secret tag for "Transcript"
                // std::endl forces the output to be sent to Python IMMEDIATELY
                std::cout << "TR:" << result << std::endl;
            }
        }
    }

    ma_device_uninit(&device);
    return 0;
}