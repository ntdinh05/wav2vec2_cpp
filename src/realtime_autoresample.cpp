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
// const float SPEECH_THRESHOLD = 0.02f;           // Amplitude threshold to trigger recording (Noise Gate)
// const float SILENCE_DURATION_THRESHOLD = 0.8f;  // Seconds of silence to mark 'End of Sentence'
// const float MAX_SPEECH_DURATION = 15.0f;        // Force inference if speech is too long (safety limit)

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
    double epsilon = 1e-5;     // Try without doing the normalization

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
    
    // SentencePiece underscore (U+2581) byte sequence
    const std::string SP_SPACE = "\xe2\x96\x81"; 

    // 3. Decode Tokens
    for (int t = 0; t < seq_len; t++) {
        // Find Argmax
        float* logits = floatarr + (t * vocab_size);
        int max_index = 0; 
        float max_val = logits[0];
        for (int v = 1; v < vocab_size; v++) {
            if (logits[v] > max_val) { max_val = logits[v]; max_index = v; }
        }

        // Greedy CTC Decode: Skip repeated tokens immediately
        if (max_index == previous_index) continue;
        previous_index = max_index;

        std::string token = vocab[max_index];

        // 1. Skip Special/Ignored Tokens
        if (token == "[PAD]" || token == "<pad>" || token == "<s>" || token == "</s>" || token == "<unk>") {
            continue;
        }

        // 2. Handle Wav2Vec/Whisper explicit space tokens
        if (token == "|" || token == "<|space|>") {
            text += " ";
            continue;
        }

        // 3. Handle SentencePiece Underscores (e.g. " word" -> " word" or "_" -> " ")
        // This loop safely replaces all instances of the special underscore with a real space
        size_t pos = 0;
        while ((pos = token.find(SP_SPACE, pos)) != std::string::npos) {
            token.replace(pos, SP_SPACE.length(), " ");
            pos += 1; 
        }

        text += token;
    }
    return text;
}

// --- MAIN ---
int main() {
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
        std::cout << "\nNo audio signal detected." << std::endl;
    }

    // --- VAD STATE VARIABLES ---
    std::vector<float> accumulated_audio;
    
    // Track previous output to detect and remove duplicates
    std::string last_transcription = ""; 

    while (g_isRunning) {
        // Fetch audio chunk (approx 100ms usually)
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        std::vector<float> chunk;
        {
            std::lock_guard<std::mutex> lock(g_bufferMutex);
            if (!g_audioBuffer.empty()) {
                chunk = g_audioBuffer;
                g_audioBuffer.clear();
            }
        }

        if (chunk.empty()) continue;

        float chunk_duration = (float)chunk.size() / SAMPLE_RATE;
        // std::cout<<"Chunk Duration: " << std::fixed << std::setprecision(3) << chunk_duration << "s\r";
        accumulated_audio.insert(accumulated_audio.end(), chunk.begin(), chunk.end());

        float current_total_duration = (float)accumulated_audio.size() / SAMPLE_RATE;
        if (current_total_duration >= 0.5f) {
            std::string current_raw = RunInference(accumulated_audio, session, vocab);
            std::string display_text = current_raw;

            // --- DE-DUPLICATION CHECK ---
            // Since we overlap audio, the tokens at the start of 'current_raw' 
            // might match the tokens at the end of 'last_transcription'.
            if (!last_transcription.empty() && !current_raw.empty()) {
                size_t max_check = std::min(last_transcription.length(), current_raw.length());
                size_t best_overlap = 0;

                // Check suffixes of 'last' against prefixes of 'current'
                // We limit the check range because the audio overlap is short (0.4s)
                size_t limit_chars = std::min(max_check, (size_t)50); 
                
                for (size_t i = 1; i <= limit_chars; i++) {
                    std::string suffix = last_transcription.substr(last_transcription.length() - i);
                    std::string prefix = current_raw.substr(0, i);
                    if (suffix == prefix) {
                        best_overlap = i;
                    }
                }
                
                // If overlap found, strip it from the start of the current text
                if (best_overlap > 0) {
                    display_text = current_raw.substr(best_overlap);
                }
            }

            if (!display_text.empty()) {
                std::cout << "TR:" << display_text << std::endl;
            }
            
            // Save the raw full output for the NEXT comparison
            last_transcription = current_raw;
            
            // --- OVERLAP LOGIC ---
            // Keep the last 0.4 seconds of audio to preserve context for cut-off words
            size_t keep_samples = (size_t)(0.4f * SAMPLE_RATE);
            
            if (accumulated_audio.size() > keep_samples) {
                // Create a new buffer starting with the tail of the old one
                std::vector<float> tail(accumulated_audio.end() - keep_samples, accumulated_audio.end());
                accumulated_audio = tail;
            } else {
                accumulated_audio.clear();
                last_transcription = ""; // Reset history if audio chain breaks
            }
        }
    }
    ma_device_uninit(&device);
    return 0;
}