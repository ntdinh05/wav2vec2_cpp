#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <fstream>
#include <cmath>
#include <onnxruntime_cxx_api.h>
#include "json.hpp" 
#include "AudioFile.h" // This relies on the header you added to 'libs'

using json = nlohmann::json;

// --- 1. Load Vocab ---
std::map<int, std::string> LoadVocab(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        std::cerr << "ERROR: Could not open vocab at " << path << std::endl;
        exit(1);
    }
    json data = json::parse(f);
    std::map<int, std::string> id_to_token;
    for (auto& element : data.items()) {
        id_to_token[element.value()] = element.key();
    }
    return id_to_token;
}

// --- 2. Load AND Normalize Audio ---
std::vector<float> LoadAudioFile(const std::string& path) {
    AudioFile<float> audioFile;
    if (!audioFile.load(path)) {
        std::cerr << "ERROR: Could not load audio file at " << path << std::endl;
        exit(1);
    }

    // Get the first channel (Mono)
    std::vector<float> samples = audioFile.samples[0];
    
    // Wav2Vec2 REQUIRES normalization (Zero Mean, Unit Variance)
    // Calculate Mean
    double sum = std::accumulate(samples.begin(), samples.end(), 0.0);
    double mean = sum / samples.size();

    // Calculate Variance/Std Dev
    double sq_sum = std::inner_product(samples.begin(), samples.end(), samples.begin(), 0.0);
    double stdev = std::sqrt(sq_sum / samples.size() - mean * mean);
    
    // Epsilon to avoid division by zero
    double epsilon = 1e-5;

    // Apply Normalization: (x - mean) / sqrt(var + epsilon)
    for (auto& x : samples) {
        x = (x - mean) / (stdev + epsilon);
    }

    return samples;
}

int main() {
    // --- PATHS (Adjusted for running from 'build' folder) ---
    // Assuming structure:
    // root/
    //   build/ (executable here)
    //   vocab/vocab.json
    //   scripts/audio_file.wav
    std::string vocab_path = "../vocab/vocab.json";
    std::string audio_path = "../audio_file.wav"; 
    std::string model_path = "../onnx_output/model.onnx"; // Verify this path matches yours

    std::cout << "Loading Model..." << std::endl;
    
    // --- Load Resources ---
    auto vocab = LoadVocab(vocab_path);
    auto input_audio = LoadAudioFile(audio_path);

    // --- ONNX Setup ---
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "Wav2Vec2");
    Ort::SessionOptions session_options;
    Ort::Session session(env, model_path.c_str(), session_options);

    // --- Create Input Tensor ---
    size_t batch_size = 1;
    size_t sequence_length = input_audio.size();
    std::vector<int64_t> input_dims = { (int64_t)batch_size, (int64_t)sequence_length };

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, 
        input_audio.data(), 
        input_audio.size(), 
        input_dims.data(), 
        input_dims.size()
    );

    // --- Run Inference ---
    Ort::AllocatorWithDefaultOptions allocator;
    auto input_name_ptr = session.GetInputNameAllocated(0, allocator);
    auto output_name_ptr = session.GetOutputNameAllocated(0, allocator);
    const char* input_names[] = { input_name_ptr.get() };
    const char* output_names[] = { output_name_ptr.get() };

    std::cout << "Running Inference..." << std::endl;
    auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);

    // --- Decode Output (CTC Greedy Decode) ---
    float* floatarr = output_tensors[0].GetTensorMutableData<float>();
    auto shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    int seq_len = shape[1];
    int vocab_size = shape[2];

    std::string full_transcription = "";
    int previous_index = -1; // To track repeated characters

    for (int t = 0; t < seq_len; t++) {
        float* step_logits = floatarr + (t * vocab_size);
        
        // Argmax: Find best index for this time step
        int max_index = 0;
        float max_val = step_logits[0];
        for (int v = 1; v < vocab_size; v++) {
            if (step_logits[v] > max_val) {
                max_val = step_logits[v];
                max_index = v;
            }
        }

        // CTC Logic:
        // 1. Ignore if it's the same as the previous index (merge repeats)
        // 2. Ignore if it's the [PAD] token (or whatever token maps to the blank ID, usually 0)
        if (max_index != previous_index) {
            std::string token = vocab[max_index];
            
            // Filter out special tokens. Adjust these strings if your vocab differs.
            if (token != "[PAD]" && token != "<pad>" && token != "|") {
                // Replace delimiter with space if using word delimiters
                if (token == "|") full_transcription += " "; 
                else full_transcription += token;
            }
        }
        previous_index = max_index;
    }

    std::cout << "Prediction: " << full_transcription << std::endl;

    return 0;
}