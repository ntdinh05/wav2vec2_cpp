#define MINIAUDIO_IMPLEMENTATION
#include "../libs/miniaudio.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <map>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <onnxruntime_cxx_api.h>
#include "json.hpp"

using json = nlohmann::json;

// --- CONFIGURATION ---
const int SAMPLE_RATE = 16000;
const std::string INPUT_WAV_PATH = "../tests/TEST_DR1_FAKS0_SA1/SA1.WAV.wav"; // <--- Ensure this file exists
const std::string OUTPUT_CSV_PATH = "../output/experiment_results_raw.csv";

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

// --- INFERENCE ENGINE ---
std::string RunInference(std::vector<float>& samples, Ort::Session& session, std::map<int, std::string>& vocab) {
    if (samples.empty()) return "";

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

    try {
        auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);

        float* floatarr = output_tensors[0].GetTensorMutableData<float>();
        auto shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
        int seq_len = shape[1];
        int vocab_size = shape[2];

        std::string text = "";
        int previous_index = -1; 
        const std::string SP_SPACE = "\xe2\x96\x81"; 

        // 3. Decode Tokens
        for (int t = 0; t < seq_len; t++) {
            float* logits = floatarr + (t * vocab_size);
            int max_index = 0; 
            float max_val = logits[0];
            for (int v = 1; v < vocab_size; v++) {
                if (logits[v] > max_val) { max_val = logits[v]; max_index = v; }
            }

            if (max_index == previous_index) continue;
            previous_index = max_index;

            std::string token = vocab[max_index];

            if (token == "[PAD]" || token == "<pad>" || token == "<s>" || token == "</s>" || token == "<unk>") continue;
            if (token == "|" || token == "<|space|>") { text += " "; continue; }

            size_t pos = 0;
            while ((pos = token.find(SP_SPACE, pos)) != std::string::npos) {
                token.replace(pos, SP_SPACE.length(), " ");
                pos += 1; 
            }
            text += token;
        }
        return text;
    } catch (const Ort::Exception& e) {
        std::cerr << " [ONNX Error: Chunk=" << samples.size() << " samples] " << std::endl;
        return "";
    }
}

// --- MAIN ---
int main() {
    std::cout.setf(std::ios::unitbuf);

    std::string vocab_path = "../vocab/vocab.json";
    std::string model_path = "../onnx_output/model.onnx";

    Ort::Env env(ORT_LOGGING_LEVEL_ERROR, "RawExperiment"); 
    Ort::SessionOptions session_options;
    Ort::Session session(env, model_path.c_str(), session_options);

    auto vocab = LoadVocab(vocab_path);

    // --- LOAD WAV FILE ---
    ma_decoder decoder;
    ma_decoder_config config = ma_decoder_config_init(ma_format_f32, 1, SAMPLE_RATE);
    
    if (ma_decoder_init_file(INPUT_WAV_PATH.c_str(), &config, &decoder) != MA_SUCCESS) {
        std::cerr << "Could not open WAV file: " << INPUT_WAV_PATH << std::endl;
        return -1;
    }

    std::vector<float> file_audio;
    ma_uint64 totalFrames;
    if (ma_decoder_get_length_in_pcm_frames(&decoder, &totalFrames) != MA_SUCCESS) {
        std::cerr << "Could not get length of WAV file." << std::endl;
        return -1;
    }

    file_audio.resize(totalFrames);
    
    ma_uint64 framesRead;
    if (ma_decoder_read_pcm_frames(&decoder, file_audio.data(), totalFrames, &framesRead) != MA_SUCCESS) {
        std::cerr << "Could not read WAV file frames." << std::endl;
        return -1;
    }
    file_audio.resize(framesRead); 
    ma_decoder_uninit(&decoder);
    
    std::cout << "Loaded " << file_audio.size() << " samples." << std::endl;

    // --- EXPERIMENT LOOP ---
    std::vector<int> chunk_lengths_ms = {50, 100, 200, 500, 1000, 2000, 5000};
    
    std::ofstream csv_file(OUTPUT_CSV_PATH);
    csv_file << "Chunk Length (ms),Raw Transcription\n";

    for (int ms : chunk_lengths_ms) {
        std::cout << "Testing chunk size: " << ms << "ms... ";
        
        int samples_per_chunk = (SAMPLE_RATE * ms) / 1000;
        std::string final_output = "";
        
        size_t cursor = 0;
        while (cursor < file_audio.size()) {
            size_t remaining = file_audio.size() - cursor;
            size_t current_chunk_size = std::min((size_t)samples_per_chunk, remaining);
            
            // Extract raw chunk
            std::vector<float> chunk(file_audio.begin() + cursor, file_audio.begin() + cursor + current_chunk_size);
            cursor += current_chunk_size;

            // Direct inference on small raw chunk
            // Wave2Vec2 has a receptive field of ~400 samples (25ms). Smaller inputs cause convolution errors.
            if (chunk.size() >= 400) {
                std::string text = RunInference(chunk, session, vocab);
                final_output += text;
            }
        }
        
        // CSV Escape
        std::string safe_output = "";
        for (char c : final_output) {
            if (c == '"') safe_output += "\"\"";
            else safe_output += c;
        }
        
        csv_file << ms << ",\"" << safe_output << "\"\n";
        std::cout << "Done." << std::endl;
    }

    csv_file.close();
    std::cout << "Results saved to " << OUTPUT_CSV_PATH << std::endl;
    return 0;
}