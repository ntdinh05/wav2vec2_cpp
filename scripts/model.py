from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC 
from datasets import load_dataset
import torch
import soundfile as sf

# load model and processor
processor = Wav2Vec2Processor.from_pretrained("vitouphy/wav2vec2-xls-r-300m-timit-phoneme")
model = Wav2Vec2ForCTC.from_pretrained("vitouphy/wav2vec2-xls-r-300m-timit-phoneme")

# Read and process the input
audio_input, sample_rate = sf.read("audio_file.wav")
inputs = processor(audio_input, sampling_rate=16_000, return_tensors="pt", padding=True)

with torch.no_grad():
    logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits

# Decode id into string
predicted_ids = torch.argmax(logits, axis=-1)      
predicted_sentences = processor.batch_decode(predicted_ids)
print(predicted_sentences)
