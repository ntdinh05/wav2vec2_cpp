import onnxruntime as ort
import soundfile as sf
import numpy as np
from transformers import Wav2Vec2Processor

# 1. Setup paths
model_path = "onnx_output/model.onnx" 
audio_path = "audio_file.wav"
processor = Wav2Vec2Processor.from_pretrained("vitouphy/wav2vec2-xls-r-300m-timit-phoneme")

# 2. Process Audio (return_tensors="np" is key for ONNX)
audio_input, sample_rate = sf.read(audio_path)
inputs = processor(audio_input, sampling_rate=16000, return_tensors="np", padding=True)

# 3. Run Inference
session = ort.InferenceSession(model_path)
input_name = session.get_inputs()[0].name

logits = session.run(None, {input_name: inputs.input_values})[0]

# 4. Decode
predicted_ids = np.argmax(logits, axis=-1)
transcription = processor.batch_decode(predicted_ids)[0]

print(f"Prediction: {transcription}")