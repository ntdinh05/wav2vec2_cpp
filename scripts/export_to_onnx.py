from transformers import Wav2Vec2Processor
from optimum.onnxruntime import ORTModelForCTC

model_id = "vitouphy/wav2vec2-xls-r-300m-timit-phoneme"
onnx_path = "onnx_output"  # saving directory for ONNX model

model = ORTModelForCTC.from_pretrained(model_id, export=True)
processor = Wav2Vec2Processor.from_pretrained(model_id)

model.save_pretrained(onnx_path)
processor.save_pretrained(onnx_path)