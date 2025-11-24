import json
from transformers import Wav2Vec2Processor

processor = Wav2Vec2Processor.from_pretrained("vitouphy/wav2vec2-xls-r-300m-timit-phoneme")
# Save the vocab to disk
with open("../vocab/vocab.json", "w") as f:
    json.dump(processor.tokenizer.get_vocab(), f)