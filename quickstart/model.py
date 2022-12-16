from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
from datasets import Dataset
import torch
from torch.utils.data import DataLoader

from config import logger, ModelConfig

# Use Cuda device when available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class ModelWrapper():

    def __init__(self, model_dir: str, download_model: bool = True) -> None:
        self._model_dir = model_dir
        if download_model:
            self._download_model()
        self._tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(device)
        
    def _download_model(self):
        logger.info(f"Downloading model with path {ModelConfig.HF_MODEL_PATH} from HuggingFace")
        tokenizer = AutoTokenizer.from_pretrained(ModelConfig.HF_MODEL_PATH)
        tokenizer.save_pretrained(self._model_dir)
        model = AutoModelForSeq2SeqLM.from_pretrained(ModelConfig.HF_MODEL_PATH)
        model.save_pretrained(self._model_dir)

    def _tokenize_sentence(self, text: str): 
        return self._tokenize_helper(text)

    def _get_tokenize_batch(self):
        def _tokenize_batch(batch):
            return self._tokenize_helper(batch["text"])
        return _tokenize_batch
    
    def _tokenize_helper(self, input):
        return self._tokenizer(
            input, 
            max_length=512, 
            truncation=True, 
            padding='max_length', 
            return_tensors='pt'
        ).to(device)

    def run_batch_inference(self, data: pd.DataFrame, text_col: str):
        data_loader = self._preprocess_data(data, text_col)
        result = []
        for batch in data_loader:
            decoded_outputs = self._infer_batch(batch)
            result.extend(decoded_outputs)
        return result

    def _preprocess_data(self, df: pd.DataFrame, text_col: str) -> DataLoader:
        dataset = Dataset.from_pandas(df[[text_col]]).with_format("torch", device=device)
        dataset.set_transform(self._get_tokenize_batch())
        data_loader = DataLoader(dataset, batch_size=16, shuffle=False)
        return data_loader

    def _infer_batch(self, batch: dict):
        model_output = self._infer(batch["input_ids"], batch["attention_mask"])
        corrected_sentences = [self._decode_output(output) for output in model_output]
        return corrected_sentences

    def infer(self, text: str):
        encoded_text = self._tokenize_sentence(text)
        model_output = self._infer(encoded_text.input_ids, encoded_text.attention_mask)
        corrected_sentence = self._decode_output(model_output[0])
        return corrected_sentence

    def _decode_output(self, model_output: torch.tensor):
        return self._tokenizer.decode(
            model_output,
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=True
        )

    def _infer(self, input_ids: torch.tensor, attention_mask: torch.tensor):
        model_output = self._model.generate(
                input_ids = input_ids,
                attention_mask = attention_mask, 
                max_length=512,
                num_beams=5,
                early_stopping=True,
        )
        return model_output
