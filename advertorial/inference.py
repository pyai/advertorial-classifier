import torch
import torch.nn.functional as F
import numpy as np
from typing import List
from transformers import AutoTokenizer, AutoModelForSequenceClassification



class AdvertorialModel:
    """Advertorial classification inference model"""

    def __init__(self, model_path:str='./prebuilt_model/230629_chinese_bert_wwm_ext', use_gpu:bool=False):
        self.model_path = model_path
        self.device = "cuda:0" if torch.cuda.is_available() and use_gpu else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path).to(self.device)

    def logit_to_scores(self, logit: torch.tensor):
        """Compute logit to score"""
        probs = F.softmax(logit, dim=-1).cpu().detach().numpy()
        ids, scores = np.argmax(probs, axis=1), np.max(probs, axis=1)

        return ids, scores

    def __call__(self, texts:List, return_logit=False):
        inputs = self.tokenizer(texts, max_length=512,
                                    padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        if return_logit:
            return outputs
        else:
            return self.logit_to_scores(outputs.logits)