import os
import sys


from pathlib import Path
import unittest

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from advertorial.inference import AdvertorialModel


import unittest

PWD = '.' 

class AdvertorialCase(unittest.TestCase):
    def setUp(self):
        self.pwd = PWD
        self.model_path = f'{self.pwd}/prebuilt_model/230519_chinese_bert_wwm_ext'
        if not Path(self.model_path).exists():
            ValueError("Prebuilt model not found.")

        self.model = AdvertorialModel(model_path = self.model_path, use_gpu=True)

    
    def test_batch_data(self):
        text = ['三年沒來日本 第一站先衝迪士尼🇯🇵', '拉麵王子推薦新宿拉麵看了嗎？吃個日本泡麵解拉麵癮']
        # 0, 1
        prediction, probs = self.model(text, return_logit=False)
        self.assertEqual(prediction.shape, (2,))
        self.assertEqual(prediction.tolist(), [0, 1]) 
        self.assertEqual(probs.shape, (2,))


    def test_csv_data(self):
        df = pd.read_csv(f'{self.pwd}/tests/test.csv')
        texts = df['text'].tolist()
        outputs = self.model(texts, return_logit=True)
        outputs = (F.softmax(outputs.logits, dim=1) > 0.6).int()
        outputs = outputs.argmax(dim=1)
        self.assertEqual(outputs.cpu().numpy().tolist(), [1, 0, 1, 1, 1, 1, 0, 1, 0, 
                                                          1, 1, 0, 0, 0, 0, 0, 1, 0])
        



# class SentimentCase(unittest.TestCase):
#     def setUp(self) -> None:
#         self.model_path = "../prebuilt_model/230201_chinese_bert_wwm_ext"
#         if os.path.exists(self.model_path):
#             self.model = SentimentModel(self.model_path)
#         else:
#             ValueError("Prebuilt model not found")

#     def test_batch_data(self):
#         text = ["我今天心情很好", "我不開心"]
#         # batch_data = tokenizer(text, max_length=512, padding=True, truncation=True, return_tensors="pt")
#         # outputs = model(**batch_data)
#         outputs = self.model(text, return_logit=False)
#         result = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().detach().numpy()
#         if "chinese_bert_wwm_ext" in self.model_path:
#             self.assertEqual(result.shape, (2, 3))
#         else:
#             self.assertEqual(result.shape, (2, 2))

#     def test_logit_to_scores(self):
#         tensor = torch.tensor([[1.5, 0.5], [0.1, 0.4]])
#         ids, scores = logit_to_scores(tensor)
#         self.assertEqual(ids.shape, (2,), msg="Check ids shape")
#         self.assertEqual(scores.shape, (2,), msg="Check scores shape")

#     def test_csv_data(self):
#         df = pd.read_csv("mention_list.csv")
#         texts = df["mention_post_text"].tolist()
#         # model = SentimentModel()
#         len_texts = len(texts)

#         batch_size: int = 8
#         result_ids, result_scores = [], []
#         for i in range(0, len_texts, batch_size):
#             batch_texts = texts[i: i + batch_size] if i + batch_size <= len_texts else texts[i:len_texts]
#             ids, scores = self.model(batch_texts)
#             result_ids.append(ids)
#             result_scores.append(scores)

#         result_ids = list(np.concatenate(result_ids, axis=0))
#         result_scores = list(np.concatenate(result_scores, axis=0))

#         if "chinese_bert_wwm_ext" in self.model_path:
#             sentiment_dt = {1: "positive", 0: "negative", 2: "natural"}
#         else:
#             sentiment_dt = {1: "positive", 0: "negative"}
#         df["label"] = [sentiment_dt[i] for i in result_ids]
#         df["score"] = result_scores
#         df.to_csv("./result.csv", index=False)


if __name__ == "__main__":
    unittest.main()
