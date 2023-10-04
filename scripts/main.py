from typing import List
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import os
import json
import time
from advertorial.inference import AdvertorialModel
from google.cloud import bigquery
from advertorial import utils

PORT = 8088

envfile='.env'
utils.check_env(envfile)

# get bq settings from environment variable
project_id, dataset_id = os.environ['GCP_PROJECT'], os.environ['GCP_BQ_DATASET']
meta_table_id= os.environ['GCP_BQ_META_TABLE']
region=os.environ['GCP_REGION']

# Load BQ table into dataframe
client = bigquery.Client(project=project_id,
                         location=region)

model = AdvertorialModel(use_gpu=True)
class Query(BaseModel):
    mode: str='sequential'
    texts: List[str]


app = FastAPI()

@app.post("/advertorial")
async def advertorial(q: Query):
    start_time = time.time()
    prediction, probs = model(q.texts, return_logit=False)
    end_time = time.time()
    return {"labels": prediction.tolist(), "odds":probs.tolist(), 'time':round(end_time-start_time,2)}

@app.post("/performance")
async def performance():
    start_time = time.time()
    sql = f"SELECT * FROM `{project_id}.{dataset_id}.{meta_table_id}` ORDER BY date DESC LIMIT 1;"
    meta = client.query(sql).to_dataframe()
    
    train_meta, test_meta = json.loads(meta['train_set_meta'].item()), json.loads(meta['test_set_meta'].item())
    train_acc = train_meta['accuracy']*100
    test_acc, test_fpr = test_meta['accuracy']*100, test_meta['false pos rate']*100
    end_time = time.time()
    return {'訓練集準確度':f'{train_acc:.2f}%', '驗證集準確度':f'{test_acc:.2f}%', '非業配文被誤判率':f'{test_fpr:.2f}%', 'time':round(end_time-start_time,2)}

def printTitle(title):
    print(f"-" * (len(title) - 9 + 4))
    print(f"| {title} |")
    print(f"-" * (len(title) - 9 + 4))

if __name__ == "__main__":
    """
    examples:
    uvicorn scripts.main:app --port 8090 --reload
    curl -X POST -H "Content-Type: application/json" -d '{"texts":["三年沒來日本 第一站先衝迪士尼🇯🇵", "拉麵王子推薦新宿拉麵看了嗎？吃個日本泡麵解拉麵癮"]}' http://127.0.0.1:8090/advertorial
    curl -X POST -H "Content-Type: application/json" -d '{"texts": ["三年 沒來日本 第一站先衝迪士尼🇯🇵", "拉麵王子推薦新宿拉麵看了嗎？吃個日本泡麵解拉麵癮"]}' https://advertorial-6tfv4ijbmq-de.a.run.app/advertorial
    curl -X POST -H "Content-Type: application/json" -d '{"texts":["兩妞進入畢業倒數😆 媽媽也從早起熱便當放保溫餐盒 進化到～ 現做餐盒11點多直送學校 #持續好幾週  #把握最後國小還在家裡附近的機會 #傳遞熱呼呼的愛♥️", "拉麵王子推薦新宿拉麵看了嗎？吃個日本泡麵解拉麵癮"]}' http://127.0.0.1:8090/advertorial
    docker run -it -p 8090:8080 3b1b765a0060
    """
    title = f"start API at sport \033[36m{PORT}\033[0m"
    printTitle(title)
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=True)


