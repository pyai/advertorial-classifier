from typing import List
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import pickle as pkl
import time
from advertorial.inference import AdvertorialModel


PORT = 8088



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
    docker run -it -p 8090:8080 3b1b765a0060
    """
    title = f"start API at sport \033[36m{PORT}\033[0m"
    printTitle(title)
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=True)
