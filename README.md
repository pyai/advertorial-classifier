# Advertorial Classifier


## Getting started

### For production

- Use `make prod` with Makefile to download pretrained model

### For dev

- Use `make init` with Makefile to download data and pretrained model

#### Execute python
- Use `python advertorial/train.py` to train model
- Check `notebooks/check.ipynb` for performance check

#### Start service
- `uvicorn scripts.main:app --port 8088 --reload`
- ` docker run -it -p 8090:8080 image`

#### Model Performance
|date|version|dataset|records|positive samples|negative samples|hit|miss|accuracy|miss rate|  
|--|--|--|--|--|--|--|--|--|--|   
|2023/05/27|v1.3|train|18900|7446|11454|18120|780|0.95873|0.04127|  

#### Access Api
- `curl -X POST -H "Content-Type: application/json" -d '{"texts": ["三年 沒來日本 第一站先衝迪士尼🇯🇵", "拉麵王子推薦新宿拉麵看了嗎？吃個日本泡麵解拉麵癮"]}'https://ml-advertorial-post-classifier-6tfv4ijbmq-de.a.run.app/advertorial`
- `curl -X POST -H "Content-Type: application/json" https://ml-advertorial-post-classifier-6tfv4ijbmq-de.a.run.app/performance`
- `curl -X POST -H "Content-Type: application/json" -d '{"texts":["三年沒來日本 第一站先衝迪士尼🇯🇵", "拉麵王子推薦新宿拉麵看了嗎？吃個日本泡麵解拉麵癮"]}' http://127.0.0.1:8088/advertorial`
- `curl -X POST -H "Content-Type: application/json"  http://127.0.0.1:8088/performance`


#### Dataset version

|date|data event|
|--|--|
|20231017|roll backed|
|20231115|relabel|  
