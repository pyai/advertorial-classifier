# For advertorial classification
gcs_uri		:= "gs://milelens_ml/advertorial_classification"
model_uri	:= "$(gcs_uri)/prebuilt_model/"
data_uri	:= "$(gcs_uri)/data/"

model_folder	:= "./prebuilt_model/"
data_folder	:= "./data/"
v		:= "v1.1"
repository      := "asia-east1-docker.pkg.dev/milelens-dev/ml-model-api/advertorial"

init: download_data download_models

prod: download_models


download_data:
	@echo "Download data from GCS"
	@gsutil ls $(data_uri) | gsutil -m cp -r -I $(data_folder)

download_models:
	@echo "Download models from GCS"
	@gsutil ls $(model_uri) | gsutil -m cp -r -I $(model_folder)


train_milelens:
	@echo "Train model"
	python advertorial/train.py train_milelens_model --train_ratio=0.8 --validation_ratio=0.2

train_milelens_all_data:
	@echo "Train model"
	python advertorial/train.py train_milelens_model --train_ratio=1 --validation_ratio=0


docker_build_and_push:
	docker build -t $(repository):$(v) .
	docker push $(repository):$(v)	
