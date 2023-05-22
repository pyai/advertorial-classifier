# For advertorial classification
gcs_uri		:= "gs://milelens_ml/advertorial_classification"
model_uri	:= "$(gcs_uri)/prebuilt_model/"
data_uri	:= "$(gcs_uri)/data/"

model_folder	:= "./prebuilt_model/"
data_folder		:= "./data/"

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
	python train_milelens.py train_milelens_model --use_wandb --train_ratio=1
