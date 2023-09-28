# For advertorial classification
gcs_uri		:= "gs://milelens_ml/advertorial_classification"
model_uri	:= "$(gcs_uri)/prebuilt_model/"
data_uri	:= "$(gcs_uri)/data/"
gcp_region  := "asia-east1"
training_job_name := "asia.gcr.io/milelens-dev/ml-advertorial-post-classifier-base"

model_folder	:= "./prebuilt_model/"
data_folder	:= "./data/"
v		:= "latest"
# repository      := "asia-east1-docker.pkg.dev/milelens-dev/ml-model-api/advertorial"
repository-base := "asia.gcr.io/milelens-dev/ml-advertorial-post-classifier-base"
repository  := "asia.gcr.io/milelens-dev/ml-advertorial-post-classifier"

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
	python advertorial/train.py train_milelens_model --use_wandb=False
#	python advertorial/train.py train_milelens_model --train_ratio=0.8 --validation_ratio=0.2 --bq_table=milelens-dev.ML.advertorial_classifier

summary_milelens:
	@echo "Summarize model"
	python advertorial/performance.py summary_milelens_model

train_milelens_all_data:
	@echo "Train model"
	python advertorial/train.py train_milelens_model --train_ratio=1 --validation_ratio=0 --bq_table=milelens-dev.ML.advertorial_classifier

docker_image_to_training_job_base: docker_build_and_push_base create_training_job_base

docker_build_and_push_base:
	docker build -t $(repository-base):$(v) . -f base.Dockerfile
	docker push $(repository-base):$(v)	

create_training_job_base:
	@gcloud ai custom-jobs create \
		--region=${gcp_region} \
		--display-name=${training_job_name} \
		--labels=ml=advertorial_post_classifier \
		--config=worker_pool_specs.yml \
		--args='-m','scripts.app','train_then_summary'