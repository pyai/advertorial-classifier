import datetime


import fire
import google.cloud.aiplatform as aip
import kfp
from google.cloud import storage
from kfp import compiler
from kfp.dsl import component

# ---

import os
from advertorial import utils

ENVFILE='.env'
utils.check_env(ENVFILE)
model_project = os.environ['MODEL_PROJECT']
project_id, dataset_id = os.environ['GCP_PROJECT'], os.environ['GCP_BQ_DATASET']
region = os.environ['GCP_REGION']
bucket_name = os.environ['GCS_BUCKET']
meta_table_id = os.environ['GCP_BQ_META_TABLE']
model_uri_base = os.environ['GCS_MODEL_URI_BASE']

pipeline_name=model_project
pipeline_root = os.environ['GCV_AI_PIPELINE']
compiled_pipe_file = "advertorial_post_classifier_pipe_file.json"
pipeline_template_file = f"{pipeline_root}/{compiled_pipe_file}"
pipeline_labels = {"ml": "advertorial_post_classifier"}
blob_name = f"advertorial_post_classification/pipeline/{compiled_pipe_file}"
job_id = f"{pipeline_name}-{datetime.datetime.now().strftime('%y%m%d-%H%M%S')}"


print(f'model_project={model_project}, project_id={project_id}, dataset_id={dataset_id}, region={region}, bucket_name={bucket_name}, meta_table_id={meta_table_id}, model_uri_base={model_uri_base}')
print('--'*30)
print(f'pipeline_name={pipeline_name}, pipeline_root={pipeline_root}, compiled_pipe_file={compiled_pipe_file}, pipeline_template_file={pipeline_template_file}, pipeline_labels={pipeline_labels}, blob_name={blob_name}, job_id={job_id}')

# -- project_id = "milelens-dev"
# -- location = "asia-east1"
# -- bucket_name = "milelens_ml"
# -- compiled_pipe_file = "advertorial_post_classifier_pipe_file.json"

# DO NOT MODIFY
# -- pipeline_name = "post-classifier"
# -- pipeline_root = f"gs://{bucket_name}/post_classification/pipeline"
# -- pipeline_template_file = f"{pipeline_root}/{compiled_pipe_file}"
# -- pipeline_labels = {"ml": "post_classifier"}
# -- blob_name = f"post_classification/pipeline/{compiled_pipe_file}"
# -- job_id = f"{pipeline_name}-{datetime.datetime.now().strftime('%y%m%d-%H%M%S')}"


# with open("worker_specs.yaml", 'r') as stream:
#     try:
#         parsed_yaml = yaml.safe_load(stream)
#         print(parsed_yaml)
#         worker_pool_specs = parsed_yaml["workerPoolSpecs"]
#     except yaml.YAMLError as exc:
#         print(exc)

@component(base_image='asia.gcr.io/milelens-dev/ml-advertorial-post-classifier-base:latest')
def train() -> str:
    from advertorial.train import train
    today = train()
    return today


@component(base_image='asia.gcr.io/milelens-dev/ml-advertorial-post-classifier-base:latest')
def evaluate(today: str):
    from advertorial.performance import summary
    summary(today=today)


@kfp.dsl.pipeline(
    name='advertorial-post-classifier-pipeline',
    description='An pipeline that executes training advertorial post classification model',
    pipeline_root=pipeline_root
)
def pipeline():
    # https://cloud.google.com/compute/vm-instance-pricing?hl=zh-tw#n1_predefined
    # https://cloud.google.com/vertex-ai/docs/training/configure-compute#specifying_gpus
    # n1-standard-4 4       15
    # n1-standard-8	8 CPUs  30GB
    # n1-highmem-4  4       26
    # n1-highmem-8	8 CPUs  52GB
    # n1-highmem-16	16 CPUs  104GB

    # custom_job_task = CustomTrainingJobOp(
    #     project="milelens-dev",
    #     display_name="Post-classifer",
    #     location="asia-east1",
    #     worker_pool_specs=[worker_pool_specs]
    # )
    train_task = train().set_cpu_limit('8').set_memory_limit('60G').add_node_selector_constraint(
        'NVIDIA_TESLA_P100')
    evaluate_task = evaluate(today=train_task.output).set_cpu_limit('8').set_memory_limit(
        '30G').add_node_selector_constraint('NVIDIA_TESLA_P100')
    evaluate_task.after(train_task)


def do_compile():
    compiler.Compiler().compile(pipeline_func=pipeline, package_path=compiled_pipe_file)
    # compiler.Compiler().compile(pipeline_func=pipeline, package_path='ig_post_token_to_pkl.json')

    # Upload compiled file
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(compiled_pipe_file)


def run():
    # Before initializing, make sure to set the GOOGLE_APPLICATION_CREDENTIALS
    # environment variable to the path of your service account.
    aip.init(
        project=project_id,
        location=region,
    )

    # Prepare the pipeline job
    job = aip.PipelineJob(
        display_name=pipeline_name,
        enable_caching=False,
        template_path=pipeline_template_file,
        pipeline_root=pipeline_root,
        job_id=job_id,
        labels=pipeline_labels,
        # parameter_values={
        #     'text': ""
        # }
    )

    job.submit()


if __name__ == '__main__':
    fire.Fire({
        'compile': do_compile,
        'run': run
    })
