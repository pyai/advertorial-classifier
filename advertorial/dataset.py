import os
from datasets import DatasetDict
from datasets import Dataset, DatasetDict
from google.cloud import bigquery
from advertorial import utils

def train_valid_test_from_file(envfile='.env') ->DatasetDict:
    '''
    correct usages:
    >>> dataset.train_valid_test_from_file(train_ratio=0.8, validation_ratio=0.1)
    DatasetDict({
        train: Dataset({
            features: ['post_text', 'label'],
            num_rows: 3680
        })
        validation: Dataset({
            features: ['post_text', 'label'],
            num_rows: 460
        })
        test: Dataset({
            features: ['post_text', 'label'],
            num_rows: 460
        })
    })

    >>> dataset.train_valid_test_from_file(train_ratio=0.8, validation_ratio=0)
    DatasetDict({
        train: Dataset({
            features: ['text', 'label'],
            num_rows: 3680
        })
        test: Dataset({
            features: ['text', 'label'],
            num_rows: 920
        })
    })

    >>> dataset.train_valid_test_from_file(train_ratio=1)
    DatasetDict({
        train: Dataset({
            features: ['text', 'label'],
            num_rows: 4600
        })
    })

    wrong usages:
    >>> dataset.train_valid_test_from_file(train_ratio=0, validation_ratio=1)
    Wrong train ratio:0.00 should be > 0
    DatasetDict({
        train: Dataset({
            features: ['text', 'label'],
            num_rows: 4600
        })
    })

    >>> dataset.train_valid_test_from_file(train_ratio=0.8, validation_ratio=0.8)
    Wrong validation ratio:4.00 should be < 1
    DatasetDict({
        train: Dataset({
            features: ['text', 'label'],
            num_rows: 3680
        })
        validation: Dataset({
            features: ['text', 'label'],
            num_rows: 920
        })
    })    
    '''

    # set environment variables
    utils.check_env(envfile)


    # get bq settings from environment variable
    project_id, dataset_id = os.environ['GCP_PROJECT'], os.environ['GCP_BQ_DATASET']
    train_table_id, test_table_id = os.environ['GCP_BQ_TRAIN_TABLE'], os.environ['GCP_BQ_TEST_TABLE']
    region=os.environ['GCP_REGION']
    
    # Load BQ table into dataframe
    client = bigquery.Client(project=project_id,
                             location=region)
    sql = f"SELECT post_text AS text, cate AS label FROM `{project_id}.{dataset_id}.{train_table_id}`"
    train = client.query(sql).to_dataframe().astype({'label': 'int16'})
    print(sql)

    sql = f"SELECT post_text AS text, cate AS label FROM `{project_id}.{dataset_id}.{test_table_id}`"
    test = client.query(sql).to_dataframe().astype({'label': 'int16'})
    print(sql)

    # Convert the pandas DataFrame into a Hugging Face Dataset
    train_dataset = Dataset.from_pandas(train)
    test_dataset = Dataset.from_pandas(test)
    dataset_dict = DatasetDict({'train': train_dataset, 'test': test_dataset})
        
    return dataset_dict




