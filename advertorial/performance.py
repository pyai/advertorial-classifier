# %%
import fire
import os
import sys
sys.path.insert(0, os.getcwd())
# %%
from advertorial import dataset
import numpy as np
import pandas as pd
from advertorial.inference import AdvertorialModel
from tqdm import tqdm
import json
from google.cloud import bigquery
from advertorial import utils

from typing import Optional


def perf_report(model, dataset, name='train'):
    N = len(dataset)
    step = 20
    ones = 0
    zeros = 0
    hits = 0
    miss = 0
    fp = 0
    fn = 0
    tp = 0
    tn = 0
    predictions = []
    for s in tqdm(range(0, N, step)):
        s, e = s, s+step
        prediction, probs = model(dataset[s:e]['text'])
        
        hits += np.sum(dataset[s:e]['label'] == prediction)
        miss += np.sum(dataset[s:e]['label'] != prediction)
        zeros += np.sum(dataset[s:e]['label'] == np.array(0))
        ones += np.sum(dataset[s:e]['label'] == np.array(1))
        fp += np.sum((dataset[s:e]['label'] != prediction) & (dataset[s:e]['label'] == np.array(0)))
        fn += np.sum((dataset[s:e]['label'] != prediction) & (dataset[s:e]['label'] == np.array(1)))
        tp += np.sum((dataset[s:e]['label'] == prediction) & (dataset[s:e]['label'] == np.array(1)))
        tn += np.sum((dataset[s:e]['label'] == prediction) & (dataset[s:e]['label'] == np.array(0)))
        predictions.append(prediction) 

    accuracy = hits/N
    print(f'accuracy:{accuracy:.2f}, positive samples:{ones}, negative samples:{zeros}')  
    performance_df = pd.DataFrame({'dataset':[name], 
                                   'records':[N], 
                                   'positive samples':[ones], 
                                   'negative samples':[zeros], 
                                   'hit':[hits],
                                   'miss':[miss],
                                   'accuracy':[accuracy], 
                                   'miss rate':[1-accuracy], 
                                   'false pos rate':[fp/(tn+fp)], 
                                   'false neg rate':[fn/(tp+fn)],
                                   'true pos rate (sensitivity)':[tp/(tp+fn)],
                                   'true neg rate (specificity)':[tn/(tn+fp)],
                                   'precision':[tp/(tp+fp)]})

    predictions = np.concatenate(predictions)
    error_ids = predictions != dataset['label']
    error_df = pd.DataFrame({'text':np.array(dataset['text'])[error_ids], 'label':np.array(dataset['label'])[error_ids], 'prediction':predictions[error_ids]})
    return error_df, performance_df


def summary(envfile:str ='.env', model_folder:Optional[str]=None):
    advertorial_dataset = dataset.train_valid_test_from_file()
    train = advertorial_dataset['train']
    test = advertorial_dataset['test']

    today = utils.set_today(today_str=model_folder)
    log_dir = utils.get_based_path('prebuilt_model/')

    #model_name =  glob.glob('./prebuilt_model/*-model')[-1]
    adv = AdvertorialModel(model_path=log_dir + "cmml_advertorial_post_classifier", use_gpu=True)

    train_error, train_perf = perf_report(adv, train, 'train')
    test_error, test_perf = perf_report(adv, test, 'test')
    performance = pd.concat([train_perf, test_perf]).reset_index(drop=True).T

    performance[0].to_dict()
    performance[1].to_dict()

    train_json_str = json.dumps(performance[0].to_dict())
    test_json_str = json.dumps(performance[1].to_dict())

    # check or set environment variables
    utils.check_env(envfile)
    today = utils.get_today()
    year, month, day = today[0:4], today[4:6], today[6:8]

    # get bq settings from environment variable
    project_id, dataset_id = os.environ['GCP_PROJECT'], os.environ['GCP_BQ_DATASET']
    region=os.environ['GCP_REGION']
    meta_table_id = os.environ['GCP_BQ_META_TABLE']
    model_uri_base = os.environ['GCS_MODEL_URI_BASE']
    table = f'`{project_id}.{dataset_id}.{meta_table_id}`'
    model_name = f'{today}-model'
    model_uri = f'{model_uri_base}{model_name}/'
    performance = json.dumps({'train_acc':train_perf['accuracy'].item(), 
                              'test_acc':test_perf['accuracy'].item()})
    
    # Load BQ table into dataframe
    client = bigquery.Client(project=project_id, 
                             location=region)
    sql = f"DELETE FROM {table} WHERE `date`=DATE({year},{month},{day})"
    client.query(sql)
    utils.logger.info(sql)
    print(sql)

    sql = f"INSERT INTO {table} (model_name, train_set_meta, test_set_meta, `date`, \
            model_uri, model_performance) VALUES ('{model_name}', '{train_json_str}', \
            '{test_json_str}', DATE({year},{month},{day}), '{model_uri}', '{performance}');"
    client.query(sql)
    utils.logger.info(sql)
    print(sql)

if __name__ == '__main__':
    fire.Fire({'summary_milelens_model':summary})




 


