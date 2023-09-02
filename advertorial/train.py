import fire
import os
import sys
sys.path.insert(0, os.getcwd())
#os.chdir('../../advertorial-classifier/')
#import sys
#sys.path.insert(0, )

# %%
from advertorial import dataset
from advertorial import utils
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments, Trainer
import wandb
import numpy as np
import evaluate

def train(envfile:str='.env', 
          use_wandb:bool=True ):
    """
    Train the advertorial classifier model by train/valid set stored in BQ and log the metrics in wandb.
    To check the environment variables, please check 

    Args:
        envfile (str, optional): Environment variables art listed in here. Defaults to '.env'.

    Returns:
        _type_: None
    """
    # check or set environment variables
    utils.check_env(envfile)
    today = utils.set_today()
    log_dir = utils.get_based_path('log/')
    prebuilt_dir = utils.get_based_path('prebuilt_model/')
    
    print(f'use_wandb:{use_wandb}')
    wandb.login(key=os.environ['WANDB_KEY'], 
                host=os.environ['WANDB_BASE_URL'])
    wandb.init(
        mode= "online" if use_wandb else "disabled",
        project=os.environ['WANDB_PROJECT'],
        config={'epochs':10}
    )

    advertorial_dataset = dataset.train_valid_test_from_file()

    id2label = {0: "no", 1: "yes"}
    label2id = {"no": 0, "yes": 1}

    pretrain_model ="hfl/chinese-bert-wwm-ext"
    tokenizer = AutoTokenizer.from_pretrained(pretrain_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrain_model, num_labels=2, id2label=id2label, label2id=label2id)

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=-1)
        metric = evaluate.load("accuracy")
        return metric.compute(predictions=predictions, references=labels)


    def preprocess_function(examples):
        return tokenizer(examples["text"], padding=True, truncation=True, max_length=512)

    tokenized_advertorial = advertorial_dataset.map(preprocess_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=log_dir,
        learning_rate=3e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=1,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_advertorial["train"],
        eval_dataset=tokenized_advertorial["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    #check_point = glob.glob('./prebuilt_model/checkpoint-*')[-1]
    #os.system(f'mv {check_point} ./prebuilt_model/{today}-model' )
    print(log_dir)
    trainer.save_model(prebuilt_dir + today)
    trainer.save_model(prebuilt_dir + "cmml_advertorial_post_classifier")


if __name__ == '__main__':
    fire.Fire({'train_milelens_model':train})