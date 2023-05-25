import os
os.chdir('../../advertorial-classifier/')

# %%
from advertorial import dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments, Trainer
import wandb
import numpy as np
import evaluate

# %%
# advertorial_dataset = dataset.train_valid_test_from_file(csv_file_path= './data/milelens_advertorial_dataset_formatted.csv')
# train, validation, test = advertorial_dataset['train'], advertorial_dataset['validation'], advertorial_dataset['test'] 
# id2label = {0: "no", 1: "yes"}
# label2id = {"no": 0, "yes": 1}

# pretrain_model ="hfl/chinese-bert-wwm-ext"
# tokenizer = AutoTokenizer.from_pretrained(pretrain_model)
# model = AutoModelForSequenceClassification.from_pretrained(
#     pretrain_model, num_labels=2, id2label=id2label, label2id=label2id)


advertorial_dataset = dataset.train_valid_test_from_file(csv_file_path= './data/milelens_advertorial_dataset_formatted.csv', train_ratio=1)
train = advertorial_dataset['train']
id2label = {0: "no", 1: "yes"}
label2id = {"no": 0, "yes": 1}

pretrain_model ="hfl/chinese-bert-wwm-ext"
tokenizer = AutoTokenizer.from_pretrained(pretrain_model)
model = AutoModelForSequenceClassification.from_pretrained(
    pretrain_model, num_labels=2, id2label=id2label, label2id=label2id)

# %%
training_args = TrainingArguments(
    #logging_steps=10000,
    #save_steps=10000,
    output_dir="prebuilt_model/log",
    learning_rate=3e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    weight_decay=0.01,
    #evaluation_strategy="steps"
    evaluation_strategy="epoch",
    save_strategy="epoch",
    #fp16=True,
    #load_best_model_at_end=True,
    #push_to_hub=True,
)

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_advertorial["train"],
#     eval_dataset=tokenized_advertorial["validation"],
#     tokenizer=tokenizer,
#     data_collator=data_collator,
#     compute_metrics=compute_metrics,
# )

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_advertorial["train"],
    eval_dataset=tokenized_advertorial["train"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()


