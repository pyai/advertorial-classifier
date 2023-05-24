from datasets import load_dataset, DatasetDict
from typing import Any, Union


import pandas as pd
from datasets import Dataset, DatasetDict

def train_valid_test_from_file(csv_file_path:str= './data/milelens_advertorial_dataset_formatted.csv', train_ratio:Union[int, float]=0.8, validation_ratio:Union[int, float]=0.1) ->DatasetDict:
    '''
    correct usages:
    >>> dataset.train_valid_test_from_file(train_ratio=0.8, validation_ratio=0.1)
    DatasetDict({
        train: Dataset({
            features: ['text', 'label'],
            num_rows: 3680
        })
        validation: Dataset({
            features: ['text', 'label'],
            num_rows: 460
        })
        test: Dataset({
            features: ['text', 'label'],
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
    test_size = 1-train_ratio
    validation_size = 0 if test_size <= 0 else validation_ratio/test_size

    # Load the CSV file using pandas
    data = pd.read_csv(csv_file_path)

    # Convert the pandas DataFrame into a Hugging Face Dataset
    dataset = Dataset.from_pandas(data)

    if 0 < test_size < 1 :
        # Split the dataset into train, validation, and test splits
        dataset = dataset.train_test_split(test_size=test_size)  # Adjust the test_size as needed

        # Assign the resulting splits to their respective variables
        train_dataset = dataset['train']
        test_dataset = dataset['test']

        if 0 < validation_size < 1:
            # Further split the train_dataset into train and validation splits
            test_dataset = test_dataset.train_test_split(test_size=1-validation_size)  # Adjust the test_size as needed

            # Assign the resulting splits to their respective variables
            validation_dataset = test_dataset['train']
            test_dataset = test_dataset['test']

            # Create a DatasetDict to store the splits
            dataset_dict = DatasetDict({'train': train_dataset, 'validation': validation_dataset, 'test': test_dataset})
        elif validation_size >= 1:
            print(f'Wrong validation ratio:{validation_size:.2f} should be < 1')
            dataset_dict = DatasetDict({'train': train_dataset, 'validation': test_dataset})
        else:
            dataset_dict = DatasetDict({'train': train_dataset, 'test': test_dataset})

    else:
        if test_size == 1:
            print(f'Wrong train ratio:{train_ratio:.2f} should be > 0')
        dataset_dict = DatasetDict({'train': dataset})
        
    return dataset_dict


