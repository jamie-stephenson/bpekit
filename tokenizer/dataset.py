from datasets import load_from_disk, load_dataset, Dataset, DatasetDict
from datasets.distributed import split_dataset_by_node

from pathlib import Path
import os

def get_dataset(path: Path, rank:int, world_size:int) -> Dataset:

    if path.exists():
        if path.suffix=='.txt':
            # Load the text file into a DatasetDict
            dataset = load_dataset('text', data_files=str(path))
        else:
            raise ValueError(f"Unsupported file type: {path}")
    elif path.is_dir():
        # Load the dataset saved to disk
        dataset = load_from_disk(path)
    else:
        # Assume it's a dataset on the Hugging Face Hub
        dataset = load_dataset(str(path))

    # If the dataset is a DatasetDict (has multiple splits), return the 'train' split or the first available split
    if isinstance(dataset, DatasetDict):
        if 'train' in dataset:
            dataset = dataset['train']
        else:
            # Return the first available split if 'train' is not present
            dataset = next(iter(dataset.values()))

    return split_dataset_by_node(dataset,rank,world_size)

