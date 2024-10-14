from datasets import load_from_disk, load_dataset, Dataset, DatasetDict
from datasets.distributed import split_dataset_by_node

from pathlib import Path

def get_dataset(path: Path, rank:int, world_size:int) -> Dataset:

    if path.exists():
        if path.suffix=='.txt':
            # Load the text file into a DatasetDict
            dataset = load_dataset('text', data_files=str(path))
        elif path.is_dir():
            dataset = load_from_disk(path)
        else:
            raise ValueError(f"Unsupported file type: {path}")
    else:
        raise FileNotFoundError(f"Cannot find {path}")         

    # If the dataset is a DatasetDict (has multiple splits), return the 'train' split or the first available split
    if isinstance(dataset, DatasetDict):
        if 'train' in dataset:
            dataset = dataset['train']
        else:
            # Return the first available split if 'train' is not present
            dataset = next(iter(dataset.values()))
    try:
        return split_dataset_by_node(dataset, rank, world_size)
    except IndexError as e:
        raise IndexError(
            "Problem encountered when attempting to split dataset across nodes. "
            "This could be due to the dataset not being structured appropriately "
            "(e.g., dataset has length 1, with all data in one entry)."
        ) from e


