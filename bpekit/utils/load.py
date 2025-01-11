from datasets import load_from_disk, load_dataset, Dataset, DatasetDict
from datasets.distributed import split_dataset_by_node

from pathlib import Path
import os


def get_dataset(
    path: Path, rank: int, world_size: int, ndocs: int | None = None
) -> Dataset:
    if path.exists():
        if path.suffix == ".txt":
            # Load the text file into a DatasetDict (each line of the .txt will be an entry)
            dataset = load_dataset("text", data_files=str(path))

        elif path.is_dir():
            txt_file = find_txt_file(path)

            if txt_file:
                dataset = load_dataset("text", data_files=str(txt_file))
            
            # Otherwise we assume directory contains a HF dataset
            else:
                dataset = load_from_disk(str(path))

        else:
            raise ValueError(f"Unsupported file type: {path}")
        
    else:
        raise FileNotFoundError(f"Cannot find {path}")

    # If the dataset is a DatasetDict (has multiple splits), return the 'train' split or the first available split
    if isinstance(dataset, DatasetDict):
        if "train" in dataset:
            dataset = dataset["train"]
        else:
            # Return the first available split if 'train' is not present
            dataset = next(iter(dataset.values()))

    # If specified, only take the first `ndocs` entries from the dataset.
    if ndocs:
        dataset = dataset.select(range(ndocs))

    try:
        return split_dataset_by_node(dataset, rank, world_size)
    except IndexError as e:
        raise IndexError(
            "Problem encountered when attempting to split dataset across nodes. "
            "This could be due to the dataset not being structured appropriately "
            "(e.g., dataset has length 1, with all data in one entry)."
        ) from e


def find_txt_file(directory_path: Path) -> Path | None:
    """
    Checks if a directory contains a .txt file.

    Args:
        directory_path (str): Path to the directory.

    Returns:
        str: Path to the first .txt file found, or None if no .txt file exists.
    """
    if not directory_path.is_dir():
        raise FileNotFoundError(f"The directory {directory_path} does not exist.")
    
    for file in directory_path.iterdir():
        if file.suffix==".txt":
            return directory_path / file.name
    
    # No .txt file found
    return None