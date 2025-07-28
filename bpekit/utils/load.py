from datasets import load_from_disk, load_dataset, Dataset, DatasetDict
from datasets.distributed import split_dataset_by_node

from pathlib import Path


def get_dataset(
    path: Path, rank: int, world_size: int, ndocs: int | None = None
) -> Dataset:
    if not path.exists():
        raise FileNotFoundError(path)

    if path.is_file():
        if path.suffix != ".txt":
            raise ValueError(f"Unsupported file type: {path.suffix}")
        ds = load_dataset("text", data_files=str(path))
    else:
        txt_file = next(path.glob("*.txt"), None)
        if txt_file:
            ds = load_dataset("text", data_files=str(txt_file))
        else: # Assume dataset is HF Dataset
            ds = load_from_disk(str(path))

    # collapse DatasetDict → Dataset
    if isinstance(ds, DatasetDict):
        ds = ds.get("train") or next(iter(ds.values()))

    # If specified, only take the first `ndocs` entries from the dataset.
    if ndocs:
        ds = ds.select(range(ndocs))

    try:
        return split_dataset_by_node(ds, rank, world_size)
    except IndexError as e:
        raise IndexError(
            "Problem encountered when attempting to split dataset across nodes. "
            "This could be due to the dataset not being structured appropriately "
            "(e.g., dataset has length 1, with all data in one entry)."
        ) from e

