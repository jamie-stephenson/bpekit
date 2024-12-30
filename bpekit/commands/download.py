from datasets import load_dataset

import argparse
import os
from pathlib import Path

def download_dataset(
        hf_path: str,
        path: Path,
        cache_dir: Path | None = None,
        name: str | None = None,
        split: str | None = 'train',
        n_proc: int | None = os.cpu_count(),
        **kwargs
    ):
    """
    Download a Hugging Face dataset, save it to disk, and remove the cached files.

    Args:
        hf_path (str): Path to the dataset on the Hugging Face Hub.
        path (str): Path to save the downloaded dataset.
        cache_dir (str): Directory for Hugging Face to use for caching the dataset.
            Defaults to ~/.cache/huggingface
        name (Optional[str]): Optional name of a specific part of the dataset. Defaults to None.
        split (Optional[str]): Dataset split to download. Defaults to 'train'.
        num_proc (Optional[int]): Number of processes to use. Defaults to the number of CPU cores available.
    """  
    
    dataset = load_dataset(
        path=hf_path,
        name=name,
        split=split,
        cache_dir=cache_dir,
        num_proc=n_proc
    )
    
    dataset.save_to_disk(str(path))
    dataset.cleanup_cache_files()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        help="Name of dataset (as found in the `configs/project_datasets/` directory)."
    )
        
    parser.add_argument(
        "--hf_path",
        "--hf-path",
        type=str,
        help="Path to dataset of Hugging Face hub."
    )
        
    parser.add_argument(
        "--path",
        type=str,
        help="path to save dataset to"
    )
        
    parser.add_argument(
        "--cache_dir",
        "--cache-dir",
        type=str,
        help="Directory for Hugging Face to use to cache dataset."
    )
        
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Optional name of specific part of the dataset." 
    )
        
    parser.add_argument(
        "--split",
        type=str,
        default='train'
    )
        
    parser.add_argument(
        "--n_proc",
        type=int,
        default=os.cpu_count()
    )

    args = parser.parse_args()

    download_dataset(**vars(args))