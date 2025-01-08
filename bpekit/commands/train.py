from bpekit.core import Tokenizer
from bpekit.utils import get_dataset

import argparse
from pathlib import Path
import os

def train_tokenizer(
        path: Path, 
        vocab_size: int, 
        merges_path: Path = Path('tokenizers/tokenizer.pkl'),
        ndocs: int | None = None,
        rank: int = 0,
        world_size: int = 1,
        **kwargs
    ) -> Tokenizer:
    """
    Trains a tokenizer and saves the merges.

    Args:
        path (Path): Path to the dataset. This can either be the path to a .txt file or the path to the
            directory containing a Hugging Face dataset.
        vocab_size (int): Vocabulary size.
        merges_path (Path): Path to save merges to.
        ndocs (Optional[int]): Number of dataset entries to train with.

    Returns:
        tokenizer (Tokenizer): The trained tokenizer.

    **MULTIPROCESSING WARNING**:

    If you run this function in parallel across multiple processes, the following will occur:
    - The first part of the training process will take advantage of multiprocessing
    - The second part uses multithreading on a single process and will kill
      any non-root processes to forcefully free up resources for the root.

    Bare this in mind when attempting to use this function in series with other
    functions that rely on a multiprocessing environment. 
    """

    assert not os.path.exists(merges_path),(
        "A tokenizer already exists at {}. Have you trained this tokenizer already?"
        .format(merges_path)
    )
    
    dataset = get_dataset(path,rank,world_size,ndocs)

    tokenizer = Tokenizer.from_dataset(dataset,vocab_size,rank,world_size)
    tokenizer.save_merges(merges_path)

    return tokenizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--vocab_size",
        type=int
    )

    parser.add_argument(
        "--path",
        type=Path,
        help="Path to data set."
    )

    parser.add_argument(
        "--merges_path",
        "--merges-path",
        type= Path,
        default='tokenizers/',
        help="Path to save merges to."
    )

    parser.add_argument(
        "--ndocs",
        type= int,
        default=None,
        help="Number of dataset entries to train with."
    )

    args = parser.parse_args()
    train_tokenizer(**vars(args))