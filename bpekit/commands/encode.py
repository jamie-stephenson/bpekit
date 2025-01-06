from bpekit.core import Tokenizer
from bpekit.utils import get_dataset

from pathlib import Path
import argparse
import os

def encode_dataset(
        path: Path, 
        merges_path: Path,
        tokens_path: Path | None = None, 
        shard_size: int = int(1e8),
        batch_size: int = 16,
        ndocs: int | None = None,
        **kwargs
    ):

    """
    Encodes a dataset using a pretrained tokenizer and saves the encoded tokens to disk.

    This function performs the following steps:
    1. Determines the rank and world size for distributed processing from environment variables.
    2. Validates the existence of the tokenizer's merges file.
    3. Checks if the target tokens directory already exists to prevent accidental overwrites.
    4. Loads the dataset, optionally limiting the number of documents to encode.
    5. Initializes the tokenizer using the provided merges file.
    6. Encodes the dataset into tokens, saving the output in shards for efficient storage and retrieval.

    Args:
        path (Path):
            The filesystem path to the dataset that needs to be encoded.
            This can either be the path to a .txt file or the path to the
            directory containing a Hugging Face dataset.
        merges_path (Path):
            The filesystem path to the tokenizer's merges file. This file should be 
            generated during the tokenizer training phase and is essential for initializing 
            the `Tokenizer` instance.
        tokens_path (Path, optional):
            The directory path where the encoded token shards will be saved. If not provided,
            it defaults to a directory named `'tokens/'`. 
            **Note:** If the specified `tokens_path` already exists, a `FileExistsError` 
            will be raised to prevent overwriting existing data.
        shard_size (int, optional):
            The maximum number of tokens per shard. This determines the size of each encoded
            shard file. The default value is `100,000,000` tokens.
        batch_size (int, optional):
            The number of datapoints to concatenate and process in each iteration. 
            The default value is `16`.
        ndocs (int, optional):
            The number of dataset entries to encode. If set to `None`, the function will 
            encode the entire dataset. This parameter is useful for limiting the encoding 
            process to a subset of the dataset, which is useful for testing.

    Raises:
        AssertionError:
            If the `merges_path` does not point to an existing tokenizer merges file, 
            indicating that the tokenizer has not been trained or the path is incorrect.
        FileExistsError:
            If the `tokens_path` directory already exists, preventing accidental 
            overwriting of previously encoded data.

    **MULTIPROCESSING**:
    
    This function can take advantage of an OpenMPI multi-process environment.
    When ran in parallel across multiple processes each with their own
    rank assigned by OpenMPI, this function will automatically splt the dataset
    across processes and encode in parallel. Where possible, each individual process uses
    multithreading to parallelize encoding its part of the dataset. 

    Environment Variables:
        OMPI_COMM_WORLD_RANK (int, optional):
            The rank of the current process in a distributed setup. Defaults to `0` if 
            not set. This is used in multi-process or distributed environments.
        OMPI_COMM_WORLD_SIZE (int, optional):
            The total number of processes in a distributed setup. Defaults to `1` if not set. 
            This determines how the dataset is partitioned and processed across multiple 
            processes.

    Notes:
        - Ensure that the tokenizer's merges file exists at the specified `merges_path` before 
          invoking this function. Without it, the tokenizer cannot be initialized.
        - Adequate disk space should be available at the `tokens_path` to store the encoded shards.

    Example:
        ```python
        from pathlib import Path

        encode_dataset(
            path=Path('/data/raw_dataset'),
            merges_path=Path('/tokenizer/merges.pkl'),
            tokens_path=Path('/data/encoded_tokens'),
            shard_size=100_000_000,
            batch_size=64,
            ndocs=1_000_000
        )
        ```
    """

    rank = int(os.getenv('OMPI_COMM_WORLD_RANK',0))
    world_size = int(os.getenv('OMPI_COMM_WORLD_SIZE',1))

    assert os.path.exists(merges_path),(
        "No tokenizer found at {}. Please train this tokenizer first before attempting to use it."
        .format(merges_path)
    )

    if tokens_path and tokens_path.exists():
        raise FileExistsError(
            f"A directory named `{tokens_path}` already exists. Have you already used `{merges_path}` to encode `{path}`?"
        )
    
    dataset = get_dataset(path,rank,world_size,ndocs)
    tokenizer = Tokenizer.from_pickled_merges(merges_path,rank)
    tokenizer.save_encoded_dataset(dataset,tokens_path,shard_size,batch_size)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--path",
        type=Path,
        help="Path to data set."
    )

    parser.add_argument(
        "--merges_path",
        "--merges-path",
        type=Path,
        help="Path to tokenizer merges."
    )

    parser.add_argument(
        "--tokens_path",
        "--tokens-path",
        type= Path,
        default='tokens/',
        help="Path to save shards to."
    )

    parser.add_argument(
        "--shard_size",
        "--shard-size",
        type= int,
        default=int(1e8),
        help="Number of tokens per shard."
    )

    parser.add_argument(
        "--batch_size",
        "--batch-size",
        type= int,
        default=16,
        help="Number of datapoints to concatenate and process per iteration."
    )

    parser.add_argument(
        "--ndocs",
        type= int,
        default=None,
        help="Number of dataset entries to encode."
    )

    args = parser.parse_args()
    encode_dataset(**vars(args))
