from utils import Tokenizer, get_dataset

from pathlib import Path
import argparse
import os

def encode(
        path: Path, 
        merges_path: Path,
        tokens_path: Path | None = None, 
        shard_size: int = int(1e8),
        ndocs: int | None = None
    ):

    rank, world_size = int(os.getenv('OMPI_COMM_WORLD_RANK',0)), int(os.getenv('OMPI_COMM_WORLD_SIZE',1))

    assert os.path.exists(merges_path),(
        "No tokenizer found at {}. Please train this tokenizer first before attempting to use it."
        .format(merges_path)
    )

    assert not os.path.exists(tokens_path),(
        "A directory named `{}` already exists. Have you already used `{}` to encode `{}`?."
        .format(tokens_path,merges_path,path)
    )
    
    dataset = get_dataset(path,rank,world_size,ndocs)
    tokenizer = Tokenizer.from_pickled_merges(merges_path,rank,world_size)
    tokenizer.save_encoded_corpus(dataset,tokens_path,shard_size)

if __name__ == '__main__':
    """Trains and saves new tokenizer based on command line input."""
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
        type= int,
        default=int(1e8),
        help="Number of tokens per shard."
    )

    parser.add_argument(
        "--ndocs",
        type= int,
        default=None,
        help="Number of dataset entries to encode."
    )

    args = parser.parse_args()
    encode(**vars(args))
