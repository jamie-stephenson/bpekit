"""Trains and saves new tokenizer"""
from utils import Tokenizer, get_dataset

import argparse
from pathlib import Path
import os

def train_tokenizer(
        path: Path, 
        vocab_size: int, 
        merges_path: Path = Path('tokenizers/'),
        tokens_path: Path | None = None, 
        shard_size: int = int(1e8),
        ndocs: int | None = None
    ) -> Tokenizer:

    rank, world_size = int(os.getenv('OMPI_COMM_WORLD_RANK',0)), int(os.getenv('OMPI_COMM_WORLD_SIZE',1))
    
    merges_path = merges_path / Path(f"{vocab_size}.pkl")

    assert not os.path.exists(merges_path),(
        "A tokenizer already exists at {}. Have you trained this tokenizer already?"
        .format(merges_path)
    )
    
    dataset = get_dataset(path,rank,world_size,ndocs)

    tokenizer = Tokenizer.from_dataset(dataset,vocab_size,rank,world_size)
    tokenizer.save_merges(merges_path)

    if args.tokens_path:
        tokenizer.save_encoded_tokenizer_corpus(tokens_path,shard_size)

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
        "--tokens_path",
        "--tokens-path",
        type= Path,
        default=None,
        help="Path to save encoded tokenizer corpus shards to."
    )
    
    parser.add_argument(
        "--shard_size",
        type= int,
        default=int(1e8),
        help="Number of tokens per shard (only used when saving encoded tokenizer corpus)."
    )
    
    parser.add_argument(
        "--ndocs",
        type= int,
        default=None,
        help="Number of dataset entries to encode."
    )

    args = parser.parse_args()
    train_tokenizer(**vars(args))