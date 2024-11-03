from tqdm.auto import tqdm
import numpy as np

import os
from typing import Iterable
from pathlib import Path


def save_tokens(
    tokens_iter: Iterable, 
    path: Path, 
    shard_size: int, 
    rank: int
):
    """
    Save tokens from an iterable to shards. 
    `tokens_iter` must be an iterable that yields lists (or numpy arrays) of tokens
    """

    os.makedirs(path, exist_ok=True)
    
    dtype = np.uint16
    split = "train"
    shard_index = 0
    # Preallocate buffer to hold current shard
    all_tokens_np = np.empty((shard_size,), dtype=dtype)
    token_count = 0
    if rank == 0:
        progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")

    for tokens in tokens_iter:
        while token_count + len(tokens) >= shard_size:
            # Write the current shard and start a new one
            filename = os.path.join(path, f"{rank}_{split}_{shard_index:06d}")
            
            # Split the document into whatever fits in this shard; the remainder goes to next one
            remainder = shard_size - token_count
            all_tokens_np[token_count:token_count + remainder] = tokens[:remainder]

            if rank == 0:
                progress_bar.update(remainder)
            
            np.save(filename, all_tokens_np)
            shard_index += 1

            token_count = 0
            tokens = tokens[remainder:]

            if rank == 0:
                progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")

        # simply append tokens to current shard
        all_tokens_np[token_count:token_count+len(tokens)] = tokens
        token_count += len(tokens)
        
        if rank == 0:
            progress_bar.update(len(tokens))

    if token_count != 0:
        split = "train" if shard_index == 0 else "val"
        filename = os.path.join(path, f"{rank}_{split}_{shard_index:06d}")
        np.save(filename, all_tokens_np[:token_count])