from rustbpe import train, encode

from datasets import Dataset
import numpy as np
from tqdm.auto import tqdm

import multiprocessing as mp
import os
import re
import pickle
from typing import Tuple, List
from time import time

class Tokenizer:
    """
    Class for training and using tokenizers that is (almost) distribution agnositic.
    """
    def __init__(self, merges, rank, world_size) -> None:
        self.rank = rank
        self.world_size = world_size
        self.merges = merges

    @classmethod
    def from_pickled_merges(cls, path, rank=0, world_size=1):
        return cls(cls.load_merges(path), rank, world_size)

    @classmethod
    def from_dataset(
        cls, 
        dataset: Dataset, 
        vocab_size: int, 
        rank: int, 
        world_size: int, 
        pattern=r'\s?\w+|\s?[^a-zA-Z0-9\s]+|\s+(?=\s)'
    ):
        """
        Trains new tokenizer from a dataset. When using distributed training, `dataset` 
        should be the chunk of the dataset that the current rank will handle.
        """
        print(f"Rank {rank} ready to train.")

        compiled_pattern = re.compile(pattern)

        blocks_str = re.findall(compiled_pattern, '\n'.join(dataset['text']))
        blocks_utf8 = [block_str.encode('utf-8') for block_str in blocks_str]

        merges = train(blocks_utf8,vocab_size) 

        tokenizer  = cls(merges, rank, world_size)

        return tokenizer


    #----------------------ENCODING-METHODS------------------------

    def encode(self, text: str) -> List[int]:
        return encode(text.encode('utf-8'), self.merges)

    def save_encoded_corpus(self, dataset, path, shard_size):
        """
        Encode and save a corpus (that differs from the tokenizer corpus) 
        to shards.
        """

        if self.rank==0:
            t0 = time()

        with mp.Pool(os.cpu_count()) as pool:
            tokens_iter = pool.imap(self.encode, dataset['text'], chunksize=16)
            self._save_tokens(tokens_iter,path,shard_size) 

        if self.rank==0:
            print(f"Encoding and saving took {time()-t0:.2f} seconds.")

    def save_encoded_tokenizer_corpus(self, path, shard_size):
        """
        Save encoded tokenizer corpus as shards.
        Must be called after `__train`.
        """
        self._save_tokens(self.blocks,path,shard_size)
    
    #------------------END-OF-ENCODING-METHODS--------------------

    #----------------------DECODING-METHODS-----------------------
        
    def decode(self, tokens: list) -> str:
        for bytepair,merged_byte in reversed(self.merges):
            tokens = self._unmerge_byte(tokens,merged_byte,bytepair)
        return bytes(tokens).decode('utf-8',errors='replace')
    
    @staticmethod
    def _unmerge_byte(lst: List[int], merged_byte: int, bytepair: Tuple[int,int]) -> List[int]:
        new_lst = []
        for i in range(len(lst)):
            if lst[i] == merged_byte:
                new_lst += list(bytepair)
            else:
                new_lst.append(lst[i])
        return new_lst
    
    def decoded_tokens(self):
        """lists all merged token chunks as plain text"""
        for _,token in self.merges:
            print(f"Token {token} decodes to {self.decode([token])}")

    #------------------END-OF-DECODING-METHODS--------------------

    #-------------------SAVING/LOADING-METHODS--------------------

    def _save_tokens(self, tokens_iter, path, shard_size):
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
        if self.rank == 0:
            progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")

        for tokens in tokens_iter:
            while token_count + len(tokens) >= shard_size:
                # Write the current shard and start a new one
                filename = os.path.join(path, f"{self.rank}_{split}_{shard_index:06d}")
                
                # Split the document into whatever fits in this shard; the remainder goes to next one
                remainder = shard_size - token_count
                all_tokens_np[token_count:token_count + remainder] = tokens[:remainder]

                if self.rank == 0:
                    progress_bar.update(remainder)
                
                np.save(filename, all_tokens_np)
                shard_index += 1

                token_count = 0
                tokens = tokens[remainder:]

                if self.rank == 0:
                    progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")

            # simply append tokens to current shard
            all_tokens_np[token_count:token_count+len(tokens)] = tokens
            token_count += len(tokens)
            
            if self.rank == 0:
                progress_bar.update(len(tokens))

        if token_count != 0:
            split = "train" if shard_index == 0 else "val"
            filename = os.path.join(path, f"{self.rank}_{split}_{shard_index:06d}")
            np.save(filename, all_tokens_np[:token_count])

    def save_merges(self, path):
        if self.rank == 0:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'wb') as file:
                pickle.dump(self.merges, file)

    @staticmethod
    def load_merges(path):
        with open(path, 'rb') as file:
            merges = pickle.load(file)
        return merges

    @staticmethod
    def load_corpus(path):
        with open(path, 'r') as file:
            file_contents = file.read()
        return file_contents
    
    #-----------------END-OF-SAVING/LOADING-METHODS------------------