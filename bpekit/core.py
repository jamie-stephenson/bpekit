from bpekit.rust import train, encode
from bpekit.utils import save_tokens

from datasets import Dataset

import multiprocessing as mp
from pathlib import Path
import os
import re
import pickle
from typing import Tuple, List
from time import time

class Tokenizer:
    """
    Class for training and using tokenizers that is (almost) distribution agnositic.
    """
    def __init__(
        self, 
        merges: List[Tuple[Tuple[int,int],int]], 
        rank: int, 
    ) -> None:

        self.rank = rank
        self.merges = merges

    @classmethod
    def from_pickled_merges(cls, path, rank=0):
        return cls(cls.load_merges(path), rank)

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
        if world_size > 1:
            print(f"Rank {rank} ready to train.")

        compiled_pattern = re.compile(pattern)

        blocks_str = (
            block 
            for doc in dataset['text']
            for block in re.findall(compiled_pattern,doc)
        )

        merges = train(blocks_str,vocab_size) 

        tokenizer  = cls(merges, rank)

        return tokenizer


    #----------------------ENCODING-METHODS------------------------

    def encode(self, text: str) -> List[int]:
        return encode(text.encode('utf-8'), self.merges)

    def save_encoded_corpus(
        self,
        dataset: Dataset,
        path: Path,
        shard_size: int
    ):
        """
        Encode and save a corpus (that differs from the tokenizer corpus) 
        to shards.
        """

        if self.rank==0:
            t0 = time()

        with mp.Pool(os.cpu_count()) as pool:
            tokens_iter = pool.imap(self.encode, dataset['text'], chunksize=16)
            save_tokens(tokens_iter,path,shard_size,self.rank) 

        if self.rank==0:
            print(f"Encoding and saving took {time()-t0:.2f} seconds.")

    def save_encoded_tokenizer_corpus(
        self,
        path: Path,
        shard_size: int
    ):
        """
        Save encoded tokenizer corpus as shards.
        Must be called after `from_dataset`.
        """
        save_tokens(self.blocks,path,shard_size,self.rank)
    
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

    def save_merges(self, path: Path):
        if self.rank == 0:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'wb') as file:
                pickle.dump(self.merges, file)

    @staticmethod
    def load_merges(path: Path):
        with open(path, 'rb') as file:
            merges = pickle.load(file)
        return merges

    @staticmethod
    def load_corpus(path: Path):
        with open(path, 'r') as file:
            file_contents = file.read()
        return file_contents
    
    #-----------------END-OF-SAVING/LOADING-METHODS------------------