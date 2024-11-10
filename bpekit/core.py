from bpekit.rust import train, encode, encode_dataset

from datasets import Dataset

from pathlib import Path
import os
import re
import pickle
from typing import Tuple, List

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
        return encode(text, self.merges)

    def save_encoded_dataset(
        self,
        dataset: Dataset,
        path: Path,
        shard_size: int,
        batch_size: int
    ):
        """
        Encode and save a corpus (that differs from the tokenizer corpus) 
        to numpy shards.
        """

        def batches(dataset, batch_size):
            total_size = len(dataset)
            for i in range(0, total_size, batch_size):
                yield ''.join(dataset[i:i + batch_size])

        encode_dataset(
            batches(dataset['text'],batch_size),
            self.merges,
            str(path),
            shard_size,
            self.rank
        )

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