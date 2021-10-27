import re
from typing import List, Union

import json
from pathlib import Path
from string import ascii_lowercase

import numpy as np
from torch import Tensor


def get_simple_alphabet():
        return list(ascii_lowercase + ' ')

class TextEncoder:
    
    def __init__(self, alphabet: List[str]):
        self.index_to_char = {k: v for k, v in enumerate(sorted(alphabet))}
        self.char_to_index = {v: k for k, v in self.index_to_char.items()}

    def __len__(self):
        return len(self.index_to_char)

    def __getitem__(self, item: int):
        assert type(item) is int
        return self.index_to_char[item]
    
    def encode(self, text) -> Tensor:
        text = self.normalize_text(text)
        try:
            return Tensor([self.char_to_index[char] for char in text]).unsqueeze(0)
        except KeyError:
            unknown_chars = set([char for char in text if char not in self.char_to_index])
            raise Exception(
                f"Can't encode text '{text}'. Unknown chars: '{' '.join(unknown_chars)}'")

    def decode(self, vector: Union[Tensor, np.ndarray, List[int]]):
        return ''.join([self.index_to_char[int(ind)] for ind in vector]).strip()

    def dump(self, file):
        with Path(file).open('w') as f:
            json.dump(self.index_to_char, f)

    @classmethod
    def from_file(cls, file):
        with Path(file).open() as f:
            index_to_char = json.load(f)
        a = cls([])
        a.index_to_char = index_to_char
        a.char_to_index = {v: k for k, v in index_to_char}
        return a

    @staticmethod
    def normalize_text(text: str):
        text = str(text).lower() #TODO: Надо обрабатывать числа
        text = re.sub(r"[^a-z ]", "", text)
        return text
