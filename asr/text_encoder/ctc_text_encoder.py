from typing import List, Tuple
from collections import defaultdict

import torch

from asr.text_encoder.text_encoder import TextEncoder


class CTCTextEncoder(TextEncoder):

    EMPTY_TOKEN = '^'
    EMPTY_INDEX = 0

    def __init__(self, alphabet: List[str]):
        super().__init__(alphabet)

        self.index_to_char = {self.EMPTY_INDEX: self.EMPTY_TOKEN}

        for token in alphabet:
            self.index_to_char[max(self.index_to_char.keys()) + 1] = token

        self.char_to_index = {value: key for key, value in self.index_to_char.items()}

    def ctc_decode(self, indexes: List[int]) -> str:
        selected_indexes = []
        is_empty_token = False

        for index in indexes:
            index = int(index)

            if index == self.EMPTY_INDEX:
                is_empty_token = True
            
            else:
                if len(selected_indexes) == 0 or selected_indexes[-1] != index or is_empty_token:
                    selected_indexes.append(index)
                    is_empty_token = False
        
        return ''.join([self.index_to_char[index] for index in selected_indexes])

    def extend_paths(self, next_char_probs, src_paths):
        new_paths = defaultdict(float)

        for next_char_index, next_char_prob in enumerate(next_char_probs):
            next_char = self.index_to_char[int(next_char_index)]

            for (text, last_char), path_prob in src_paths.items():
                new_text = text if next_char == last_char else (text + next_char)
                new_text = new_text.replace(self.EMPTY_TOKEN, '')
                new_paths[(new_text, next_char)] += path_prob * next_char_prob

        return new_paths

    def select_best_paths(self, paths, beam_size):
        return dict(sorted(paths.items(), key=lambda x: x[1])[-beam_size:])

    def ctc_beam_search(self, probs: torch.tensor, beam_size: int = 100) -> List[Tuple[str, float]]:
            """
            Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
            """
            assert len(probs.shape) == 2
            char_length, voc_size = probs.shape
            assert voc_size == len(self.index_to_char)

            paths = {('', self.EMPTY_TOKEN) : 1.0}

            for next_char_probs in probs:
                paths = self.extend_paths(next_char_probs, paths)
                paths = self.select_best_paths(paths, beam_size)
            
            return [(hypothesis, hypothesis_prob) 
                    for (hypothesis, _), hypothesis_prob 
                    in sorted(paths.items(), key=lambda x: -x[1])]
