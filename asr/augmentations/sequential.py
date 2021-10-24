import random

from typing import List, Callable
from torch import Tensor


class SequentialAugmentation:

    AUGMENTATION_PROB = 0.8
    SPECIFIC_AUGMENTATION_PROB = 0.3

    def __init__(self, augmentation_list: List[Callable]):
        self.augmentation_list = augmentation_list

    def __call__(self, data: Tensor) -> Tensor:
        x = data

        if random.random() < self.AUGMENTATION_PROB: 
            for augmentation in self.augmentation_list:
                if random.random() < self.SPECIFIC_AUGMENTATION_PROB:
                    x = augmentation(x)

        return x