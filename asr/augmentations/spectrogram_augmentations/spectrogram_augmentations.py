import numpy as np
import random

import torchaudio
from torch import nn
from torch import Tensor

class SpectAugmentation():

    def __init__(self, filling_value = 'mean', 
                 n_freq_masks=3, n_time_masks=3,
                 max_freq=30, max_time=40):

        self.filling_value = filling_value
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks
        self.max_freq = max_freq
        self.max_time = max_time

        if self.filling_value == 'const':
            self.value = 0  # or something else

    def __call__(self, spectrogram: Tensor):

        augmented_spectrogram = spectrogram.clone()
  
        if self.filling_value == 'mean':
            value = spectrogram.mean()

        elif self.filling_value == 'min':
            value = spectrogram.min()

        elif self.filling_value == 'max':
            value = spectrogram.max()

        elif self.filling_value == 'const':
            value = self.value

        else:
            ValueError("You can choose `mean`, `min`, `max` or `const` methods only.")

        for _ in range(self.n_freq_masks):
            frqency_1 = np.random.randint(0, augmented_spectrogram.shape[0])
            frqency_2 = frqency_1 + random.randrange(0, self.max_freq)
            augmented_spectrogram[frqency_1:frqency_2, :].fill_(value)
        
        for _ in range(self.n_time_masks):
            time_1 = np.random.randint(0, augmented_spectrogram.shape[1])
            time_2 = time_1 + random.randrange(0, self.max_time)
            augmented_spectrogram[:, time_1:time_2].fill_(value)
      
        return augmented_spectrogram
