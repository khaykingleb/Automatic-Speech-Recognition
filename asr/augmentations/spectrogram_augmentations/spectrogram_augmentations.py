import torchaudio
from torch import Tensor

import random

class SpectAugmentation():

    def __init__(self, n_freq_masks: int = 3, n_time_masks: int = 3,
                 freq_portion: float = 0.15, time_portion: float = 0.15, 
                 augmentation_prob: float = 0.6) -> None:
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks
        self.freq_portion = freq_portion
        self.time_portion = time_portion
        self.augmentation_prob = augmentation_prob

    def __call__(self, spectrogram: Tensor) -> Tensor:
        freq_size, time_size = spectrogram.squeeze().shape
        
        if random.random() < self.augmentation_prob: 
            frequency_transform = torchaudio.transforms.FrequencyMasking(round(self.freq_portion * freq_size))
            time_transform = torchaudio.transforms.TimeMasking(round(self.time_portion * time_size))

            for _ in range(self.n_freq_masks):
                spectrogram = frequency_transform(spectrogram)

            for _ in range(self.n_time_masks):
                spectrogram = time_transform(spectrogram)
            
            return spectrogram
        
        else:        
            return spectrogram
