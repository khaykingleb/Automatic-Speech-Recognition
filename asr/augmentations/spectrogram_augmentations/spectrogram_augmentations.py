import torchaudio
from torch import Tensor

class SpectAugmentation():

    def __init__(self, n_freq_masks: int = 3, n_time_masks: int = 3,
                 freq_portion: float = 0.15, time_portion: float = 0.15):

        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks
        self.freq_portion = freq_portion
        self.time_portion = time_portion

    def __call__(self, spectrogram: Tensor):
        freq_size, time_size = spectrogram.squeeze().shape

        frequency_transform = torchaudio.transforms.FrequencyMasking(round(self.freq_portion * freq_size))
        time_transform = torchaudio.transforms.TimeMasking(round(self.time_portion * time_size))

        for _ in range(self.n_freq_masks):
            spectrogram = frequency_transform(spectrogram)

        for _ in range(self.n_time_masks):
            spectrogram = time_transform(spectrogram)
        
        return spectrogram
