import numpy as np

from torch import Tensor
import torch
import torchaudio
import librosa

class GaussianNoiseAugmentation():

    def __init__(self, std=0.05):
        self.dist = torch.distributions.Normal(0, std)

    def __call__(self, audio: Tensor):
        augmented_audio = audio + self.dist.sample(audio.shape)

        return augmented_audio

class TimeStretchingAugmentation():
    def __init__(self, stretch="random"):
        self.stretch = np.random.uniform(0.5, 2) if stretch == "random" else float(stretch)
        
    def __call__(self, audio: Tensor):
        augmented_audio = librosa.effects.time_stretch(audio.numpy().squeeze(), self.stretch)

        return torch.tensor(augmented_audio).reshape(1, -1)

class PitchShiftingAugmentation():

    def __init__(self, sample_rate=22050, n_steps="random"):
        self.sample_rate = sample_rate
        self.n_steps = np.random.randint(-5, 5) if n_steps == "random" else int(n_steps)
        
    def __call__(self, audio: Tensor):
        augmented_audio = librosa.effects.pitch_shift(audio.numpy().squeeze(), 
                                                      self.sample_rate, self.n_steps)

        return torch.tensor(augmented_audio).reshape(1, -1)

class VolumeAugmentation():

    def __init__(self, gain="random"):
        self.gain = np.random.uniform(0.2, 1) if gain == "random" else float(gain)
        
    def __call__(self, audio: Tensor):
        augmentation = torchaudio.transforms.Vol(self.gain, gain_type='amplitude')
        augmented_audio = augmentation(audio)

        return augmented_audio