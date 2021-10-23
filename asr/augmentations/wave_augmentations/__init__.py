from asr.augmentations.wave_augmentations.wave_augmentations import GaussianNoiseAugmentation
from asr.augmentations.wave_augmentations.wave_augmentations import TimeStretchingAugmentation
from asr.augmentations.wave_augmentations.wave_augmentations import PitchShiftingAugmentation
from asr.augmentations.wave_augmentations.wave_augmentations import VolumeAugmentation

__all__ = [
    "GaussianNoiseAugmentation",
    "TimeStretchingAugmentation",
    "PitchShiftingAugmentation",
    "VolumeAugmentation"
]
