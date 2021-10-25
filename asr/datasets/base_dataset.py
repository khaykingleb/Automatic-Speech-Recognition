import logging

import random
import numpy as np

import torch
from torch import Tensor
import torchaudio

from torch.utils.data import Dataset

from asr.text_encoder.text_encoder import TextEncoder
from asr.utils.parse_config import ConfigParser

logger = logging.getLogger(__name__)


def normalize_spectrogram(type: str, spectrogram: Tensor) -> Tensor:
    spectrogram = torch.log(torch.clamp(spectrogram, min=1e-16))
    
    normalized_spectrogram = (spectrogram - torch.mean(spectrogram, dim=1, keepdim=True)) \
                            / (torch.std(spectrogram, dim=1, keepdim=True) + 1e-16)

    return normalized_spectrogram


class BaseDataset(Dataset):
    
    def __init__(self,
                 index,
                 text_encoder: TextEncoder,
                 config_parser: ConfigParser,
                 wave_augs=None,
                 spec_augs=None,
                 limit=None,
                 max_audio_length=None,
                 max_text_length=None):

        self.text_encoder = text_encoder
        self.config_parser = config_parser
        self.wave_augs = wave_augs
        self.spec_augs = spec_augs

        for entry in index:
            assert "audio_len" in entry, ("Each dataset item should include field 'audio_len'"
                                          " - duration of audio (in seconds).")

            assert "path" in entry, ("Each dataset item should include field 'path'" " - path to audio file.")

            assert "text" in entry, ("Each dataset item should include field 'text'"
                                     " - text transcription of the audio.")

        index = self.filter_records_from_dataset(index, max_audio_length, max_text_length, limit)

        # It's a good idea to sort index by audio length
        # It would be easier to write length-based batch samplers later
        index = self.sort_index(index)
        self.index = index

    def __getitem__(self, ind):
        data_dict = self.index[ind]
        audio_path = data_dict["path"]
        audio_wave = self.load_audio(audio_path)
        audio_wave, audio_spec = self.process_wave(audio_wave)

        return {
            "audio": audio_wave,
            "spectrogram": audio_spec,
            "duration": data_dict["audio_len"],
            "text": data_dict["text"],
            "text_encoded": self.text_encoder.encode(data_dict["text"]),
            "audio_path": audio_path
        }

    @staticmethod
    def sort_index(index):
        return sorted(index, key=lambda x: x["audio_len"])

    def __len__(self):
        return len(self.index)

    def load_audio(self, path):
        audio_tensor, sr = torchaudio.load(path)
        audio_tensor = audio_tensor[0:1, :]  # remove all channels but the first
        target_sr = self.config_parser["preprocessing"]["sr"]
        if sr != target_sr:
            transform = torchaudio.transforms.Resample(sr, target_sr)
            audio_tensor = transform(audio_tensor)
        return audio_tensor

    def process_wave(self, audio_tensor_wave: Tensor):
        with torch.no_grad():

            if self.wave_augs is not None:
                audio_tensor_wave = self.wave_augs(audio_tensor_wave)

            wave2spec = self.config_parser.init_obj(self.config_parser["preprocessing"]["spectrogram"],
                                                    torchaudio.transforms)
            
            audio_tensor_spec = wave2spec(audio_tensor_wave)
            audio_tensor_spec = normalize_spectrogram(audio_tensor_spec)

            if self.spec_augs is not None:
                audio_tensor_spec = self.spec_augs(audio_tensor_spec)

            return audio_tensor_wave, audio_tensor_spec

    @staticmethod
    def filter_records_from_dataset(index: list, max_audio_length, max_text_length, limit) -> list:
        initial_size = len(index)

        if max_audio_length is not None:
            exceeds_audio_length = (np.array([el["audio_len"] for el in index]) >= max_audio_length)
            total = exceeds_audio_length.sum()
            logger.info(f"{total} ({total / initial_size:.1%}) records are longer then "
                        f"{max_audio_length} seconds. Excluding them.")
        else:
            exceeds_audio_length = False

        initial_size = len(index)

        if max_text_length is not None:
            exceeds_text_length = (np.array([len(TextEncoder.normalize_text(el["text"])) for el in index])
                                   >= max_text_length)
            total = exceeds_text_length.sum()
            logger.info(f"{total} ({total / initial_size:.1%}) records are longer then"
                        f"{max_text_length} characters. Excluding them.")
        else:
            exceeds_text_length = False

        records_to_filter = exceeds_text_length | exceeds_audio_length

        if records_to_filter is not False and records_to_filter.any():
            total = records_to_filter.sum()
            index = [el for el, exclude in zip(index, records_to_filter) if not exclude]
            logger.info(f"Filtered {total}({total / initial_size:.1%}) records  from dataset")

        if limit is not None:
            random.seed(42)  # best seed for deep learning
            random.shuffle(index)
            index = index[:limit]
        
        return index
