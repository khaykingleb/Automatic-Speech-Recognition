import logging
from pathlib import Path

import torchaudio

from asr.datasets.base_dataset import BaseDataset
from asr.text_encoder.ctc_text_encoder import CTCTextEncoder
from asr.utils.parse_config import ConfigParser

logger = logging.getLogger(__name__)


class CustomAudioDataset(BaseDataset):
    def __init__(self, data, split=None, *args, **kwargs):
        index = data
        for entry in data:
            assert "path" in entry
            assert Path(entry["path"]).exists()
            entry["path"] = str(Path(entry["path"]).absolute().resolve())
            entry["text"] = entry.get("text", "")
            t_info = torchaudio.info(entry["path"])
            entry["audio_len"] = t_info.num_frames / t_info.sample_rate

        super().__init__(index, *args, **kwargs)

    def __getitem__(self, ind):
        data_dict = self._index[ind]
        wav_path = data_dict["path"]
        audio_wave = self.load_audio(wav_path)
        audio_wave, audio_spec = self.process_wave(audio_wave)
        return {
            "audio": audio_wave,
            "spectrogram": audio_spec,
            "duration": data_dict["audio_len"],
            "text": data_dict["text"],
            "text_encoded": self.text_encoder.encode(data_dict["text"]),
            "audio_path": wav_path,
        }


if __name__ == "__main__":
    text_encoder = CTCTextEncoder.get_simple_alphabet()
    config_parser = ConfigParser.get_default_configs()

    data = [
        {
            "path": "data/datasets/mini_librispeech/LibriSpeech"
                    "/dev-clean-2/84/121550/84-121550-0000.flac",
            "text": "BUT WITH FULL RAVISHMENT THE HOURS OF PRIME SINGING "
                    "RECEIVED THEY IN THE MIDST OF LEAVES THAT EVER BORE A BURDEN TO THEIR RHYMES"
        },
        {
            "path": "data/datasets/mini_librispeech/LibriSpeech/"
                    "dev-clean-2/84/121550/84-121550-0001.flac",
        }
    ]

    ds = CustomAudioDataset(data, text_encoder=text_encoder, config_parser=config_parser)
