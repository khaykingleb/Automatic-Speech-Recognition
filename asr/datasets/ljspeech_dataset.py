import json
import logging
import os
import shutil

import pandas as pd

import torchaudio
from speechbrain.utils.data_utils import download_file

from asr.datasets.base_dataset import BaseDataset
from asr.text_encoder.ctc_text_encoder import CTCTextEncoder
from asr.utils import ROOT_PATH
from asr.utils.parse_config import ConfigParser

logger = logging.getLogger(__name__)

URL = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"


class LJSpeechDataset(BaseDataset):

    def __init__(self, part, data_dir=None, *args, **kwargs):

        if data_dir is None:
            try:
                data_dir = ROOT_PATH / "data" / "datasets" / "ljspeech"
                data_dir.mkdir(exist_ok=True, parents=True)

            except Exception:
                print('Directory is already created.')

        self._data_dir = data_dir

        index = self.get_or_load_index(part)

        super().__init__(index, *args, **kwargs)

    def load_part(self, part):
        arch_path = self._data_dir / "LJSpeech-1.1.tar.bz2"

        print(f"Loading LJSpeech-1.1")
        download_file(URL, arch_path)

        shutil.unpack_archive(arch_path, self._data_dir)

        for fpath in (self._data_dir / "LJSpeech-1.1").iterdir():
            shutil.move(str(fpath), str(self._data_dir / fpath.name))

        os.remove(str(arch_path))

        shutil.rmtree(str(self._data_dir / "LJSpeech-1.1"))

    def get_or_load_index(self, part):

        index_path = self._data_dir / f"{part}_index.json"

        if index_path.exists():
            with index_path.open() as f:
                index = json.load(f)
        else:
            index = self.create_index(part)
            with index_path.open("w") as f:
                json.dump(index, f, indent=2)

        return index
    
    def create_index(self, part):
        index = []

        self.load_part(part)

        wavs_dir = self._data_dir / 'wavs'

        df_metadata = pd.read_csv(self._data_dir / 'metadata.csv', sep='|', 
                                  header=None)

        train_size = round(len(df_metadata) * 0.6)
        val_size = round(len(df_metadata) * 0.2)

        if part == "train":
            df_metadata = df_metadata[:train_size]
        elif part == "val":
            df_metadata = df_metadata[train_size:train_size+val_size]
        elif part == "test":
            df_metadata = df_metadata[train_size+val_size:]
        else:
            raise ValueError("There is no such part for the given dataset.")
        
        for i, audio_name in enumerate(df_metadata[0]):
            wav_path = wavs_dir / str(audio_name + '.wav')
            wav_path = str(wav_path.absolute().resolve())

            text = CTCTextEncoder.normalize_text(df_metadata[1][i])

            t_info = torchaudio.info(wav_path)
            audio_len = t_info.num_frames / t_info.sample_rate

            index.append({"path": wav_path,
                          "text": text,
                          "audio_len": audio_len})
             
        return index

if __name__ == "__main__":
    text_encoder = CTCTextEncoder.get_simple_alphabet()
    config_parser = ConfigParser.get_default_configs()

    dataset = LJSpeechDataset(text_encoder=text_encoder, config_parser=config_parser)
    item = dataset[0]
    print(item)
