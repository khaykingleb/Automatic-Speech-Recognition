import json
import logging
import os
import shutil

import random
import pandas as pd

import torchaudio
from speechbrain.utils.data_utils import download_file

from asr.datasets.base_dataset import BaseDataset
from asr.text_encoder.ctc_text_encoder import CTCTextEncoder
from asr.utils import ROOT_PATH
from asr.utils.parse_config import ConfigParser

logger = logging.getLogger(__name__)

URL = 'https://common-voice-data-download.s3.amazonaws.com/cv_corpus_v1.tar.gz'


class MozillaCommonVoice(BaseDataset):

    def __init__(self, part, data_dir=None, *args, **kwargs):

        if data_dir is None:
            try:
                data_dir = ROOT_PATH / "data" / "datasets" / "mozilla_common_voice"
                data_dir.mkdir(exist_ok=True, parents=True)
            except Exception:
                print('Directory is already created.')

        self._data_dir = data_dir

        index = self.get_or_load_index(part)

        super().__init__(index, *args, **kwargs)

    def load_part(self, part):
        arch_path = self._data_dir / "cv_corpus_v1.tar.gz"

        print(f"Loading Mozilla Common Voice")
        download_file(URL, arch_path)

        shutil.unpack_archive(arch_path, self._data_dir)

        for fpath in (self._data_dir / "cv_corpus_v1").iterdir():
            shutil.move(str(fpath), str(self._data_dir / fpath.name))

        os.remove(str(arch_path))

        shutil.rmtree(str(self._data_dir / "cv_corpus_v1"))

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
        split_dir = self._data_dir / "cv-other-train.csv"
        if not split_dir.exists():
            self.load_part(part)

        df_metadata = pd.read_csv(self._data_dir / "cv-other-train.csv")

        indexes = df_metadata.index.tolist()
        random.Random(42).shuffle(indexes)
        df_metadata = df_metadata.loc[indexes]

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
        
        df_metadata.reset_index(inplace=True)
        
        for i in range(len(df_metadata)):
            mp3_path = self._data_dir / df_metadata["filename"][i]
            mp3_path = str(mp3_path.absolute().resolve())

            text = CTCTextEncoder.normalize_text(df_metadata["text"][i])

            t_info = torchaudio.info(mp3_path)
            audio_len = t_info.num_frames / t_info.sample_rate

            index.append({"path": mp3_path,
                          "text": text,
                          "audio_len": audio_len})
             
        return index

if __name__ == "__main__":
    text_encoder = CTCTextEncoder.get_simple_alphabet()
    config_parser = ConfigParser.get_default_configs()

    dataset = MozillaCommonVoice(text_encoder=text_encoder, config_parser=config_parser)
    item = dataset[0]
    print(item)
