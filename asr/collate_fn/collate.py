import logging
from typing import List

import torch
from torch.nn.utils.rnn import pad_sequence

from asr.text_encoder.text_encoder import TextEncoder

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items.
    """
    audio = [item["audio"] for item in dataset_items]

    spectrogram_length = torch.LongTensor([item["spectrogram"].size()[-1] for item in dataset_items])

    spectrogram = [item["spectrogram"].squeeze().T for item in dataset_items]
    spectrogram = pad_sequence(sequences=spectrogram, padding_value=0)
    spectrogram = spectrogram.permute(1, 0, 2)

    text_encoded = [item["text_encoded"].T for item in dataset_items]
    text_encoded = pad_sequence(sequences=text_encoded, padding_value=0)
    text_encoded = text_encoded.permute(1, 0, 2).squeeze()

    text_encoded_length = torch.LongTensor([item["text_encoded"].shape[1] for item in dataset_items])

    text = [TextEncoder.normalize_text(item["text"]) for item in dataset_items]

    batch = {
        "audio": audio,
        "spectrogram": spectrogram,
        "spectrogram_length": spectrogram_length,
        "text_encoded": text_encoded,
        "text_encoded_length": text_encoded_length,
        "text": text
    }
    
    return batch
