from typing import List

import torch
from torch import Tensor

from asr.text_encoder.text_encoder import TextEncoder
from asr.metrics.utils import calc_cer


class ArgmaxCERMetric():

    def __init__(self, text_encoder: TextEncoder, name=None, *args, **kwargs):
        self.name = name if name is not None else type(self).__name__
        self.text_encoder = text_encoder

    def __call__(self, log_probs: Tensor, text: List[str], *args, **kwargs):
        cers = []

        predictions = torch.argmax(log_probs.cpu(), dim=-1)

        for log_prob_vec, target_text in zip(predictions, text):
            if hasattr(self.text_encoder, "ctc_decode"):
                pred_text = self.text_encoder.ctc_decode(log_prob_vec)
            else:
                pred_text = self.text_encoder.decode(log_prob_vec)

            cers.append(calc_cer(target_text, pred_text))

        return sum(cers) / len(cers)