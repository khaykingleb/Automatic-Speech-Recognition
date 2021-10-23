from torch import nn
import torch

from asr.models.base_model import BaseModel

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


class DummyModel(BaseModel):

    def __init__(self, n_feats, n_class, gru_hidden=512, gru_num_layers=3, 
                 gru_dropout=0, *args, **kwargs):
        super().__init__(n_feats, n_class, *args, **kwargs)

        self.gru_hidden = gru_hidden
        self.gru_num_layers = gru_num_layers

        self.gru = nn.GRU(input_size=n_feats, hidden_size=gru_hidden,
                          num_layers=gru_num_layers, dropout=gru_dropout, batch_first=False)

        self.fc = nn.Linear(in_features=gru_hidden, out_features=n_class)

    def forward(self, spectrogram, *args, **kwargs):
        h_0 = torch.zeros(self.gru_num_layers, spectrogram.shape[1], self.gru_hidden).to(device).requires_grad_()
        output, (h_n) = self.gru(spectrogram, (h_0.detach()))
        output = self.fc(output)

        return {"logits": output}

    def transform_input_lengths(self, input_lengths):
        return input_lengths
