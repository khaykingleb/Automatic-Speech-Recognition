from torch import nn
import torch

from torch import Tensor

from asr.models.base_model import BaseModel


class Deepspeech(BaseModel):

    def __init__(self, n_feats, n_class, 
                 fc_hidden=512, fc_dropout=0.1,
                 gru_hidden=512, gru_num_layers=3, 
                 *args, **kwargs):
        super().__init__(n_feats, n_class, *args, **kwargs)
        self.fc_hidden = fc_hidden
        self.gru_hidden = gru_hidden
        self.gru_num_layers = gru_num_layers

        self.encoder = nn.Sequential(nn.Linear(in_features=n_feats, out_features=fc_hidden, bias=True), 
                                     nn.ReLU(),
                                     nn.Hardtanh(min_val=0, max_val=20),
                                     nn.Dropout(p=fc_dropout),
                                     nn.Linear(in_features=fc_hidden, out_features=fc_hidden, bias=True),
                                     nn.ReLU(),
                                     nn.Hardtanh(min_val=0, max_val=20),
                                     nn.Dropout(p=fc_dropout),
                                     nn.Linear(in_features=fc_hidden, out_features=fc_hidden, bias=True),
                                     nn.ReLU(),
                                     nn.Hardtanh(min_val=0, max_val=20),
                                     nn.Dropout(p=fc_dropout))
        
        self.bi_gru = nn.GRU(input_size=fc_hidden, hidden_size=gru_hidden, num_layers=gru_num_layers, 
                             bidirectional=True)

        self.decoder = nn.Sequential(nn.Linear(in_features=fc_hidden, out_features=fc_hidden, bias=True),
                                     nn.ReLU(),
                                     nn.Hardtanh(min_val=0, max_val=20),
                                     nn.Dropout(p=fc_dropout),
                                     nn.Linear(in_features=fc_hidden, out_features=n_class))

    def forward(self, spectrogram, *args, **kwargs):
        output = self.encoder(spectrogram)

        output = output.squeeze(1).transpose(0, 1)
        output, (h_n) = self.bi_gru(output)

        output = output[:, :, :self.gru_hidden] + output[:, :, self.gru_hidden:]
        output = self.decoder(output)

        return {"logits": output.permute(1, 0, 2)}

    def transform_input_lengths(self, input_lengths):
        return input_lengths
