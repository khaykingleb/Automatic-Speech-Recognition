from torch import nn
import torch
from typing import List

from collections import OrderedDict

from asr.models.base_model import BaseModel

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


class QuartzBlock(nn.Module):

    def __init__(self, R, repeat, in_channels, out_channels, kernel_size, 
                 activation_function, normalization):
        super().__init__()
        self.activation_function = activation_function()
        self.almost_quartz_block = nn.ModuleList()

        for i in range(R):               
            base_module = nn.ModuleList([nn.Conv1d(in_channels=in_channels \
                                                   if i == 0 else out_channels,  # 1D Depthwise Conv
                                                   out_channels=out_channels,
                                                   groups=in_channels \
                                                   if i == 0 else out_channels,
                                                   kernel_size=kernel_size,
                                                   padding="same"),
                                         nn.Conv1d(in_channels=out_channels,  # Pointwise Conv
                                                   out_channels=out_channels, 
                                                   kernel_size=1),
                                         normalization(out_channels)])
            if i != R - 1:
                base_module.append(activation_function())

            self.almost_quartz_block.extend(base_module)

        self.almost_quartz_block = nn.Sequential(*self.almost_quartz_block)

        self.skip_connection = nn.Sequential(nn.Conv1d(in_channels=in_channels \
                                                       if repeat == 0 else out_channels,  # Pointwise Convolution
                                                       out_channels=out_channels, 
                                                       kernel_size=1),
                                             normalization(out_channels))
        
    def forward(self, x): 
        output = self.almost_quartz_block(x)
        output = self.activation_function(output + self.skip_connection(x))
        return output


class QuartzNet(BaseModel):
    """
    QuartzNet’s design implementation based on the proposed paper.
    """

    def __init__(self, 
                 n_feats=128, 
                 n_class=28,
                 hidden_channels=256,
                 normalization_type=None, 
                 activation_function_type=None, 
                 B=5,
                 S=1,
                 R=5,
                 kernel_sizes_for_blocks: List[int] = [33, 39, 51, 63, 75], 
                 *args, **kwargs):
        super().__init__(n_feats, n_class, *args, **kwargs)

        self.n_feats = n_feats
        
        normalization = nn.BatchNorm1d if normalization_type is None else normalization_type
        activation_function = nn.ReLU if activation_function_type is None else activation_function_type

        # C_1: Conv-BN-ReLU         
        self.C_1 = nn.Sequential(nn.Conv1d(in_channels=n_feats,  # 1D Depthwise Conv
                                           out_channels=hidden_channels, 
                                           groups=n_feats,  # The depthwise convolution is 
                                                            # applied independently for
                                                            # each channel
                                           kernel_size=33, 
                                           padding=(33-1)//2,
                                           stride=2),       # C_1 layer has a stride of 2
                                 normalization(hidden_channels),
                                 activation_function())
        
        # Blocks B_1, ...., B_B blocks that repeated S times with R sublocks in each block
        self.B = OrderedDict()

        for i in range(B):
            # For the first three blocks we have "hidden_channels" input channels
            in_channels = hidden_channels if i < 3 else hidden_channels * 2

            # For the first two block we have "hidden_channels" output channels
            out_channels = hidden_channels if i < 2 else hidden_channels * 2

            for j in range(S):
                in_channels = in_channels if i < 2 or j == 0 else out_channels

                self.B[f'B_{i}{j}'] = QuartzBlock(R=R, 
                                                  repeat=j,
                                                  in_channels=in_channels, 
                                                  out_channels=out_channels, 
                                                  kernel_size=kernel_sizes_for_blocks[i], 
                                                  activation_function=activation_function, 
                                                  normalization=normalization) 
        self.B = nn.Sequential(self.B)

        # C_2: Conv-BN-ReLU 
        self.C_2 = nn.Sequential(nn.Conv1d(in_channels=hidden_channels * 2, 
                                           out_channels=hidden_channels * 2, 
                                           groups=hidden_channels * 2, 
                                           kernel_size=87,
                                           padding="same"),
                                 normalization(hidden_channels * 2),
                                 activation_function())
        
        # C_3: Conv-BN-ReLU 
        self.C_3 = nn.Sequential(nn.Conv1d(in_channels=hidden_channels * 2,  
                                           out_channels=hidden_channels * 4, 
                                           groups=hidden_channels * 2,
                                           kernel_size=1),
                                 normalization(hidden_channels * 4),
                                 activation_function())
        
        # C_4: Pointwise Conv 
        self.C_4 = nn.Sequential(nn.Conv1d(in_channels=hidden_channels * 4, 
                                           out_channels=n_class, 
                                           kernel_size=1, 
                                           dilation=2)) # C_4 layer has a dilation of 2

    def forward(self, spectrogram, *args, **kwargs):

        # Use permute, so we have (batch_size, mel_scale, mel_length)
        output = self.C_1(spectrogram.permute(0, 2, 1))
        output = self.B(output)
        output = self.C_2(output)
        output = self.C_3(output)
        output = self.C_4(output)

        return {"logits": output.permute(0, 2, 1)}

    def transform_input_lengths(self, input_lengths):
        transformed_input_lengths = torch.floor((input_lengths - 33) / 2 + 1) + 16
        return transformed_input_lengths.int()
