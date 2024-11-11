import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention
import sys


try : 
    from .RawNetBasicBlock import Bottle2neck, PreEmphasis
    from .asteroid_filterbanks import Encoder, ParamSincFB, MultiphaseGammatoneFB
except ImportError :
    from RawNetBasicBlock import Bottle2neck, PreEmphasis
    from asteroid_filterbanks import Encoder, ParamSincFB, MultiphaseGammatoneFB, MelGramFB

class PositionalEncoding(nn.Module):
    """
    Позиционное кодирование для последовательных данных.
    """
    def __init__(self, d_model, max_len=16000):
        super().__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (
            torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(1, 2)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :, : x.size(2)]
        
class RawNet3(nn.Module):
    def __init__(self, block, model_scale, context, summed, C=1024, sr=16000, nOut=256, **kwargs):
        super().__init__()

        self.context = context
        self.encoder_type = "ECA"
        self.log_sinc = kwargs["log_sinc"]
        self.norm_sinc = kwargs["norm_sinc"]
        self.out_bn = kwargs["out_bn"]
        self.summed = summed
        self.sr = sr

#        print("context : {}".format(self.context))
#        print("log_sinc: {}".format(self.log_sinc))
#        print("norm_sinc: {}".format(self.norm_sinc))
#        print("out_bn: {}".format(self.out_bn))
#        print("summed: {}".format(self.summed))

        self.preprocess = nn.Sequential(
            PreEmphasis(coeffs=[1, -0.97], channels=1, trainable=True),
            nn.BatchNorm1d(1,eps=1e-4,affine=True)
        )
        self.conv1 = Encoder(
            MultiphaseGammatoneFB(
                C // 4,
                251,
                stride=3,
                fmax=sr//2,
                sample_rate=sr
            )
        )
        self.relu = nn.GELU()
        self.bn1 = nn.BatchNorm1d(C // 4)

        self.layer1 = block(
            C // 4, C, kernel_size=3, dilation=2, scale=model_scale, pool=5
        )
        self.layer2 = block(
            C, C, kernel_size=3, dilation=3, scale=model_scale, pool=3
        )
        self.layer3 = block(C, C, kernel_size=3, dilation=4, scale=model_scale)
        self.layer4 = nn.Conv1d(3 * C, 1536, kernel_size=1)

        """
        if self.context:
            attn_input = 1536 * 3
        else:
            attn_input = 1536
        """
        attn_input = 1536 * 3
        attn_output = 1536

        self.attention = nn.Sequential(
            nn.Conv1d(attn_input, 128, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, attn_output, kernel_size=1),
            nn.Softmax(dim=2),
        )
        self.positional_encoding = PositionalEncoding(d_model=C // 4)
        
        self.bn5 = nn.BatchNorm1d(3072)

        self.fc6 = nn.Linear(3072, nOut)
        self.fc7 = nn.Linear(nOut, 5)
        self.bn6 = nn.BatchNorm1d(nOut)
#        self.bn7 = nn.BatchNorm1d(nOut)
        self.mp3 = nn.MaxPool1d(3)
        
    @torch.amp.autocast(device_type='cuda', dtype=torch.float16)
    def forward(self, x):
        # Предварительная обработка и преобразования
        x = self.preprocess(x)
        x = torch.abs(self.conv1(x))
        x = torch.log(x + 1e-6)
        x = x - torch.mean(x, dim=-1, keepdim=True)

        x = self.positional_encoding(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(self.mp3(x1) + x2)
        
        x = self.layer4(torch.cat((self.mp3(x1), x2, x3), dim=1)) 
        x = self.relu(x)
        t = x.size()[-1]

        global_x = torch.cat(
            (
                x,
                torch.mean(x, dim=2, keepdim=True).repeat(1, 1, t),
                torch.sqrt(
                    torch.var(x, dim=2, keepdim=True).clamp(
                        min=1e-4, max=1e4
                    )
                ).repeat(1, 1, t),
            ),
            dim=1,
        )

        w = self.attention(global_x)

        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt(
            (torch.sum((x**2) * w, dim=2) - mu**2).clamp(min=1e-4, max=1e4)
        )
        x = torch.cat((mu, sg), 1)
        x = self.bn5(x)
        x = F.relu(self.fc6(x))
#        x = self.bn6(x)
        x = self.fc7(x)
        
        return F.log_softmax(x, dim=1)


def MainModel(**kwargs):

    model = RawNet3(
        Bottle2neck, model_scale=8, context=True, summed=True, out_bn=False, log_sinc=True, norm_sinc="mean", grad_mult=1, **kwargs
    )
    return model
