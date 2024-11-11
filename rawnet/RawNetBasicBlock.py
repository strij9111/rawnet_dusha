import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PreEmphasis(nn.Module):
    """
    Класс для применения преэмфазного фильтра к аудиосигналу с возможностью
    использования фильтров произвольного порядка и обучаемыми коэффициентами.

    Параметры:
        coeffs (list или numpy.array): Коэффициенты фильтра. Если None, используется [1, -0.97].
        channels (int): Количество каналов во входном сигнале. По умолчанию 1.
        trainable (bool): Если True, коэффициенты фильтра обучаются. По умолчанию False.

    Пример использования:
        pre_emphasis = PreEmphasis(coeffs=[1, -0.97], channels=1, trainable=False)
        output = pre_emphasis(input_tensor)
    """
    def __init__(self, coeffs=None, channels=1, trainable=False):
        super().__init__()
        self.channels = channels
        self.trainable = trainable

        if coeffs is None:
            coeffs = [1.0, -0.97]

        coeffs = torch.tensor(coeffs, dtype=torch.float32).view(1, 1, -1)

        if trainable:
            self.coeffs = nn.Parameter(coeffs.repeat(channels, 1, 1))
        else:
            self.register_buffer('coeffs', coeffs.repeat(channels, 1, 1))

    def forward(self, input):
        if input.dim() == 2:
            input = input.unsqueeze(1)  # Добавляем размерность каналов
        elif input.dim() != 3:
            raise ValueError("Входной тензор должен иметь 2 или 3 измерения")

        batch_size, channels, signal_length = input.shape
        if channels != self.channels:
            raise ValueError(f"Ожидалось {self.channels} каналов, но получено {channels}")

        # Применяем свертку с соответствующим паддингом
        padding = self.coeffs.shape[2] - 1
        output = F.conv1d(input, self.coeffs, padding=padding, groups=self.channels)
        output = output[:, :, :signal_length]  # Обрезаем до исходной длины сигнала
        return output


class AFMS(nn.Module):
    """
    Блок Squeeze-and-Excitation (SE) для канального внимания.

    Параметры:
        nb_dim (int): Количество входных каналов.
        reduction (int): Коэффициент сокращения. По умолчанию 16.

    Ссылка:
        Hu, Jie, et al. "Squeeze-and-excitation networks." CVPR, 2018.
    """
    def __init__(self, nb_dim, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(nb_dim, nb_dim // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(nb_dim // reduction, nb_dim, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y


class Bottle2neck(nn.Module):
    def __init__(
        self,
        inplanes,
        planes,
        kernel_size=None,
        dilation=None,
        scale=4,
        pool=False,
    ):

        super().__init__()

        width = int(math.floor(planes / scale))

        self.conv1 = nn.Conv1d(inplanes, width * scale, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(width * scale)

        self.nums = scale - 1

        convs = []
        bns = []

        num_pad = math.floor(kernel_size / 2) * dilation

        for i in range(self.nums):
            convs.append(
                nn.Conv1d(
                    width,
                    width,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    padding=num_pad,
                )
            )
            bns.append(nn.BatchNorm1d(width))

        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv1d(width * scale, planes, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(planes)

        self.relu = nn.ReLU()

        self.width = width

        self.mp = nn.MaxPool1d(pool) if pool else False
        self.afms = AFMS(planes)

        if inplanes != planes:  # if change in number of filters
            self.residual = nn.Sequential(
                nn.Conv1d(inplanes, planes, kernel_size=1, stride=1, bias=False)
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        residual = self.residual(x)

        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(sp)
            sp = self.bns[i](sp)
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)

        out = torch.cat((out, spx[self.nums]), 1)

        out = self.conv3(out)
        out = self.relu(out)
        out = self.bn3(out)

        out += residual
        if self.mp:
            out = self.mp(out)
        out = self.afms(out)

        return out
