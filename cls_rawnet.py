from pytorch_metric_learning import losses
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchaudio.transforms as T
import torchaudio.functional as TAF
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch

import sys, time, os, argparse

sys.path.append("rawnet")
import RawNet3 as RawNet3

from dataclasses import dataclass, field

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_curve

from typing import List, Tuple
import functools
from tqdm import tqdm
import numpy as np
import pandas as pd
import random

from audiomentations import Compose, SevenBandParametricEQ, RoomSimulator, AirAbsorption, TanhDistortion, TimeStretch, PitchShift, AddGaussianNoise, Gain, Shift, BandStopFilter, AddBackgroundNoise, PolarityInversion


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

root_dir = "c:\\Users\\Profi\\Downloads\\ru-uz\\data"
batch_size = 16
device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    
class CommandDataset(Dataset):

    def __init__(self, meta, root_dir, sample_rate, labelmap, augment=True):
        self.meta = meta
        self.root_dir = root_dir
        self.sample_rate = sample_rate
        self.labelmap = labelmap
        self.augment = augment
        self.sigma = 30
        self.scales = 3
        self.thresholdingFactor = 1
        
        n_mels = 256
        hop_length = 64
        sample_rate = 16000 
        
#        self.curv_trans = CurveletsOperator(256, n_angles, nu_a = 0.2, nu_b = 0.1)  
        self.resampler = torchaudio.transforms.Resample(orig_freq=32000, new_freq=16000)
        
        if self.augment:
#            self.augmentations = Compose([
#                TimeStretch(min_rate=0.8, max_rate=1.2, p=0.1),
#                PitchShift(min_semitones=-6, max_semitones=6, p=0.1),
#                AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.030, p=0.1),
#                Gain(min_gain_in_db=-3, max_gain_in_db=3, p=0.1),
#                BandStopFilter(min_bandwidth_fraction=0.01, max_bandwidth_fraction=0.25, p=0.1),
#                PolarityInversion(p=0.1),
#                Shift(min_fraction=-0.1, max_fraction=0.1, p=0.1),
#                AirAbsorption(p=0.4),
#                TanhDistortion(p=0.1),
#                SevenBandParametricEQ(p=0.3)
#            ])
            
            self.augmentations = Compose([
                PitchShift(min_semitones=-2, max_semitones=2, p=0.1),
                AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.030, p=0.1),
                Gain(min_gain_in_db=-3, max_gain_in_db=3, p=0.1),
                BandStopFilter(min_bandwidth_fraction=0.01, max_bandwidth_fraction=0.15, p=0.1),
                Shift(min_fraction=-0.1, max_fraction=0.1, p=0.1),
                AirAbsorption(p=0.1),
                TanhDistortion(p=0.05),
                SevenBandParametricEQ(p=0.1),
                
                RoomSimulator(
                    min_room_size=1,
                    max_room_size=30,
                    p=0.1
                ),  # Симуляция различных помещений
                
                ClippingDistortion(
                    min_percentile_threshold=20,
                    max_percentile_threshold=40,
                    p=0.05
                ),  # Легкие искажения, имитирующие плохой микрофон
                
                PeakingFilter(
                    min_center_freq=50,
                    max_center_freq=5000,
                    min_gain_db=-6,
                    max_gain_db=6,
                    p=0.1
                ),  # Усиление/ослабление определенных частот
                
                LowPassFilter(
                    min_cutoff_freq=2000,
                    max_cutoff_freq=7500,
                    p=0.1
                ),  # Имитация телефонной связи/плохого качества записи
                
                HighPassFilter(
                    min_cutoff_freq=20,
                    max_cutoff_freq=200,
                    p=0.1
                ),  # Убирает низкочастотный шум
                
                Mp3Compression(
                    min_bitrate=48000,
                    max_bitrate=96000,
                    p=0.1
                ),  # Имитация сжатия аудио
                
                LowShelfFilter(
                    min_center_freq=50,
                    max_center_freq=200,
                    min_gain_db=-3,
                    max_gain_db=3,
                    p=0.1
                ),  # Тонкая настройка низких частот
                
                HighShelfFilter(
                    min_center_freq=2000,
                    max_center_freq=7500,
                    min_gain_db=-3,
                    max_gain_db=3,
                    p=0.1
                )  # Тонкая настройка высоких частот
            ])

    
    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file_name = self.meta['path'].iloc[idx]
        signal, sample_rate = torchaudio.load(file_name)
        
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            signal = resampler(signal)
        
        if signal.shape[0] != 1:
            signal = signal.mean(dim=0, keepdim=True)
            
        size = 16000*3

        if signal.shape[1] < size:
            padding_size = size - signal.shape[1]
            signal = F.pad(signal, (0, padding_size))

        start_sample = random.randint(0, signal.shape[1] - size)
        end_sample = start_sample + size
        signal = signal[:, start_sample:end_sample]
#        signal = signal[:, 0:size]
        
        if self.augment:
            signal = self.augmentations(samples=signal.numpy(), sample_rate=16000)
            signal = torch.from_numpy(signal)

        label = self.meta['label'].iloc[idx] 

        return signal, self.labelmap[label]


labels = {
    'angry': 0,
    'neutral': 1,
    'other': 2,
    'positive': 3,
    'sad': 4
}

data = pd.DataFrame([
    {'label': i[0].split("\\")[-1], 'path': i[0] + "\\" + j}
    for i in os.walk(root_dir)
    for j in i[2]
])

train, val, _, _ = train_test_split(data, data['label'], test_size=0.2)

train_dataset = CommandDataset(
    meta=train, root_dir=root_dir, sample_rate=16000, labelmap=labels, augment=False)
train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

val_dataset = CommandDataset(
    meta=val, root_dir=root_dir, sample_rate=16000, labelmap=labels, augment=False)
val_dataloader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, num_workers=1)


def number_of_correct(pred, target):
    # count number of correct predictions
    return pred.squeeze().eq(target).sum().item()


def get_likely_index(tensor):
    # find most likely label index for each element in the batch
    return tensor.argmax(dim=-1)

model = (RawNet3.MainModel(
    nOut=256,
    encoder_type="ECA",
    sinc_stride=3,
    max_frame = 200,
    sr=16000
    ))

model = model.to("cuda:0")
#state_dict = torch.load("rawnet_temp_model.pth")
#model.load_state_dict(state_dict)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

EPOCHS = 40
lr = 0.00001
best_val_loss = float('inf')
epochs_without_improvement = 0

optimizer = optim.AdamW(
    model.parameters(), 
    lr=lr,
    weight_decay=0.01,
    betas=(0.9, 0.999)
)

#scheduler = CosineAnnealingLR(
#    optimizer,
#    T_max=EPOCHS,
#    eta_min=lr * 0.1
#)
#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

if __name__ == '__main__':
    for epoch in range(EPOCHS):

        model.train()

        train_loss = []
        for batch, targets in tqdm(train_dataloader, desc=f"Epoch: {epoch}"):
            optimizer.zero_grad()
            batch = batch.to(device).squeeze()
            targets = targets.to(device)
            predictions = model(batch)

            loss = F.nll_loss(predictions, targets)
            loss.backward()
        
            optimizer.step()

            train_loss.append(loss.item())

        print('Training loss:', np.mean(train_loss))

        model.eval()

        val_loss = []
        correct = 0
        all_preds = []
        all_targets = []
        all_probs = []

        for batch, targets in tqdm(val_dataloader, desc=f"Epoch: {epoch}"):

            with torch.no_grad():

                if len(batch.size()) > 2:
                    batch = batch.squeeze()
                batch = batch.to(device)
                targets = targets.to(device)
                input = batch
                input1 = input.cpu() 
                
                predictions = model(batch)
                
                loss = F.nll_loss(predictions, targets)

                pred = get_likely_index(predictions).to(device)
                correct += number_of_correct(pred, targets)

                val_loss.append(loss.item())

                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
            
            
        precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='weighted')
        print(f'Precision: {precision:.2f}')
        print(f'Recall: {recall:.2f}')
        print(f'F1 score: {f1:.2f}')
        
        if np.mean(val_loss) < best_val_loss:
            best_val_loss = np.mean(val_loss)
            epochs_without_improvement = 0
            
            torch.save(model.state_dict(), 'rawnet_temp_model.pth')

        else:
            epochs_without_improvement += 1

        # Ранняя остановка
        if epochs_without_improvement >= 5:
            print(f'Early stopping at epoch {epoch + 1}')
            break

        print(
            f"\nTest Epoch: {epoch}\tAccuracy: {correct}/{len(val_dataloader.dataset)} ({100. * correct / len(val_dataloader.dataset):.0f}%)\n")
        print('Val loss:', np.mean(val_loss))

#        scheduler.step()

    torch.save(model.state_dict(), 'model.pth')
