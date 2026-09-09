#!/usr/bin/env python3
"""Generate a small, synthetic Torch frontend oracle for dependency-free Go CI."""
import json
from pathlib import Path
import numpy as np
import torch
import torchaudio

assert torch.__version__.split('+')[0] == '2.7.1'
n = 3200
t = np.arange(n, dtype=np.float64) / 16000
pcm = np.round(24000 * np.sin(2 * np.pi * (220*t + 1800*t*t)) * (1 - np.arange(n)/n)**2).astype(np.int16)
x = torch.from_numpy(pcm.astype(np.float32) / 32768)
transform = torchaudio.transforms.MelSpectrogram(sample_rate=16000,
    n_fft=320, win_length=320, hop_length=160, n_mels=64, center=False)
features = transform(x).clamp(1e-9, 1e9).log().flatten().tolist()
out = Path(__file__).resolve().parents[1] / 'internal/inference/testdata/features.json'
out.write_text(json.dumps(dict(torch='2.7.1', pcm=pcm.tolist(), features=features))+'\n')
