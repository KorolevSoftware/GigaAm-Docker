#!/usr/bin/env python3
"""Generate fixed float32 frontend coefficients with the reference toolchain."""
import json
from pathlib import Path
import torch
import torchaudio

assert torch.__version__.split('+')[0] == '2.7.1'
assert torchaudio.__version__.split('+')[0] == '2.7.1'
transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000, n_fft=320, win_length=320, hop_length=160,
    n_mels=64, center=False)
fb = transform.mel_scale.fb
record = dict(torch='2.7.1', torchaudio='2.7.1', sample_rate=16000,
              n_fft=320, hop_length=160, center=False,
              window=transform.spectrogram.window.tolist(),
              filters=[dict(bin=k, mel=m, weight=float(fb[k, m]))
                       for k in range(161) for m in range(64) if fb[k, m] != 0])
path = Path(__file__).resolve().parents[1] / 'internal/inference/feature_coefficients.json'
path.write_text(json.dumps(record, indent=2) + '\n')
