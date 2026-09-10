#!/usr/bin/env python3
"""Generate golden fixtures using pinned official GigaAM code, not Go output.

Use only public/consented short WAV fixtures: this release tool writes features
and expected text to disk. The production server never invokes this tool.
"""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import sys
import types

import numpy as np
import onnxruntime as ort
from omegaconf import OmegaConf
import soundfile as sf
import torch

REVISION = "7447938d791c4f3e643386ee22c33777004293a5"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--official", type=Path, required=True, help="Official GigaAM checkout at pinned revision")
    p.add_argument("--models", type=Path, required=True)
    p.add_argument("--tokenizers", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("wav", nargs="+", type=Path)
    args = p.parse_args()
    source_lock = json.loads(Path(__file__).with_name("reference-sources.json").read_text())
    for name, checksum in source_lock["files"].items():
        if hashlib.sha256((args.official / "gigaam" / name).read_bytes()).hexdigest() != checksum:
            raise ValueError("Official source revision mismatch: " + name)
    # Avoid the package __init__ loader: it may download PyTorch checkpoints.
    package = types.ModuleType("gigaam")
    package.__path__ = [str(args.official / "gigaam")]
    sys.modules["gigaam"] = package
    preprocess = importlib.import_module("gigaam.preprocess")
    decoding = importlib.import_module("gigaam.decoding")
    onnx_utils = importlib.import_module("gigaam.onnx_utils")
    manifest = json.loads((Path(__file__).resolve().parents[1] / "internal/models/manifest.json").read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    for bundle in manifest["bundles"]:
        if bundle["id"] == "silero-vad" or bundle["precision"] != "fp32":
            continue
        model = bundle["id"]
        prefix = "v3_e2e_" + bundle["architecture"]
        cache = args.models / model / bundle["revision"] / "fp32"
        cfg = OmegaConf.load(cache / (prefix + ".yaml"))
        tokpath = args.tokenizers / (prefix + "_tokenizer.model")
        expected_md5 = cfg.hashes.tokenizer
        if hashlib.md5(tokpath.read_bytes()).hexdigest() != expected_md5:
            raise ValueError("Tokenizer does not match the pinned model export")
        tok = decoding.Tokenizer([], str(tokpath))
        # Assert the Go vocabulary is exactly the official SentencePiece map.
        vocab = (cache / (prefix + "_vocab.txt")).read_text().splitlines()
        assert len(vocab) == len(tok) + 1
        for i in range(len(tok)):
            assert vocab[i].rsplit(" ", 1) == [tok.id_to_str(i), str(i)]
        pre = preprocess.FeatureExtractor(16000, 64, win_length=320, hop_length=160, n_fft=320, center=False)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        def session(suffix):
            return ort.InferenceSession(str(cache / (prefix + suffix)), sess_options=options, providers=["CPUExecutionProvider"])
        sessions = [session(".onnx")] if bundle["architecture"] == "ctc" else [session("_encoder.onnx"), session("_decoder.onnx"), session("_joint.onnx")]
        dest = args.output / model
        dest.mkdir(exist_ok=True)
        for wav in args.wav:
            x, sr = sf.read(wav, dtype="float32")
            assert sr == 16000 and x.ndim == 1 and 320 <= len(x) <= 30 * sr
            copied = args.output / wav.name
            if copied.resolve() != wav.resolve():
                shutil.copyfile(wav, copied)
            with torch.inference_mode():
                features, lengths = pre(torch.from_numpy(x[None, :]), torch.tensor([len(x)]))
            f = features.numpy()
            enc = sessions[0].run(None, dict(zip([v.name for v in sessions[0].get_inputs()], [f, lengths.numpy()])))
            if bundle["architecture"] == "ctc":
                # This pinned export has one CTC output; official subsampling
                # produces ceil(feature_length / 4) valid encoder frames.
                n = (lengths.numpy() - 1) // 4 + 1
                text = onnx_utils._decode_ctc_batch(enc[0].argmax(-1), n, tok)[0]
            else:
                text = onnx_utils._decode_rnnt_batch(enc[0], enc[1], cfg, sessions, tok)[0]
            record = dict(wav=wav.name, text=text, features=f.reshape(-1).tolist(), official_revision=REVISION, model_revision=bundle["revision"])
            (dest / (wav.stem + ".json")).write_text(json.dumps(record, ensure_ascii=False))
            print(model, wav.name, "reference generated", flush=True)


if __name__ == "__main__":
    main()
