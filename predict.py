#!/usr/bin/env python3
"""
predict.py - Run Whisper model inference on a single audio file.
Usage: python predict.py <path_to_audio_file> <model_dir>
Outputs a single line: the predicted label (e.g. "A", "DONE", etc.)
"""

import sys
import re
import os
import torch
import torchaudio
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# ---- Allowed labels (must match what the model was trained on) ----
EXTRA_WORDS = {"DONE","HELP","BACK","NEXT","REPEAT","START","TICK","CROSS","DELETE","TUTORIAL","SCREENING",
               "ENTER","BACKSPACE","AGAIN","UNDO"}
ALLOWED = set(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")) | EXTRA_WORDS

def normalize_label(x):
    x = x.strip()
    x = re.sub(r"\s+", " ", x)
    return x.upper()

def levenshtein(a, b):
    a, b = a.upper(), b.upper()
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev = dp[0]
        dp[0] = i
        for j, cb in enumerate(b, 1):
            cur = dp[j]
            cost = 0 if ca == cb else 1
            dp[j] = min(dp[j] + 1, dp[j-1] + 1, prev + cost)
            prev = cur
    return dp[-1]

def map_to_allowed(text):
    compact = re.sub(r"[^A-Z]", "", text.upper())
    if not compact:
        return ""
    for lab in sorted(ALLOWED, key=len, reverse=True):
        if compact.startswith(lab):
            return lab
    best, best_d = None, 10**9
    for lab in ALLOWED:
        d = levenshtein(compact, lab)
        if d < best_d:
            best_d, best = d, lab
    return best or ""

def load_audio_16k_mono(path: str) -> torch.Tensor:
    # Convert webm to wav if needed using ffmpeg
    if path.endswith('.webm'):
        import subprocess
        wav_path = path.replace('.webm', '.wav')
        subprocess.run([
            'ffmpeg', '-y', '-i', path,
            '-ar', '16000', '-ac', '1', wav_path
        ], capture_output=True)
        path = wav_path
    
    wav, sr = torchaudio.load(path)
    wav = wav.mean(dim=0)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    return wav

def predict(audio_path, model_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = WhisperProcessor.from_pretrained(model_dir)
    model = WhisperForConditionalGeneration.from_pretrained(model_dir).to(device)
    model.eval()

    try:
        model.generation_config.language = "en"
        model.generation_config.task = "transcribe"
        model.generation_config.max_length = None
    except Exception:
        pass

    wav = load_audio_16k_mono(audio_path)
    feats = processor.feature_extractor(
        wav.numpy(), sampling_rate=16000,
        return_tensors="pt", return_attention_mask=True
    )
    input_features = feats["input_features"].to(device)
    attn = feats.get("attention_mask", None)
    if attn is not None:
        attn = attn.to(device)

    with torch.no_grad():
        pred_ids = model.generate(
            input_features,
            attention_mask=attn,
            max_new_tokens=4,
            num_beams=1,
            do_sample=False,
            repetition_penalty=1.25,
            no_repeat_ngram_size=2,
        )

    raw_text = processor.tokenizer.decode(pred_ids[0], skip_special_tokens=True)
    raw_text = normalize_label(raw_text)
    pred = map_to_allowed(raw_text)

    # Single letter must stay single character
    # (only trim if the gold was a letter — we don't know gold here, so just return pred)
    return pred

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("ERROR: Usage: python predict.py <audio_path> <model_dir>", file=sys.stderr)
        sys.exit(1)

    audio_path = sys.argv[1]
    model_dir = sys.argv[2]

    if not os.path.exists(audio_path):
        print(f"ERROR: Audio file not found: {audio_path}", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(model_dir):
        print(f"ERROR: Model directory not found: {model_dir}", file=sys.stderr)
        sys.exit(1)

    result = predict(audio_path, model_dir)
    print(result)  # One clean line — the predicted label