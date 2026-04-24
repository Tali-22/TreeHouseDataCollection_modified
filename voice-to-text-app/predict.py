#!/usr/bin/env python3
"""
predict.py - Universal voice model inference.
Usage: python predict.py <path_to_audio_file> <model_dir>
Outputs a single line: the predicted label (e.g. "A", "DONE", etc.)

Auto-detects the model type from files present in model_dir:

  Model Type          │ Required files in model_dir
  ────────────────────┼───────────────────────────────────────────────────
  Whisper (HF)        │ config.json  (with "model_type": "whisper")
  Keras CNN           │ best_voice_model.keras  +  label_classes.npy
  PyTorch (custom)    │ model.pt  +  label_classes.npy
  ────────────────────┴───────────────────────────────────────────────────

To add a new model type: implement a loader in MODEL_BACKENDS below.
"""

import sys
import re
import os
import json
import subprocess
import numpy as np

# ── Allowed labels (keep in sync with your training labels) ──────────────────
EXTRA_WORDS = {
    "DONE","HELP","BACK","NEXT","REPEAT","START","TICK","CROSS",
    "DELETE","TUTORIAL","SCREENING","ENTER","BACKSPACE","AGAIN","UNDO"
}
ALLOWED = set(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")) | EXTRA_WORDS


# =============================================================================
# LABEL UTILITIES  (shared across all model types — do not change)
# =============================================================================

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
    """Map any raw model output string → nearest label in ALLOWED."""
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


# =============================================================================
# AUDIO LOADING  (shared — handles wav / mp3 / flac / webm)
# =============================================================================

def load_audio_16k_mono_np(path: str) -> np.ndarray:
    """Load any audio file → 16 kHz mono float32 numpy array."""
    import soundfile as sf
    import librosa

    if any(path.endswith(ext) for ext in ('.webm', '.ogg', '.m4a', '.opus')):
        wav_path = path.rsplit('.', 1)[0] + '_tmp.wav'
        subprocess.run(
            ['ffmpeg', '-y', '-i', path, '-ar', '16000', '-ac', '1', wav_path],
            capture_output=True
        )
        path = wav_path

    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=16000)
    return audio.astype(np.float32)


def load_audio_16k_mono_torch(path: str):
    """Load any audio file → 16 kHz mono torch.Tensor (for Whisper / PyTorch)."""
    import torch
    import torchaudio

    if any(path.endswith(ext) for ext in ('.webm', '.ogg', '.m4a', '.opus')):
        wav_path = path.rsplit('.', 1)[0] + '_tmp.wav'
        subprocess.run(
            ['ffmpeg', '-y', '-i', path, '-ar', '16000', '-ac', '1', wav_path],
            capture_output=True
        )
        path = wav_path

    wav, sr = torchaudio.load(path)
    wav = wav.mean(dim=0)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    return wav


# =============================================================================
# MODEL TYPE DETECTION
# =============================================================================

def detect_model_type(model_dir: str) -> str:
    """
    Inspect model_dir and return one of:
      "whisper"  │  "keras"  │  "pytorch"  │  "unknown"
    """
    files = set(os.listdir(model_dir))

    # ── Whisper: HuggingFace config.json with model_type = "whisper" ─────────
    config_path = os.path.join(model_dir, "config.json")
    if "config.json" in files:
        try:
            with open(config_path) as f:
                cfg = json.load(f)
            if cfg.get("model_type", "").lower() == "whisper":
                return "whisper"
        except Exception:
            pass

    # ── Keras MFCC CNN: .keras + label_classes.npy + model_config.json ──────────
    has_keras  = any(f.endswith(".keras") for f in files)
    has_labels = "label_classes.npy" in files
    has_config = "model_config.json" in files

    if has_keras and has_labels and has_config:
        try:
            with open(os.path.join(model_dir, "model_config.json")) as f:
                cfg = json.load(f)
            if cfg.get("feature_type") == "mfcc":
                return "keras_mfcc"
        except Exception:
            pass

    # ── Keras mel-spectrogram CNN: .keras + label_classes.npy ────────────────
    if has_keras and has_labels:
        return "keras"

    # ── PyTorch custom: model.pt + label map ──────────────────────────────────
    has_pt = "model.pt" in files or "model.pth" in files
    if has_pt and has_labels:
        return "pytorch"

    return "unknown"


# =============================================================================
# MODEL BACKENDS
# Each backend receives (audio_path, model_dir) and returns a raw label string.
# map_to_allowed() is applied afterwards by predict() — backends just return
# whatever the model outputs (a class name, transcription, index, etc.).
# =============================================================================

# ── Backend: Whisper (HuggingFace) ───────────────────────────────────────────
def _infer_whisper(audio_path: str, model_dir: str) -> str:
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = WhisperProcessor.from_pretrained(model_dir)
    model     = WhisperForConditionalGeneration.from_pretrained(model_dir).to(device)
    model.eval()

    try:
        model.generation_config.language  = "en"
        model.generation_config.task      = "transcribe"
        model.generation_config.max_length = None
    except Exception:
        pass

    wav   = load_audio_16k_mono_torch(audio_path)
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
    return normalize_label(raw_text)


# ── Backend: Keras CNN (from voice_recognition_train.py) ─────────────────────
def _infer_keras(audio_path: str, model_dir: str) -> str:
    import librosa
    import tensorflow as tf

    TARGET_SR    = 16_000
    CLIP_SEC     = 3.0
    N_MELS       = 128
    HOP_LENGTH   = 256
    N_FFT        = 1024
    FMIN         = 20
    FMAX         = 8_000
    CLIP_SAMPLES = int(CLIP_SEC * TARGET_SR)
    T_FIXED      = int(np.ceil(CLIP_SAMPLES / HOP_LENGTH)) + 1

    # Locate .keras file dynamically
    keras_files = [f for f in os.listdir(model_dir) if f.endswith(".keras")]
    model_path  = os.path.join(model_dir, keras_files[0])
    labels_path = os.path.join(model_dir, "label_classes.npy")

    model      = tf.keras.models.load_model(model_path)
    le_classes = np.load(labels_path, allow_pickle=True)

    # Preprocess — mirrors training pipeline exactly
    audio = load_audio_16k_mono_np(audio_path)

    audio, _ = librosa.effects.trim(audio, top_db=20)

    stft       = librosa.stft(audio)
    mag, phase = librosa.magphase(stft)
    mag_clean  = mag * (mag > np.mean(mag) * 0.08)
    audio      = librosa.istft(mag_clean * phase)

    if len(audio) >= CLIP_SAMPLES:
        start = (len(audio) - CLIP_SAMPLES) // 2
        audio = audio[start : start + CLIP_SAMPLES]
    else:
        pad_l = (CLIP_SAMPLES - len(audio)) // 2
        pad_r = CLIP_SAMPLES - len(audio) - pad_l
        audio = np.pad(audio, (pad_l, pad_r))

    peak  = np.max(np.abs(audio)) + 1e-9
    audio = (audio / peak).astype(np.float32)

    mel     = librosa.feature.melspectrogram(
        y=audio, sr=TARGET_SR,
        n_mels=N_MELS, hop_length=HOP_LENGTH,
        n_fft=N_FFT, fmin=FMIN, fmax=FMAX
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    log_mel = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min() + 1e-9)
    log_mel = log_mel.astype(np.float32)

    if log_mel.shape[1] < T_FIXED:
        log_mel = np.pad(log_mel, ((0, 0), (0, T_FIXED - log_mel.shape[1])))
    else:
        log_mel = log_mel[:, :T_FIXED]

    inp   = log_mel[np.newaxis, ..., np.newaxis]   # (1, N_MELS, T_FIXED, 1)
    probs = model.predict(inp, verbose=0)[0]
    return normalize_label(str(le_classes[np.argmax(probs)]))


# ── Backend: PyTorch custom model ────────────────────────────────────────────
def _infer_pytorch(audio_path: str, model_dir: str) -> str:
    """
    Generic PyTorch backend.
    Expects model.pt to be a TorchScript or state-dict model that accepts
    a (1, samples) float32 tensor and returns class logits.
    """
    import torch

    labels_path = os.path.join(model_dir, "label_classes.npy")
    model_file  = "model.pt" if os.path.exists(os.path.join(model_dir, "model.pt")) else "model.pth"
    model_path  = os.path.join(model_dir, model_file)

    le_classes = np.load(labels_path, allow_pickle=True)
    model      = torch.jit.load(model_path)   # assumes TorchScript export
    model.eval()

    wav = load_audio_16k_mono_torch(audio_path).unsqueeze(0)   # (1, samples)
    with torch.no_grad():
        logits = model(wav)
    pred_idx = int(torch.argmax(logits, dim=-1).item())
    return normalize_label(str(le_classes[pred_idx]))


# ── Backend: Keras MFCC 1D-CNN (from train_mfcc_cnn.py) ─────────────────────
def _infer_keras_mfcc(audio_path: str, model_dir: str) -> str:
    import librosa
    import tensorflow as tf
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    # Load config saved during training
    with open(os.path.join(model_dir, "model_config.json")) as f:
        cfg = json.load(f)

    TARGET_SR  = cfg["target_sr"]
    N_MFCC     = cfg["n_mfcc"]
    max_length = cfg["max_length"]
    denoise    = cfg.get("denoise", True)

    keras_files = [f for f in os.listdir(model_dir) if f.endswith(".keras")]
    model_path  = os.path.join(model_dir, keras_files[0])
    labels_path = os.path.join(model_dir, "label_classes.npy")
    mean_path   = os.path.join(model_dir, "feature_mean.npy")
    std_path    = os.path.join(model_dir, "feature_std.npy")

    model      = tf.keras.models.load_model(model_path)
    le_classes = np.load(labels_path, allow_pickle=True)
    feat_mean  = np.load(mean_path)  if os.path.exists(mean_path)  else None
    feat_std   = np.load(std_path)   if os.path.exists(std_path)   else None

    # Load and pad audio to same length as training data
    audio = load_audio_16k_mono_np(audio_path)
    audio = pad_sequences([audio], maxlen=max_length,
                          dtype='float32', padding='post',
                          truncating='post', value=0.0)[0]

    # Denoise (same as training)
    if denoise:
        stft             = librosa.stft(audio)
        magnitude, phase = librosa.magphase(stft)
        threshold        = np.mean(magnitude) * 0.1
        clean_mag        = magnitude * (magnitude > threshold)
        audio            = librosa.istft(clean_mag * phase)
        audio            = pad_sequences([audio], maxlen=max_length,
                                         dtype='float32', padding='post',
                                         truncating='post', value=0.0)[0]

    # Extract MFCC + delta + delta-delta
    mfccs        = librosa.feature.mfcc(y=audio, sr=TARGET_SR, n_mfcc=N_MFCC)
    delta_mfccs  = librosa.feature.delta(mfccs)
    delta2_mfccs = librosa.feature.delta(mfccs, order=2)
    features     = np.vstack((mfccs, delta_mfccs, delta2_mfccs)).T   # (time, 39)

    # Normalize using training stats
    if feat_mean is not None and feat_std is not None:
        features = (features - feat_mean) / feat_std

    inp   = np.expand_dims(features, axis=(0, -1))   # (1, time, 39, 1)
    probs = model.predict(inp, verbose=0)[0]
    return normalize_label(str(le_classes[np.argmax(probs)]))


# ── Backend registry — add new model types here ──────────────────────────────
MODEL_BACKENDS = {
    "whisper"    : _infer_whisper,
    "keras"      : _infer_keras,
    "keras_mfcc" : _infer_keras_mfcc,
    "pytorch"    : _infer_pytorch,
}


# =============================================================================
# MAIN PREDICT FUNCTION
# =============================================================================

def predict(audio_path: str, model_dir: str) -> str:
    model_type = detect_model_type(model_dir)

    if model_type == "unknown":
        print(
            f"[ERROR] Could not detect model type in: {model_dir}\n"
            f"  Expected one of:\n"
            f"    Whisper : config.json (model_type=whisper)\n"
            f"    Keras   : *.keras + label_classes.npy\n"
            f"    PyTorch : model.pt  + label_classes.npy",
            file=sys.stderr
        )
        sys.exit(1)

    print(f"[predict] Detected model type: {model_type}", file=sys.stderr)

    raw_text = MODEL_BACKENDS[model_type](audio_path, model_dir)
    pred     = map_to_allowed(raw_text)
    return pred


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("ERROR: Usage: python predict.py <audio_path> <model_dir>", file=sys.stderr)
        sys.exit(1)

    audio_path = sys.argv[1]
    model_dir  = sys.argv[2]

    if not os.path.exists(audio_path):
        print(f"ERROR: Audio file not found: {audio_path}", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(model_dir):
        print(f"ERROR: Model directory not found: {model_dir}", file=sys.stderr)
        sys.exit(1)

    result = predict(audio_path, model_dir)
    print(result)   # One clean line — the predicted label