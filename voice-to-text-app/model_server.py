#!/usr/bin/env python3
"""
model_server.py — Persistent prediction server with model caching.

Loads the model ONCE at startup (or on /switch_model) and keeps it in memory.
Every subsequent /predict call reuses the cached model — no reloading.

Usage:
    python model_server.py <model_dir> [--port 5001]
"""

from flask import Flask, request, jsonify
import argparse
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from predict import detect_model_type, map_to_allowed, load_audio_16k_mono_np

app = Flask(__name__)

# ── Global model cache ──────────────────────────────────────────────────────
# These are populated once by _load_model() and reused on every /predict call.
_model       = None   # The loaded model object (Keras / PyTorch / HF)
_le_classes  = None   # numpy array of label strings
_processor   = None   # HuggingFace processor (Whisper only)
_cfg         = None   # model_config.json dict (keras_mfcc only)
_feat_mean   = None   # training mean (keras_mfcc only)
_feat_std    = None   # training std  (keras_mfcc only)

MODEL_TYPE   = None
MODEL_DIR    = None


# ═══════════════════════════════════════════════════════════════════════════════
# LOAD — called once at startup and on /switch_model
# ═══════════════════════════════════════════════════════════════════════════════

def _load_model(model_dir: str):
    """Detect model type and load all artefacts into global cache."""
    global _model, _le_classes, _processor, _cfg, _feat_mean, _feat_std
    global MODEL_TYPE, MODEL_DIR

    mtype = detect_model_type(model_dir)
    if mtype == "unknown":
        raise ValueError(f"Cannot detect model type in: {model_dir}")

    print(f"[SERVER] Loading model type={mtype} from {model_dir} …", flush=True)

    if mtype == "whisper":
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        device = "cuda" if torch.cuda.is_available() else "cpu"
        proc  = WhisperProcessor.from_pretrained(model_dir)
        model = WhisperForConditionalGeneration.from_pretrained(model_dir).to(device)
        model.eval()
        try:
            model.generation_config.language   = "en"
            model.generation_config.task       = "transcribe"
            model.generation_config.max_length = None
        except Exception:
            pass
        _model      = model
        _processor  = proc
        _le_classes = None

    elif mtype == "keras_mfcc":
        import json, tensorflow as tf
        with open(os.path.join(model_dir, "model_config.json")) as f:
            cfg = json.load(f)
        keras_files = [f for f in os.listdir(model_dir) if f.endswith(".keras")]
        _model      = tf.keras.models.load_model(os.path.join(model_dir, keras_files[0]))
        _le_classes = np.load(os.path.join(model_dir, "label_classes.npy"), allow_pickle=True)
        _cfg        = cfg
        mean_p = os.path.join(model_dir, "feature_mean.npy")
        std_p  = os.path.join(model_dir, "feature_std.npy")
        _feat_mean  = np.load(mean_p) if os.path.exists(mean_p) else None
        _feat_std   = np.load(std_p)  if os.path.exists(std_p)  else None

    elif mtype == "keras":
        import tensorflow as tf
        keras_files = [f for f in os.listdir(model_dir) if f.endswith(".keras")]
        _model      = tf.keras.models.load_model(os.path.join(model_dir, keras_files[0]))
        _le_classes = np.load(os.path.join(model_dir, "label_classes.npy"), allow_pickle=True)

    elif mtype == "pytorch":
        import torch
        model_file = "model.pt" if os.path.exists(os.path.join(model_dir, "model.pt")) else "model.pth"
        _model      = torch.jit.load(os.path.join(model_dir, model_file))
        _model.eval()
        _le_classes = np.load(os.path.join(model_dir, "label_classes.npy"), allow_pickle=True)

    else:
        raise ValueError(f"Unsupported model type: {mtype}")

    MODEL_TYPE = mtype
    MODEL_DIR  = model_dir
    print(f"[SERVER] ✅ Model ready: {mtype}", flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# INFER — reuses cached model; mirrors the exact training pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def _infer(audio_path: str) -> str:
    """
    Run inference using whichever model is currently cached.
    Each branch mirrors the training preprocessing exactly — same parameters,
    same order of operations.
    """
    if MODEL_TYPE == "whisper":
        import torch
        import torchaudio
        import subprocess

        device = next(_model.parameters()).device

        # Convert webm/ogg → wav if needed (same as load_audio_16k_mono_torch)
        path = audio_path
        if any(path.endswith(ext) for ext in ('.webm', '.ogg', '.m4a', '.opus')):
            wav_path = path.rsplit('.', 1)[0] + '_tmp.wav'
            import subprocess
            subprocess.run(
                ['ffmpeg', '-y', '-i', path, '-ar', '16000', '-ac', '1', wav_path],
                capture_output=True
            )
            path = wav_path

        wav, sr = torchaudio.load(path)
        wav = wav.mean(dim=0)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)

        feats = _processor.feature_extractor(
            wav.numpy(), sampling_rate=16000,
            return_tensors="pt", return_attention_mask=True
        )
        input_features = feats["input_features"].to(device)
        attn = feats.get("attention_mask", None)
        if attn is not None:
            attn = attn.to(device)

        with torch.no_grad():
            pred_ids = _model.generate(
                input_features,
                attention_mask=attn,
                max_new_tokens=4,
                num_beams=1,
                do_sample=False,
                repetition_penalty=1.25,
                no_repeat_ngram_size=2,
            )
        raw = _processor.tokenizer.decode(pred_ids[0], skip_special_tokens=True)
        return raw.strip().upper()

    elif MODEL_TYPE == "keras":
        import librosa

        # ── Must match voice_recognition_train.py exactly ──
        TARGET_SR    = 16_000
        CLIP_SEC     = 3.0
        N_MELS       = 128
        HOP_LENGTH   = 256
        N_FFT        = 1024
        FMIN         = 20
        FMAX         = 8_000
        CLIP_SAMPLES = int(CLIP_SEC * TARGET_SR)
        T_FIXED      = int(np.ceil(CLIP_SAMPLES / HOP_LENGTH)) + 1

        audio = load_audio_16k_mono_np(audio_path)          # resample → mono → float32

        audio, _ = librosa.effects.trim(audio, top_db=20)   # silence trim

        # Spectral gating (noise reduction)
        stft       = librosa.stft(audio)
        mag, phase = librosa.magphase(stft)
        mag_clean  = mag * (mag > np.mean(mag) * 0.08)
        audio      = librosa.istft(mag_clean * phase)

        # Center-crop or pad to fixed length
        if len(audio) >= CLIP_SAMPLES:
            start = (len(audio) - CLIP_SAMPLES) // 2
            audio = audio[start : start + CLIP_SAMPLES]
        else:
            pad_l = (CLIP_SAMPLES - len(audio)) // 2
            pad_r = CLIP_SAMPLES - len(audio) - pad_l
            audio = np.pad(audio, (pad_l, pad_r))

        # Peak normalise
        peak  = np.max(np.abs(audio)) + 1e-9
        audio = (audio / peak).astype(np.float32)

        # Mel spectrogram → log scale → min-max [0,1]
        mel     = librosa.feature.melspectrogram(
            y=audio, sr=TARGET_SR,
            n_mels=N_MELS, hop_length=HOP_LENGTH,
            n_fft=N_FFT, fmin=FMIN, fmax=FMAX
        )
        log_mel = librosa.power_to_db(mel, ref=np.max)
        log_mel = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min() + 1e-9)
        log_mel = log_mel.astype(np.float32)

        # Pad/crop time axis
        if log_mel.shape[1] < T_FIXED:
            log_mel = np.pad(log_mel, ((0, 0), (0, T_FIXED - log_mel.shape[1])))
        else:
            log_mel = log_mel[:, :T_FIXED]

        inp   = log_mel[np.newaxis, ..., np.newaxis]          # (1, 128, T, 1)
        probs = _model.predict(inp, verbose=0)[0]
        return str(_le_classes[np.argmax(probs)]).strip().upper()

    elif MODEL_TYPE == "keras_mfcc":
        import librosa
        from tensorflow.keras.preprocessing.sequence import pad_sequences

        TARGET_SR  = _cfg["target_sr"]
        N_MFCC     = _cfg["n_mfcc"]
        max_length = _cfg["max_length"]
        denoise    = _cfg.get("denoise", True)

        audio = load_audio_16k_mono_np(audio_path)
        audio = pad_sequences([audio], maxlen=max_length,
                              dtype='float32', padding='post',
                              truncating='post', value=0.0)[0]

        if denoise:
            stft             = librosa.stft(audio)
            magnitude, phase = librosa.magphase(stft)
            threshold        = np.mean(magnitude) * 0.1
            clean_mag        = magnitude * (magnitude > threshold)
            audio            = librosa.istft(clean_mag * phase)
            audio            = pad_sequences([audio], maxlen=max_length,
                                             dtype='float32', padding='post',
                                             truncating='post', value=0.0)[0]

        mfccs        = librosa.feature.mfcc(y=audio, sr=TARGET_SR, n_mfcc=N_MFCC)
        delta_mfccs  = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)
        features     = np.vstack((mfccs, delta_mfccs, delta2_mfccs)).T   # (time, 3*N_MFCC)

        if _feat_mean is not None and _feat_std is not None:
            features = (features - _feat_mean) / (_feat_std + 1e-9)

        inp   = np.expand_dims(features, axis=(0, -1))        # (1, time, 39, 1)
        probs = _model.predict(inp, verbose=0)[0]
        return str(_le_classes[np.argmax(probs)]).strip().upper()

    elif MODEL_TYPE == "pytorch":
        import torch
        import torchaudio

        path = audio_path
        if any(path.endswith(ext) for ext in ('.webm', '.ogg', '.m4a', '.opus')):
            import subprocess
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

        with torch.no_grad():
            logits = _model(wav.unsqueeze(0))
        pred_idx = int(torch.argmax(logits, dim=-1).item())
        return str(_le_classes[pred_idx]).strip().upper()

    else:
        raise ValueError(f"No infer logic for model type: {MODEL_TYPE}")


# ═══════════════════════════════════════════════════════════════════════════════
# FLASK ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok",
        "model_type": MODEL_TYPE,
        "model_dir": MODEL_DIR,
        "model_loaded": _model is not None,
    })


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    data       = request.json or {}
    audio_path = data.get('audio_path')

    if not audio_path or not os.path.exists(audio_path):
        return jsonify({"error": f"audio file not found: {audio_path}"}), 400

    if _model is None:
        return jsonify({"error": "No model loaded yet"}), 503

    try:
        raw  = _infer(audio_path)
        pred = map_to_allowed(raw)
        return jsonify({"label": pred, "raw": raw})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/switch_model', methods=['POST'])
def switch_model():
    """Hot-swap the model without restarting the server."""
    data    = request.json or {}
    new_dir = data.get('model_dir', '')

    # Allow HuggingFace hub IDs (e.g. "openai/whisper-tiny") — no local path check
    is_hub_id = '/' in new_dir and not os.path.isabs(new_dir)
    if not is_hub_id and not os.path.exists(new_dir):
        return jsonify({"error": f"model_dir not found: {new_dir}"}), 400

    try:
        _load_model(new_dir)
        return jsonify({"status": "switched", "model_type": MODEL_TYPE, "model_dir": MODEL_DIR})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════════════
# STARTUP
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Persistent model prediction server")
    parser.add_argument('model_dir', help="Path to model directory (or HuggingFace model ID)")
    parser.add_argument('--port', type=int, default=5001)
    args = parser.parse_args()

    try:
        _load_model(args.model_dir)
    except Exception as e:
        print(f"[SERVER] ❌ Failed to load model: {e}")
        sys.exit(1)

    print(f"[SERVER] 🚀 Listening on port {args.port}")
    app.run(host='0.0.0.0', port=args.port, debug=False, threaded=False)
    # threaded=False is intentional — TensorFlow/PyTorch models are NOT thread-safe.
    # Requests are queued and served one at a time, which is correct for inference.