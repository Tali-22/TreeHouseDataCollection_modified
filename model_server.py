#!/usr/bin/env python3
"""
model_server.py - Persistent prediction server.
Loads the model ONCE and serves predictions over HTTP.
Usage: python model_server.py <model_dir> [--port 5001]
"""

from flask import Flask, request, jsonify
import argparse
import sys
import os

# Reuse all the logic from predict.py
sys.path.insert(0, os.path.dirname(__file__))
from predict import detect_model_type, MODEL_BACKENDS, map_to_allowed, normalize_label

app = Flask(__name__)

# Global — loaded once at startup
MODEL_TYPE   = None
INFER_FN     = None
MODEL_DIR    = None

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    data       = request.json
    audio_path = data.get('audio_path')

    if not audio_path or not os.path.exists(audio_path):
        return jsonify({"error": "audio file not found"}), 400

    try:
        raw_text = INFER_FN(audio_path, MODEL_DIR)
        pred     = map_to_allowed(raw_text)
        return jsonify({"label": pred})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "model_type": MODEL_TYPE, "model_dir": MODEL_DIR})

@app.route('/switch_model', methods=['POST'])
def switch_model():
    """Hot-swap the model without restarting the server."""
    global MODEL_TYPE, INFER_FN, MODEL_DIR
    data      = request.json
    new_dir   = data.get('model_dir')
    if not new_dir or not os.path.exists(new_dir):
        return jsonify({"error": "model_dir not found"}), 400
    MODEL_TYPE = detect_model_type(new_dir)
    INFER_FN   = MODEL_BACKENDS.get(MODEL_TYPE)
    MODEL_DIR  = new_dir
    return jsonify({"status": "switched", "model_type": MODEL_TYPE})

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir')
    parser.add_argument('--port', type=int, default=5001)
    args = parser.parse_args()

    MODEL_DIR  = args.model_dir
    MODEL_TYPE = detect_model_type(MODEL_DIR)
    INFER_FN   = MODEL_BACKENDS.get(MODEL_TYPE)

    if not INFER_FN:
        print(f"[ERROR] Unknown model type in {MODEL_DIR}")
        sys.exit(1)

    # Warm up — load model into memory now
    print(f"[SERVER] Loading {MODEL_TYPE} model from {MODEL_DIR}...")
    print(f"[SERVER] Ready on port {args.port}")
    app.run(port=args.port, debug=False)