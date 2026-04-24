# TreeHouseDataCollection_modified

This is a modified version of the voice-recorder-app to make user-friendly live testing of all the upcoming versions of voice recognition models. To use this interface and help with any initial user queries, refer to the report for detailed instructions and information.

---

## Prerequisites

* Anaconda (already installed)
* Node.js (v14 or later) + npm (v6 or later)
* Modern web browser with microphone access
* Homebrew (for ffmpeg — see setup below)

---

## One-Time Setup (Fresh Mac with Anaconda)

Run these steps once. After that, `npm start` is all you ever need.

### 1. Clone the repository

```bash
git clone https://github.com/Tali-22/TreeHouseDataCollection_modified.git
cd "TreeHouseDataCollection_modified/voice-to-text-app"
```

### 2. Install Node dependencies

```bash
npm install
```

### 3. Install Python dependencies

```bash
/opt/anaconda3/bin/pip install flask
/opt/anaconda3/bin/pip install soundfile
/opt/anaconda3/bin/pip install librosa
/opt/anaconda3/bin/pip install transformers
/opt/anaconda3/bin/pip install torch torchaudio
/opt/anaconda3/bin/pip install tensorflow
```

> **Note:** If you are only using one model type, you can skip irrelevant installs — e.g. skip `tensorflow` if only using Whisper, skip `torch` if only using Keras.

### 4. Install ffmpeg (required for audio conversion)

First check if Homebrew is installed:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Then install ffmpeg:

```bash
brew install ffmpeg
```

### 5. Model Files

The trained Whisper model files are included in this repository under `whisper_cmd_az/`. The folder contains:
- `config.json`
- `model.safetensors`
- `tokenizer.json`
- `generation_config.json`
- `processor_config.json`
- `tokenizer_config.json`
- `training_args.bin`

A second model (`my_voice_model`) is also included in the repository root for comparison testing.

> **Note:** Checkpoint folders (`checkpoint-250`, `checkpoint-300`) are excluded from this repository as they exceed GitHub's file size limits. These are intermediate training snapshots and are not needed to run the app.

---

## Running the App

```bash
npm start
```

This single command starts everything automatically:
* The Python model server (loads the model into memory once at startup)
* Frontend at: **http://localhost:3000**
* Backend API at: **http://localhost:3001**

> The model server is designed to keep the model cached in memory — so switching models in the GUI hot-swaps to the new one instantly without restarting the server.

---

## File Structure

```
TreeHouseDataCollection_modified/
├── whisper_cmd_az/               # Fine-tuned Whisper model files
├── my_voice_model/               # Additional voice model for comparison
├── voice-to-text-app/            # Main application
│   ├── public/                   # Static files
│   ├── server/                   # Backend server code
│   │   └── index.js              # Express server with upload, predict, and model management endpoints
│   ├── src/                      # Frontend source code
│   │   ├── components/           # React components
│   │   │   ├── Home.js           # Landing page with student ID input
│   │   │   ├── RecordingScreen.js# Recording + prediction display
│   │   │   ├── LiveTestScreen.js # Audio preprocessing before sending to model
│   │   │   ├── ModelManager.js   # Model switching, comparison, and history panel
│   │   │   └── CompletionScreen.js # Session complete screen
│   │   ├── services/             # API services
│   │   │   └── recordingService.js # All API calls including model management
│   │   ├── App.js                # Main application component
│   │   └── index.js              # Application entry point
│   ├── model_server.py           # Persistent Python server — loads model once, serves all predictions
│   ├── predict.py                # Python script that runs Whisper inference
│   ├── uploads/                  # Saved audio recordings (auto-created)
│   ├── package.json              # Project dependencies and scripts
│   └── README.md                 # This file
└── Talitha Kummarikunta_UROP Report_iVESA.pdf  # Project report
```

---

## Usage

### Home Screen
- Enter the student's ID number in the input field
- The server status indicator shows if the backend is connected
- Click **Start Recording Session** to begin

### Recording Screen
- The current letter or word to say is displayed prominently
- The **Model Testing Panel** sits above the recording area with two tabs:
  - **Models** — view, switch, add, and remove model versions
  - **History** — see a log of all predictions with accuracy per model
- Click the **microphone button** to start recording (auto-stops after 5 seconds)
- After recording, the app displays:
  - A **green box** if the model correctly recognised what was said
  - A **red box** if the model heard something different
- Click **Redo Previous** to go back and re-record the previous item
- The session automatically advances through all 26 letters and 9 command words

### Completion Screen
- Displayed after all items have been recorded
- Click **Return to Home** to start a new session

---

## Model Testing Features

### Switching Models
In the **Models** tab of the Model Testing Panel:
- The currently active model is highlighted with an **ACTIVE** badge
- Click the swap icon next to any model to switch to it instantly — no restart needed
- Accuracy percentages are shown per model based on session history

### Adding a New Model Version
1. Click **Add New Model Version** in the Models tab
2. Enter a name (e.g. `whisper-kids-v2`) and the full path to the model folder on your Mac, or a HuggingFace model ID (e.g. `openai/whisper-tiny`)
3. Click **Add Model** — it appears in the list immediately

### Comparison Mode
1. Click the **Compare** button in the Model Testing Panel header
2. Assign **Model A** and **Model B** by clicking the A/B buttons next to each model
3. Record as normal — both models will analyse the same audio simultaneously
4. Results are shown side by side with correct/incorrect indicators for each model

### Prediction History
- The **History** tab shows a running log of all predictions in the current session
- Each entry shows: expected label, what the model heard, correct/incorrect, model name, and timestamp
- Accuracy percentages are calculated automatically per model
- Click **Clear History** to reset the log

---

## Built-in Models

Two models are included in this repository:

| Model | Location | Description |
|---|---|---|
| `whisper_cmd_az` | `/whisper_cmd_az/` | Custom trained model on Singaporean children's voices |
| `my_voice_model` | `/my_voice_model/` | Additional voice model for comparison testing |

A baseline model is also available via HuggingFace:

| Model | Description |
|---|---|
| `openai/whisper-tiny (Base)` | The original base Whisper model — useful as a baseline comparison |

---

## File Naming Convention

Recordings are saved in the `uploads/` folder as `.wav` files:

```
{studentId}_{runNumber}{letterOrWord}.wav
```

Example: Student ID `001` recording the letter `A` on run 1:

```
001_1A.wav
```

---

## Configuration

The model directory and Python path can be customised in `server/index.js`:

```
WHISPER_MODEL_DIR   Path to your model folder (default: ../whisper_cmd_az relative to repo root)
PYTHON_CMD          Python executable to use (default: /opt/anaconda3/bin/python3)
PREDICT_SCRIPT      Path to predict.py (default: project root)
```

---
