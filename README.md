# TreeHouseDataCollection_modified
This is a modified code to make user-friendly testing of all the upcoming versions of voice recognition models. To setup first, make sure to install the previous whisper models. To update the voice recognition model, refer to the report for detailed instructions on how to update the code.
---

## Prerequisites

* Node.js (v14 or later)
* npm (v6 or later)
* Python 3.12 (via Anaconda recommended)
* Modern web browser with microphone access
* The trained Whisper model folder (`whisper_cmd_az`) downloaded to `~/Downloads/`

---

## Python Setup

Install the required Python packages using Anaconda:

```bash
conda install pytorch torchaudio -c pytorch
pip install transformers
```

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Tali-22/TreeHouseDataCollection_modified.git
cd TreeHouseDataCollection_modified/voice-to-text-app
```

### 2. Install dependencies

```bash
npm install
```

### 3. Download the Whisper model

You will need the trained Whisper model files. Contact the project owner to get access. Once downloaded, place the unzipped folder at:

```
~/Downloads/whisper_cmd_az/
```

The folder should contain files like `config.json`, `model.safetensors`, `tokenizer.json` etc.

### 4. Start the development servers

```bash
npm run dev
```

This will start:

* Frontend development server at: **http://localhost:3000**
* Backend API server at: **http://localhost:3001**

Alternatively, run them separately:

```bash
# Terminal 1 - Start the backend server
npm run server

# Terminal 2 - Start the frontend
npm start
```

---

## File Structure

```
voice-to-text-app/
├── public/                 # Static files
├── server/                 # Backend server code
│   └── index.js            # Express server with upload + predict endpoint
├── src/                    # Frontend source code
│   ├── components/         # React components
│   │   ├── Home.js         # Landing page with student ID input
│   │   ├── RecordingScreen.js  # Recording + prediction display
│   │   └── CompletionScreen.js # Session complete screen
│   ├── services/           # API services
│   │   └── recordingService.js # Upload and predict API calls
│   ├── App.js              # Main application component
│   └── index.js            # Application entry point
├── predict.py              # Python script that runs Whisper inference
├── uploads/                # Saved audio recordings (auto-created)
├── package.json            # Project dependencies and scripts
└── README.md               # This file
```

---

## Usage

### Home Screen
- Enter the student's ID number in the input field
- The server status indicator shows if the backend is connected
- Click **Start Recording Session** to begin

### Recording Screen
- The current letter or word to say is displayed prominently
- Click the **microphone button** to start recording (auto-stops after 5 seconds)
- After recording, the app sends the audio to the Whisper model and displays:
  - A **green box** if the model correctly recognised what was said
  - A **red box** if the model heard something different
- Click **Redo Previous** to go back and re-record the previous item
- The session automatically advances through all 26 letters and 9 command words

### Completion Screen
- Displayed after all items have been recorded
- Click **Return to Home** to start a new session

---

## File Naming Convention

Recordings are saved in the `uploads/` folder with this pattern:

```
{studentId}_{runNumber}{letterOrWord}.webm
```
If sucessful detection of recording by the model, another file is saved with the same pattern:
```
{studentId}_{runNumber}{letterOrWord}.wav
```

Example: Student ID `001` recording the letter `A`:

```
001_1A.webm
```

Successful recognition will upload the below file:

```
001_1A.wav
```
---

## Configuration

The model directory and Python path can be customised via environment variables in `server/index.js`:

```
WHISPER_MODEL_DIR   Path to your whisper_cmd_az folder (default: ~/Downloads/whisper_cmd_az)
PYTHON_CMD          Python executable to use (default: /opt/anaconda3/bin/python3)
PREDICT_SCRIPT      Path to predict.py (default: project root)
```
