const express = require('express');
const multer = require('multer');
const cors = require('cors');
const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');
const rateLimit = require('express-rate-limit');
const helmet = require('helmet');
const mongoSanitize = require('express-mongo-sanitize');
const xss = require('xss-clean');
const hpp = require('hpp');

// ─── CONFIG ────────────────────────────────────────────────────────────────────
const DEFAULT_MODEL_DIR = process.env.WHISPER_MODEL_DIR || path.join(process.env.HOME || process.env.USERPROFILE, 'Downloads', 'whisper_cmd_az');
const PYTHON_CMD        = process.env.PYTHON_CMD        || '/opt/anaconda3/bin/python3';
const PREDICT_SCRIPT    = process.env.PREDICT_SCRIPT    || path.join(__dirname, '..', 'predict.py');
const MODEL_SERVER_URL  = process.env.MODEL_SERVER_URL  || 'http://localhost:5001';  // ← Flask model server
// ───────────────────────────────────────────────────────────────────────────────

// ─── MODEL REGISTRY ───────────────────────────────────────────────────────────
let modelRegistry = {
  activeModelId: 'default',
  models: {
    'default': {
      id: 'default',
      name: 'whisper-singaporean-kids (Fine-tuned)',
      path: DEFAULT_MODEL_DIR,
      addedAt: new Date().toISOString(),
      isDefault: true,
    },
    'base-whisper': {
      id: 'base-whisper',
      name: 'openai/whisper-tiny (Base)',
      path: 'openai/whisper-tiny',
      addedAt: new Date().toISOString(),
      isDefault: false,
    }
  }
};

// ─── PREDICTION HISTORY ───────────────────────────────────────────────────────
let predictionHistory = [];

// ─── UTILITY FUNCTIONS ────────────────────────────────────────────────────────
const ensureUploadsDir = (dir) => {
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
};

const isValidFilename = (filename) => {
  return filename.indexOf('\0') === -1 && !filename.includes('/') && !filename.includes('\\');
};

const deleteFile = (filename, directory) => {
  return new Promise((resolve, reject) => {
    const filePath = path.join(directory, filename);
    fs.unlink(filePath, (err) => {
      if (err) {
        if (err.code === 'ENOENT') return resolve(true);
        return reject(err);
      }
      resolve(true);
    });
  });
};

const uploadsDir = path.join(__dirname, '../uploads');
ensureUploadsDir(uploadsDir);

// ─── EXPRESS SETUP ────────────────────────────────────────────────────────────
const app = express();
app.use(helmet());
app.use(cors());

const limiter = rateLimit({
  max: 500,
  windowMs: 15 * 60 * 1000,
  message: 'Too many requests from this IP, please try again in 15 minutes!'
});
app.use('/api', limiter);

app.use(express.json({ limit: '10kb' }));
app.use(express.urlencoded({ extended: true, limit: '10kb' }));
app.use(mongoSanitize());
app.use(xss());
app.use(hpp());

const PORT         = process.env.PORT         || 3001;
const MAX_FILE_SIZE = parseInt(process.env.MAX_FILE_SIZE) || 10 * 1024 * 1024;

// ─── MULTER STORAGE ───────────────────────────────────────────────────────────
const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, uploadsDir),
  filename: (req, file, cb) => {
    const ext      = path.extname(file.originalname).toLowerCase();
    const baseName = path.basename(file.originalname, ext);
    console.log(`[DEBUG] Testing baseName: "${baseName}"`);
    const expectedPattern = /^\d+_\d+(?:[A-Z]|Done|Enter|Delete|Repeat|Backspace|Again|Undo|Tutorial|Screening)$/i;
    if (expectedPattern.test(baseName) && (ext === '.wav' || ext === '.webm')) {
      cb(null, file.originalname);
    } else {
      const err = new Error('Invalid filename format');
      err.code = 'INVALID_FILENAME';
      cb(err);
    }
  }
});

const fileFilter = (req, file, cb) => {
  if (file.mimetype.startsWith('audio/')) cb(null, true);
  else cb(new Error('Only audio files are allowed'), false);
};

const upload = multer({ storage, limits: { fileSize: MAX_FILE_SIZE }, fileFilter });


// ─── PYTHON PREDICTION HELPER (fallback — spawns predict.py directly) ─────────
// Used when the Flask model server is not running.
const runPredictScript = (audioFilePath, modelPath) => {
  return new Promise((resolve, reject) => {
    console.log(`[PREDICT] Fallback: spawning predict.py for model: ${modelPath}`);
    const proc = spawn(PYTHON_CMD, [PREDICT_SCRIPT, audioFilePath, modelPath]);
    let stdout = '';
    let stderr = '';

    proc.stdout.on('data', (data) => { stdout += data.toString(); });
    proc.stderr.on('data', (data) => { stderr += data.toString(); });

    proc.on('close', (code) => {
      if (code !== 0) {
        console.error(`[PREDICT] Python exited with code ${code}: ${stderr}`);
        return reject(new Error(`Prediction failed: ${stderr || 'Unknown error'}`));
      }
      const result = stdout.trim();
      console.log(`[PREDICT] Result: "${result}"`);
      resolve(result);
    });

    proc.on('error', (err) => {
      console.error(`[PREDICT] Failed to spawn process: ${err.message}`);
      reject(new Error(`Could not start prediction process: ${err.message}`));
    });
  });
};


// ─── FLASK MODEL SERVER HELPERS ───────────────────────────────────────────────

// Check if the Flask model server is running
const isModelServerRunning = async () => {
  try {
    const res = await fetch(`${MODEL_SERVER_URL}/health`, { signal: AbortSignal.timeout(500) });
    return res.ok;
  } catch {
    return false;
  }
};

// Ask the Flask server to switch to a different model (no restart needed)
const switchModelServer = async (modelPath) => {
  try {
    const res = await fetch(`${MODEL_SERVER_URL}/switch_model`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model_dir: modelPath }),
      signal: AbortSignal.timeout(3000),
    });
    const data = await res.json();
    console.log(`[MODEL SERVER] Switched to: ${data.model_type} at ${modelPath}`);
    return true;
  } catch (err) {
    console.warn(`[MODEL SERVER] Could not switch model on server: ${err.message}`);
    return false;
  }
};

// ─── MAIN PREDICTION FUNCTION ─────────────────────────────────────────────────
// Tries Flask model server first (fast, model pre-loaded).
// Falls back to spawning predict.py if the server isn't running.
const runPrediction = async (audioFilePath, modelPath) => {
  console.log(`[PREDICT] Running with model: ${modelPath}`);

  const serverRunning = await isModelServerRunning();

  if (serverRunning) {
    try {
      const res = await fetch(`${MODEL_SERVER_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ audio_path: audioFilePath }),
        signal: AbortSignal.timeout(10000),  // 10s max per prediction
      });
      const data = await res.json();
      if (data.error) throw new Error(data.error);
      console.log(`[PREDICT] Server result: "${data.label}"`);
      return data.label;
    } catch (err) {
      console.warn(`[PREDICT] Server call failed, falling back to script: ${err.message}`);
    }
  } else {
    console.log('[PREDICT] Model server not running — using predict.py fallback');
  }

  // Fallback: spawn predict.py directly (slower but always works)
  return runPredictScript(audioFilePath, modelPath);
};


// ─── API ROUTES ───────────────────────────────────────────────────────────────

// Health check
app.get('/api/health', (req, res) => {
  res.status(200).json({ status: 'Server is running' });
});

// Model server status (lets the frontend show whether fast mode is active)
app.get('/api/model-server/status', async (req, res) => {
  const running = await isModelServerRunning();
  res.status(200).json({
    status: 'success',
    data: {
      running,
      url: MODEL_SERVER_URL,
      message: running
        ? 'Fast model server is active — predictions will be near-instant.'
        : 'Model server not running — using fallback (slower). Run model_server.py to enable fast mode.',
    }
  });
});

// ── Model Management Routes ──

app.get('/api/models', (req, res) => {
  res.status(200).json({
    status: 'success',
    data: {
      activeModelId: modelRegistry.activeModelId,
      models: Object.values(modelRegistry.models),
    }
  });
});

app.post('/api/models', (req, res) => {
  const { name, path: modelPath } = req.body;

  if (!name || !modelPath) {
    return res.status(400).json({ status: 'error', message: 'Model name and path are required.' });
  }

  const isHuggingFaceId = modelPath.includes('/') && !modelPath.startsWith('/');
  if (!isHuggingFaceId && !fs.existsSync(modelPath)) {
    return res.status(400).json({ status: 'error', message: `Model path does not exist: ${modelPath}` });
  }

  const id = `model_${Date.now()}`;
  modelRegistry.models[id] = {
    id,
    name,
    path: modelPath,
    addedAt: new Date().toISOString(),
    isDefault: false,
  };

  console.log(`[MODEL] Added new model: ${name} at ${modelPath}`);
  res.status(201).json({ status: 'success', data: { model: modelRegistry.models[id] } });
});

// PUT switch active model — also tells Flask server to switch
app.put('/api/models/active', async (req, res) => {
  const { modelId } = req.body;

  if (!modelId || !modelRegistry.models[modelId]) {
    return res.status(400).json({ status: 'error', message: 'Invalid model ID.' });
  }

  modelRegistry.activeModelId = modelId;
  const activeModel = modelRegistry.models[modelId];
  console.log(`[MODEL] Switched to: ${activeModel.name}`);

  // Notify Flask server to hot-swap the model (non-blocking)
  switchModelServer(activeModel.path).catch(() => {});

  res.status(200).json({
    status: 'success',
    message: `Switched to model: ${activeModel.name}`,
    data: { activeModel }
  });
});

app.delete('/api/models/:modelId', (req, res) => {
  const { modelId } = req.params;

  if (modelId === 'default' || modelId === 'base-whisper') {
    return res.status(400).json({ status: 'error', message: 'Cannot delete built-in models.' });
  }

  if (!modelRegistry.models[modelId]) {
    return res.status(404).json({ status: 'error', message: 'Model not found.' });
  }

  if (modelRegistry.activeModelId === modelId) {
    modelRegistry.activeModelId = 'default';
  }

  delete modelRegistry.models[modelId];
  res.status(200).json({ status: 'success', message: 'Model removed from registry.' });
});

// ── Prediction History Routes ──

app.get('/api/history', (req, res) => {
  const { modelId, limit = 50 } = req.query;
  let history = predictionHistory;

  if (modelId) {
    history = history.filter(h => h.modelId === modelId);
  }

  history = history.slice(-parseInt(limit)).reverse();

  const accuracyByModel = {};
  predictionHistory.forEach(h => {
    if (!accuracyByModel[h.modelId]) {
      accuracyByModel[h.modelId] = { correct: 0, total: 0 };
    }
    accuracyByModel[h.modelId].total++;
    if (h.isCorrect) accuracyByModel[h.modelId].correct++;
  });

  res.status(200).json({
    status: 'success',
    data: { history, accuracyByModel, totalPredictions: predictionHistory.length }
  });
});

app.delete('/api/history', (req, res) => {
  predictionHistory = [];
  res.status(200).json({ status: 'success', message: 'History cleared.' });
});

// ── Upload Routes ──

app.post('/api/upload', upload.single('audio'), (req, res) => {
  if (!req.file) return res.status(400).json({ status: 'error', message: 'No file uploaded.' });
  res.status(200).json({
    status: 'success',
    message: 'File uploaded successfully',
    data: { filename: req.file.filename, size: req.file.size }
  });
});

// Upload + predict using active model (or specified model)
app.post('/api/upload-and-predict', upload.single('audio'), async (req, res) => {
  if (!req.file) return res.status(400).json({ status: 'error', message: 'No file uploaded.' });

  const savedFilePath = path.join(uploadsDir, req.file.filename);
  const modelId       = req.body.modelId || modelRegistry.activeModelId;
  const model         = modelRegistry.models[modelId] || modelRegistry.models[modelRegistry.activeModelId];
  const expectedLabel = req.body.expectedLabel || null;

  try {
    const prediction = await runPrediction(savedFilePath, model.path);
    const isCorrect  = expectedLabel ? prediction.toUpperCase() === expectedLabel.toUpperCase() : null;

    const historyEntry = {
      id: `pred_${Date.now()}`,
      timestamp: new Date().toISOString(),
      filename: req.file.filename,
      expectedLabel,
      prediction,
      isCorrect,
      modelId: model.id,
      modelName: model.name,
    };
    predictionHistory.push(historyEntry);

    res.status(200).json({
      status: 'success',
      data: {
        filename: req.file.filename,
        size: req.file.size,
        prediction,
        isCorrect,
        modelId: model.id,
        modelName: model.name,
      }
    });
  } catch (predErr) {
    console.error('[PREDICT ERROR]', predErr.message);
    res.status(200).json({
      status: 'partial',
      message: 'File saved but prediction failed',
      data: {
        filename: req.file.filename,
        size: req.file.size,
        prediction: null,
        predictionError: predErr.message,
        modelId: model.id,
        modelName: model.name,
      }
    });
  }
});

// Upload + predict using TWO models simultaneously (comparison mode)
app.post('/api/upload-and-compare', upload.single('audio'), async (req, res) => {
  if (!req.file) return res.status(400).json({ status: 'error', message: 'No file uploaded.' });

  const savedFilePath            = path.join(uploadsDir, req.file.filename);
  const { modelIdA, modelIdB, expectedLabel } = req.body;

  const modelA = modelRegistry.models[modelIdA] || modelRegistry.models['default'];
  const modelB = modelRegistry.models[modelIdB] || modelRegistry.models['base-whisper'];

  try {
    const [predictionA, predictionB] = await Promise.allSettled([
      runPrediction(savedFilePath, modelA.path),
      runPrediction(savedFilePath, modelB.path),
    ]);

    const resultA = predictionA.status === 'fulfilled' ? predictionA.value : null;
    const resultB = predictionB.status === 'fulfilled' ? predictionB.value : null;

    if (resultA) {
      predictionHistory.push({
        id: `pred_${Date.now()}_a`,
        timestamp: new Date().toISOString(),
        filename: req.file.filename,
        expectedLabel,
        prediction: resultA,
        isCorrect: expectedLabel ? resultA.toUpperCase() === expectedLabel.toUpperCase() : null,
        modelId: modelA.id,
        modelName: modelA.name,
        comparisonMode: true,
      });
    }

    if (resultB) {
      predictionHistory.push({
        id: `pred_${Date.now()}_b`,
        timestamp: new Date().toISOString(),
        filename: req.file.filename,
        expectedLabel,
        prediction: resultB,
        isCorrect: expectedLabel ? resultB.toUpperCase() === expectedLabel.toUpperCase() : null,
        modelId: modelB.id,
        modelName: modelB.name,
        comparisonMode: true,
      });
    }

    res.status(200).json({
      status: 'success',
      data: {
        filename: req.file.filename,
        expectedLabel,
        modelA: { id: modelA.id, name: modelA.name, prediction: resultA, isCorrect: expectedLabel ? resultA?.toUpperCase() === expectedLabel.toUpperCase() : null },
        modelB: { id: modelB.id, name: modelB.name, prediction: resultB, isCorrect: expectedLabel ? resultB?.toUpperCase() === expectedLabel.toUpperCase() : null },
      }
    });
  } catch (err) {
    res.status(500).json({ status: 'error', message: err.message });
  }
});

// Recordings management
app.get('/api/recordings/:studentId', (req, res) => {
  const { studentId } = req.params;
  if (!/^\d+$/.test(studentId)) return res.status(400).json({ status: 'error', message: 'Invalid student ID.' });
  fs.readdir(uploadsDir, (err, files) => {
    if (err) return res.status(500).json({ status: 'error', message: 'Cannot read recordings directory.' });
    const studentRecordings = files.filter(file => file.startsWith(`${studentId}_`) && isValidFilename(file));
    res.status(200).json({ status: 'success', data: { recordings: studentRecordings } });
  });
});

app.delete('/api/recordings/:filename', async (req, res) => {
  const { filename } = req.params;
  if (!isValidFilename(filename)) return res.status(400).json({ status: 'error', message: 'Invalid filename.' });
  try {
    await deleteFile(filename, uploadsDir);
    res.status(200).json({ status: 'success', message: 'File deleted.' });
  } catch (error) {
    res.status(500).json({ status: 'error', message: 'Error deleting file.' });
  }
});

// ─── ERROR HANDLING ───────────────────────────────────────────────────────────
app.all('*', (req, res) => {
  res.status(404).json({ status: 'error', message: `Can't find ${req.originalUrl} on this server!` });
});

app.use((err, req, res, next) => {
  console.error('ERROR 💥', err);
  if (err instanceof multer.MulterError) {
    if (err.code === 'LIMIT_FILE_SIZE') return res.status(413).json({ status: 'error', message: 'File too large.' });
  }
  if (err.code === 'INVALID_FILENAME' || err.message === 'Invalid filename format') {
    return res.status(400).json({ status: 'error', message: 'Invalid filename format provided.' });
  }
  res.status(500).json({ status: 'error', message: 'Something went wrong!' });
});

// ─── SERVER STARTUP ───────────────────────────────────────────────────────────
const server = app.listen(PORT, async () => {
  console.log(`Server running on port ${PORT}`);
  console.log(`Active model: ${modelRegistry.models[modelRegistry.activeModelId].name}`);
  console.log(`Python command: ${PYTHON_CMD}`);
  console.log(`Predict script: ${PREDICT_SCRIPT}`);

  // Check if Flask model server is already running
  const serverRunning = await isModelServerRunning();
  if (serverRunning) {
    console.log(`[MODEL SERVER] ✅ Fast model server detected at ${MODEL_SERVER_URL}`);
    // Sync the active model to the server on startup
    const activeModel = modelRegistry.models[modelRegistry.activeModelId];
    switchModelServer(activeModel.path).catch(() => {});
  } else {
    console.log(`[MODEL SERVER] ⚠️  No fast model server detected — using predict.py fallback`);
    console.log(`[MODEL SERVER]    To enable fast mode, run in a separate terminal:`);
    console.log(`[MODEL SERVER]    python model_server.py <model_dir>`);
  }
});

process.on('unhandledRejection', (err) => {
  console.log('UNHANDLED REJECTION! 💥 Shutting down...');
  server.close(() => process.exit(1));
});

module.exports = app;