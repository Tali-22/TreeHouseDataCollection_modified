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
// 👇 Set this to wherever you downloaded your trained model folder
const MODEL_DIR = process.env.WHISPER_MODEL_DIR || path.join(process.env.HOME || process.env.USERPROFILE, 'Downloads', 'whisper_cmd_az');
// 👇 Python executable (use 'python' on Windows if 'python3' doesn't work)
const PYTHON_CMD = process.env.PYTHON_CMD || '/opt/anaconda3/bin/python3';
// 👇 Path to predict.py — put it next to server/index.js, or set this env var
const PREDICT_SCRIPT = process.env.PREDICT_SCRIPT || path.join(__dirname, '..', 'predict.py');
// ───────────────────────────────────────────────────────────────────────────────

// Utility functions
const ensureUploadsDir = (dir) => {
  if (!fs.existsSync(dir)){
    fs.mkdirSync(dir, { recursive: true });
  }
};

const isValidFilename = (filename) => {
  return filename.indexOf('\0') === -1 && !filename.includes('/') && !filename.includes('\\');
};

const deleteFile = (filename, directory) => {
    return new Promise((resolve, reject) => {
        const filePath = path.join(directory, filename);
        fs.unlink(filePath, (err) => {
            if (err) {
                if (err.code === 'ENOENT') {
                    console.log(`File not found, but operation successful: ${filename}`);
                    return resolve(true);
                }
                return reject(err);
            }
            resolve(true);
        });
    });
};

// Ensure uploads directory exists
const uploadsDir = path.join(__dirname, '../uploads');
ensureUploadsDir(uploadsDir);

// Create Express app
const app = express();

app.use(helmet());
app.use(cors());

const limiter = rateLimit({
  max: 200,
  windowMs: 15 * 60 * 1000,
  message: 'Too many requests from this IP, please try again in 15 minutes!'
});
app.use('/api', limiter);

app.use(express.json({ limit: '10kb' }));
app.use(express.urlencoded({ extended: true, limit: '10kb' }));
app.use(mongoSanitize());
app.use(xss());
app.use(hpp());

const PORT = process.env.PORT || 3001;
const MAX_FILE_SIZE = parseInt(process.env.MAX_FILE_SIZE) || 10 * 1024 * 1024; // 10MB

// ─── MULTER STORAGE ─────────────────────────────────────────────────────────
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, uploadsDir);
  },
  filename: (req, file, cb) => {
    const ext = path.extname(file.originalname).toLowerCase();
    const baseName = path.basename(file.originalname, ext);
  
    console.log(`[DEBUG] Testing baseName: "${baseName}"`);

    // ✅ UPDATED: Now accepts run number 1 only (single digit or any number)
    // Pattern: {studentId}_{runNumber}{LABEL}
    // e.g. 12345_1A, 12345_1DONE, 12345_1Tutorial
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
  if (file.mimetype.startsWith('audio/')) {
    cb(null, true);
  } else {
    cb(new Error('Only audio files are allowed'), false);
  }
};

const upload = multer({ 
  storage: storage,
  limits: { fileSize: MAX_FILE_SIZE },
  fileFilter: fileFilter
});

// ─── HELPER: Run Python prediction ──────────────────────────────────────────
const runPrediction = (audioFilePath) => {
  return new Promise((resolve, reject) => {
    console.log(`[PREDICT] Running: ${PYTHON_CMD} ${PREDICT_SCRIPT} "${audioFilePath}" "${MODEL_DIR}"`);
    
    const proc = spawn(PYTHON_CMD, [PREDICT_SCRIPT, audioFilePath, MODEL_DIR]);
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

// ─── API ROUTES ──────────────────────────────────────────────────────────────

app.get('/api/health', (req, res) => {
  res.status(200).json({ status: 'Server is running' });
});

// Upload route (unchanged behaviour)
app.post('/api/upload', upload.single('audio'), (req, res) => {
  if (!req.file) {
    return res.status(400).json({ status: 'error', message: 'No file uploaded.' });
  }
  res.status(200).json({ 
    status: 'success',
    message: 'File uploaded successfully',
    data: { filename: req.file.filename, size: req.file.size }
  });
});

// ✅ NEW: Upload + predict route (used by the voice-to-text screen)
app.post('/api/upload-and-predict', upload.single('audio'), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ status: 'error', message: 'No file uploaded.' });
  }

  const savedFilePath = path.join(uploadsDir, req.file.filename);

  try {
    const prediction = await runPrediction(savedFilePath);
    res.status(200).json({
      status: 'success',
      message: 'File uploaded and predicted successfully',
      data: {
        filename: req.file.filename,
        size: req.file.size,
        prediction: prediction   // e.g. "A", "DONE", "TUTORIAL"
      }
    });
  } catch (predErr) {
    console.error('[PREDICT ERROR]', predErr.message);
    // File was saved successfully even if prediction failed
    res.status(200).json({
      status: 'partial',
      message: 'File saved but prediction failed',
      data: {
        filename: req.file.filename,
        size: req.file.size,
        prediction: null,
        predictionError: predErr.message
      }
    });
  }
});

app.get('/api/recordings/:studentId', (req, res) => {
  const { studentId } = req.params;
  if (!/^\d+$/.test(studentId)) {
    return res.status(400).json({ status: 'error', message: 'Invalid student ID.' });
  }
  fs.readdir(uploadsDir, (err, files) => {
    if (err) {
      return res.status(500).json({ status: 'error', message: 'Cannot read recordings directory.' });
    }
    const studentRecordings = files.filter(file => file.startsWith(`${studentId}_`) && isValidFilename(file));
    res.status(200).json({ 
      status: 'success',
      data: { recordings: studentRecordings }
    });
  });
});

app.delete('/api/recordings/:filename', async (req, res) => {
  const { filename } = req.params;
  if (!isValidFilename(filename)) {
    return res.status(400).json({ status: 'error', message: 'Invalid filename.' });
  }
  try {
    await deleteFile(filename, uploadsDir);
    res.status(200).json({ status: 'success', message: 'File deleted.' });
  } catch (error) {
    res.status(500).json({ status: 'error', message: 'Error deleting file.' });
  }
});

// ─── ERROR HANDLING ──────────────────────────────────────────────────────────

app.all('*', (req, res) => {
  res.status(404).json({
    status: 'error',
    message: `Can't find ${req.originalUrl} on this server!`
  });
});

app.use((err, req, res, next) => {
  console.error('ERROR 💥', err);
  if (err instanceof multer.MulterError) {
    if (err.code === 'LIMIT_FILE_SIZE') {
      return res.status(413).json({ status: 'error', message: `File too large. Max size is ${MAX_FILE_SIZE / (1024 * 1024)}MB.` });
    }
  }
  if (err.code === 'INVALID_FILENAME' || err.message === 'Invalid filename format') {
    return res.status(400).json({ status: 'error', message: 'Invalid filename format provided.' });
  }
  res.status(500).json({
    status: 'error',
    message: 'Something went wrong!'
  });
});

// ─── SERVER STARTUP ───────────────────────────────────────────────────────────

const server = app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
  console.log(`Model directory: ${MODEL_DIR}`);
  console.log(`Python command: ${PYTHON_CMD}`);
  console.log(`Predict script: ${PREDICT_SCRIPT}`);
});

process.on('unhandledRejection', (err) => {
  console.log('UNHANDLED REJECTION! 💥 Shutting down...');
  console.log(err.name, err.message);
  server.close(() => {
    process.exit(1);
  });
});

module.exports = app;