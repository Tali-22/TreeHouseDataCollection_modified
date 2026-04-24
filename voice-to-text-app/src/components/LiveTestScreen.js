import React, { useState, useRef, useEffect, useCallback } from 'react';
import Container from '@mui/material/Container';
import Box from '@mui/material/Box';
import Paper from '@mui/material/Paper';
import Typography from '@mui/material/Typography';
import Button from '@mui/material/Button';
import IconButton from '@mui/material/IconButton';
import Chip from '@mui/material/Chip';
import Divider from '@mui/material/Divider';
import LinearProgress from '@mui/material/LinearProgress';
import CircularProgress from '@mui/material/CircularProgress';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableContainer from '@mui/material/TableContainer';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import Select from '@mui/material/Select';
import MenuItem from '@mui/material/MenuItem';
import FormControl from '@mui/material/FormControl';
import InputLabel from '@mui/material/InputLabel';
import Alert from '@mui/material/Alert';
import Tooltip from '@mui/material/Tooltip';
import MicIcon from '@mui/icons-material/Mic';
import MicOffIcon from '@mui/icons-material/MicOff';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import CancelIcon from '@mui/icons-material/Cancel';
import ShuffleIcon from '@mui/icons-material/Shuffle';
import RestartAltIcon from '@mui/icons-material/RestartAlt';
import SkipNextIcon from '@mui/icons-material/SkipNext';
import HourglassEmptyIcon from '@mui/icons-material/HourglassEmpty';
import Recorder from 'recorder-js';
import { getModels, uploadAndPredict } from '../services/recordingService';

// ── Vocabulary (keep in sync with ALLOWED in predict.py) ─────────────────────
const LETTERS       = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('');
const COMMAND_WORDS = ['DONE','DELETE','BACKSPACE','ENTER','REPEAT','AGAIN','UNDO','TUTORIAL','SCREENING'];
const ALL_WORDS     = [...LETTERS, ...COMMAND_WORDS];

const RECORDING_MS  = 4000;   // max recording length (matches training clip length)

// ── Helpers ───────────────────────────────────────────────────────────────────
const shuffle = (arr) => [...arr].sort(() => Math.random() - 0.5);

const pct = (n, d) => (d === 0 ? 0 : Math.round((n / d) * 100));

const STATUS = { IDLE: 'idle', RECORDING: 'recording', PROCESSING: 'processing', RESULT: 'result' };

// ─────────────────────────────────────────────────────────────────────────────
export default function LiveTestScreen() {
  // ── Model state ──
  const [models, setModels]           = useState([]);
  const [activeModelId, setActiveModelId] = useState('default');
  const [modelsLoading, setModelsLoading] = useState(true);

  // ── Test session state ──
  const [wordList, setWordList]       = useState(shuffle(ALL_WORDS));
  const [wordIndex, setWordIndex]     = useState(0);
  const [filter, setFilter]           = useState('all');   // 'all' | 'letters' | 'commands'
  const [status, setStatus]           = useState(STATUS.IDLE);
  const [history, setHistory]         = useState([]);      // [{word, predicted, correct, modelName}]
  const [lastResult, setLastResult]   = useState(null);    // most recent prediction result
  const [error, setError]             = useState('');

  // ── Audio ──
  const audioCtxRef   = useRef(null);
  const recorderRef   = useRef(null);
  const streamRef     = useRef(null);
  const timerRef      = useRef(null);

  // ── Computed ──
  const currentWord   = wordList[wordIndex] ?? null;
  const correct       = history.filter(h => h.correct).length;
  const accuracy      = pct(correct, history.length);

  const filteredWords = useCallback(() => {
    if (filter === 'letters')  return LETTERS;
    if (filter === 'commands') return COMMAND_WORDS;
    return ALL_WORDS;
  }, [filter]);

  // ── Load models on mount ──
  useEffect(() => {
    getModels()
      .then(data => {
        setModels(data.models || []);
        setActiveModelId(data.activeModelId || 'default');
      })
      .catch(() => setError('Could not load models from server.'))
      .finally(() => setModelsLoading(false));
  }, []);

  // ── Re-shuffle when filter changes ──
  useEffect(() => {
    setWordList(shuffle(filteredWords()));
    setWordIndex(0);
    setLastResult(null);
    setStatus(STATUS.IDLE);
  }, [filter, filteredWords]);

  // ── Init AudioContext once ──
  useEffect(() => {
    audioCtxRef.current = new (window.AudioContext || window.webkitAudioContext)();
    recorderRef.current = new Recorder(audioCtxRef.current, {});
    return () => stopMic();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ── Mic helpers ──
  const stopMic = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(t => t.stop());
      streamRef.current = null;
    }
  };

  // ── Stop recording → upload → predict ────────────────────────────────────
  const stopAndPredict = useCallback(async () => {
    if (timerRef.current) { clearTimeout(timerRef.current); timerRef.current = null; }
    const recorder = recorderRef.current;
    if (!recorder) return;

    setStatus(STATUS.PROCESSING);
    try {
      const { blob } = await recorder.stop();
      stopMic();

      // Use a synthetic student ID and repetition so the server saves the file
      const result = await uploadAndPredict(
        blob,
        'TEST',           // studentId
        currentWord,      // item (= expectedLabel)
        '1',              // repetition
        activeModelId,    // pass selected model
      );

      const entry = {
        word:      currentWord,
        predicted: result.prediction ?? '—',
        correct:   result.isCorrect ?? false,
        modelName: result.modelName ?? activeModelId,
      };

      setHistory(prev => [entry, ...prev]);
      setLastResult(entry);
      setStatus(STATUS.RESULT);

      // Auto-advance after 1.8 s
      timerRef.current = setTimeout(() => {
        setWordIndex(i => i + 1);
        setLastResult(null);
        setStatus(STATUS.IDLE);
      }, 1800);

    } catch (err) {
      setError(err.message || 'Prediction failed.');
      stopMic();
      setStatus(STATUS.IDLE);
    }
  }, [currentWord, activeModelId]);

  // ── Start recording ────────────────────────────────────────────────────────
  const startRecording = useCallback(async () => {
    if (status !== STATUS.IDLE) return;
    setError('');
    setLastResult(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
      streamRef.current = stream;
      const recorder = recorderRef.current;
      await recorder.init(stream);
      await recorder.start();
      setStatus(STATUS.RECORDING);
      // Auto-stop at RECORDING_MS to match training clip length
      timerRef.current = setTimeout(stopAndPredict, RECORDING_MS);
    } catch (err) {
      setError(err.message || 'Microphone access denied.');
    }
  }, [status, stopAndPredict]);

  // ── Skip word ──────────────────────────────────────────────────────────────
  const skipWord = () => {
    if (timerRef.current) clearTimeout(timerRef.current);
    stopMic();
    setWordIndex(i => i + 1);
    setLastResult(null);
    setStatus(STATUS.IDLE);
  };

  // ── Reset session ──────────────────────────────────────────────────────────
  const resetSession = () => {
    if (timerRef.current) clearTimeout(timerRef.current);
    stopMic();
    setHistory([]);
    setWordList(shuffle(filteredWords()));
    setWordIndex(0);
    setLastResult(null);
    setStatus(STATUS.IDLE);
  };

  // ── Reshuffle same word set ───────────────────────────────────────────────
  const reshuffleWords = () => {
    if (timerRef.current) clearTimeout(timerRef.current);
    stopMic();
    setWordList(shuffle(filteredWords()));
    setWordIndex(0);
    setLastResult(null);
    setStatus(STATUS.IDLE);
  };

  // ── Keyboard shortcut: Space = record/stop ────────────────────────────────
  useEffect(() => {
    const onKey = (e) => {
      if (e.code === 'Space' && e.target.tagName !== 'INPUT') {
        e.preventDefault();
        if (status === STATUS.IDLE)      startRecording();
        else if (status === STATUS.RECORDING) stopAndPredict();
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [status, startRecording, stopAndPredict]);

  // Cleanup on unmount
  useEffect(() => () => {
    if (timerRef.current) clearTimeout(timerRef.current);
    stopMic();
  }, []);

  // ── Session complete (all words attempted) ────────────────────────────────
  const sessionDone = wordIndex >= wordList.length;

  // ── Render ─────────────────────────────────────────────────────────────────
  return (
    <Container maxWidth="md" sx={{ mt: 4, mb: 8 }}>

      {/* ── Header ── */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Box>
          <Typography variant="h4" fontWeight="bold">🎙 Live Model Test</Typography>
          <Typography variant="body2" color="text.secondary">
            Say each word aloud — the model predicts it using the exact training pipeline.
          </Typography>
        </Box>
        <Chip
          label={`${accuracy}% accuracy  (${correct}/${history.length})`}
          color={accuracy >= 80 ? 'success' : accuracy >= 50 ? 'warning' : history.length === 0 ? 'default' : 'error'}
          sx={{ fontSize: '1rem', px: 1, py: 2.5, fontWeight: 'bold' }}
        />
      </Box>

      {error && (
        <Alert severity="error" onClose={() => setError('')} sx={{ mb: 2 }}>{error}</Alert>
      )}

      {/* ── Controls Bar ── */}
      <Paper elevation={1} sx={{ p: 2, mb: 3, display: 'flex', gap: 2, flexWrap: 'wrap', alignItems: 'center' }}>

        {/* Model selector */}
        <FormControl size="small" sx={{ minWidth: 220 }} disabled={modelsLoading}>
          <InputLabel>Model</InputLabel>
          <Select
            value={activeModelId}
            label="Model"
            onChange={e => setActiveModelId(e.target.value)}
          >
            {models.map(m => (
              <MenuItem key={m.id} value={m.id}>{m.name}</MenuItem>
            ))}
          </Select>
        </FormControl>

        {/* Word filter */}
        <FormControl size="small" sx={{ minWidth: 150 }}>
          <InputLabel>Words</InputLabel>
          <Select value={filter} label="Words" onChange={e => setFilter(e.target.value)}>
            <MenuItem value="all">All ({ALL_WORDS.length})</MenuItem>
            <MenuItem value="letters">Letters only (A–Z)</MenuItem>
            <MenuItem value="commands">Commands only</MenuItem>
          </Select>
        </FormControl>

        <Box sx={{ flexGrow: 1 }} />

        <Tooltip title="Reshuffle word order">
          <span>
            <Button
              variant="outlined"
              startIcon={<ShuffleIcon />}
              onClick={reshuffleWords}
              disabled={status !== STATUS.IDLE}
              size="small"
            >Shuffle</Button>
          </span>
        </Tooltip>

        <Tooltip title="Reset session — clear history and start over">
          <span>
            <Button
              variant="outlined"
              color="error"
              startIcon={<RestartAltIcon />}
              onClick={resetSession}
              disabled={status === STATUS.RECORDING}
              size="small"
            >Reset</Button>
          </span>
        </Tooltip>
      </Paper>

      {/* ── Progress bar ── */}
      <Box sx={{ mb: 2 }}>
        <Typography variant="caption" color="text.secondary">
          Word {Math.min(wordIndex + 1, wordList.length)} of {wordList.length}
        </Typography>
        <LinearProgress
          variant="determinate"
          value={pct(wordIndex, wordList.length)}
          sx={{ height: 6, borderRadius: 3, mt: 0.5 }}
        />
      </Box>

      {/* ── Main card ── */}
      <Paper elevation={3} sx={{ p: 4, mb: 3, textAlign: 'center', minHeight: 320 }}>

        {sessionDone ? (
          /* ── Session complete ── */
          <Box sx={{ py: 4 }}>
            <Typography variant="h4" gutterBottom>🏁 Session Complete!</Typography>
            <Typography variant="h2" fontWeight="bold" color={accuracy >= 80 ? 'success.main' : 'error.main'}>
              {accuracy}%
            </Typography>
            <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
              {correct} correct out of {history.length} words
            </Typography>
            <Button variant="contained" startIcon={<RestartAltIcon />} onClick={resetSession} size="large">
              Start New Session
            </Button>
          </Box>

        ) : status === STATUS.PROCESSING ? (
          /* ── Processing ── */
          <Box sx={{ py: 4 }}>
            <HourglassEmptyIcon sx={{ fontSize: 48, color: 'text.secondary' }} />
            <Typography variant="h6" color="text.secondary" sx={{ mt: 1 }}>Analysing…</Typography>
            <CircularProgress size={28} sx={{ mt: 2 }} />
          </Box>

        ) : status === STATUS.RESULT && lastResult ? (
          /* ── Result flash ── */
          <Box sx={{ py: 2 }}>
            <Typography variant="overline" color="text.secondary">You said "{lastResult.word}"</Typography>
            <Typography
              variant="h1"
              fontFamily="monospace"
              fontWeight="bold"
              color={lastResult.correct ? 'success.main' : 'error.main'}
              sx={{ fontSize: { xs: '4rem', sm: '7rem' }, lineHeight: 1, my: 1 }}
            >
              {lastResult.predicted}
            </Typography>
            <Chip
              icon={lastResult.correct ? <CheckCircleIcon /> : <CancelIcon />}
              label={lastResult.correct ? 'Correct!' : `Expected: ${lastResult.word}`}
              color={lastResult.correct ? 'success' : 'error'}
              sx={{ fontSize: '1rem', fontWeight: 'bold', px: 1 }}
            />
            <Typography variant="caption" display="block" color="text.secondary" sx={{ mt: 1 }}>
              Next word loading…
            </Typography>
          </Box>

        ) : (
          /* ── Ready to record ── */
          <Box sx={{ py: 1 }}>
            {/* Word display */}
            <Typography variant="overline" color="text.secondary">Say this word:</Typography>
            <Typography
              variant="h1"
              fontFamily="monospace"
              fontWeight="bold"
              color="primary.main"
              sx={{ fontSize: { xs: '4rem', sm: '7rem' }, lineHeight: 1, my: 2 }}
            >
              {currentWord}
            </Typography>

            {/* Upcoming words preview */}
            <Box sx={{ display: 'flex', justifyContent: 'center', gap: 0.5, mb: 3, flexWrap: 'wrap' }}>
              {wordList.slice(wordIndex + 1, wordIndex + 6).map((w, i) => (
                <Chip key={i} label={w} size="small" variant="outlined" sx={{ opacity: 0.5 - i * 0.08 }} />
              ))}
            </Box>

            {/* Record button */}
            <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', gap: 3 }}>
              <Tooltip title="Skip this word">
                <IconButton onClick={skipWord} color="default" sx={{ opacity: 0.6 }}>
                  <SkipNextIcon />
                </IconButton>
              </Tooltip>

              <Tooltip title={status === STATUS.RECORDING ? 'Tap to stop (or wait)' : 'Tap to record  [Space]'}>
                <IconButton
                  onClick={status === STATUS.RECORDING ? stopAndPredict : startRecording}
                  color={status === STATUS.RECORDING ? 'error' : 'primary'}
                  sx={{
                    width: 88, height: 88,
                    border: '3px solid',
                    borderColor: status === STATUS.RECORDING ? 'error.main' : 'primary.main',
                    animation: status === STATUS.RECORDING ? 'pulse 1s ease-in-out infinite' : 'none',
                    '@keyframes pulse': {
                      '0%,100%': { boxShadow: '0 0 0 0 rgba(211,47,47,0.4)' },
                      '50%':     { boxShadow: '0 0 0 12px rgba(211,47,47,0)' },
                    },
                  }}
                >
                  {status === STATUS.RECORDING
                    ? <MicOffIcon sx={{ fontSize: 44 }} />
                    : <MicIcon    sx={{ fontSize: 44 }} />}
                </IconButton>
              </Tooltip>

              <Box sx={{ width: 40 }} /> {/* balance skip button */}
            </Box>

            <Typography variant="caption" color="text.secondary" sx={{ mt: 2, display: 'block' }}>
              {status === STATUS.RECORDING
                ? `Recording… (auto-stops after ${RECORDING_MS / 1000} s)`
                : 'Press Space or tap the mic'}
            </Typography>
          </Box>
        )}
      </Paper>

      {/* ── Per-label accuracy breakdown ── */}
      {history.length > 0 && (
        <Paper elevation={1} sx={{ p: 2, mb: 3 }}>
          <Typography variant="subtitle2" fontWeight="bold" gutterBottom>Accuracy by label</Typography>
          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.75 }}>
            {Object.entries(
              history.reduce((acc, h) => {
                if (!acc[h.word]) acc[h.word] = { c: 0, t: 0 };
                acc[h.word].t++;
                if (h.correct) acc[h.word].c++;
                return acc;
              }, {})
            ).sort(([a], [b]) => a.localeCompare(b))
              .map(([word, { c, t }]) => (
                <Chip
                  key={word}
                  label={`${word}: ${c}/${t}`}
                  size="small"
                  color={c === t ? 'success' : c === 0 ? 'error' : 'warning'}
                  variant="filled"
                />
              ))}
          </Box>
        </Paper>
      )}

      {/* ── Session history table ── */}
      {history.length > 0 && (
        <Paper elevation={1}>
          <Box sx={{ p: 2, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <Typography variant="subtitle2" fontWeight="bold">
              Prediction history ({history.length} entries)
            </Typography>
          </Box>
          <Divider />
          <TableContainer sx={{ maxHeight: 320 }}>
            <Table size="small" stickyHeader>
              <TableHead>
                <TableRow>
                  <TableCell>#</TableCell>
                  <TableCell>Expected</TableCell>
                  <TableCell>Predicted</TableCell>
                  <TableCell>Result</TableCell>
                  <TableCell>Model</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {history.map((h, i) => (
                  <TableRow
                    key={i}
                    sx={{ bgcolor: h.correct ? 'success.50' : 'error.50' }}
                  >
                    <TableCell sx={{ color: 'text.secondary' }}>{history.length - i}</TableCell>
                    <TableCell>
                      <Typography fontFamily="monospace" fontWeight="bold">{h.word}</Typography>
                    </TableCell>
                    <TableCell>
                      <Typography
                        fontFamily="monospace"
                        fontWeight="bold"
                        color={h.correct ? 'success.main' : 'error.main'}
                      >
                        {h.predicted}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      {h.correct
                        ? <CheckCircleIcon color="success" fontSize="small" />
                        : <CancelIcon color="error" fontSize="small" />}
                    </TableCell>
                    <TableCell>
                      <Typography variant="caption" color="text.secondary" noWrap>{h.modelName}</Typography>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </Paper>
      )}
    </Container>
  );
}