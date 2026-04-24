import React, { useState, useRef, useEffect, useCallback } from 'react';
import Button from '@mui/material/Button';
import Container from '@mui/material/Container';
import Typography from '@mui/material/Typography';
import Box from '@mui/material/Box';
import CircularProgress from '@mui/material/CircularProgress';
import Paper from '@mui/material/Paper';
import IconButton from '@mui/material/IconButton';
import Snackbar from '@mui/material/Snackbar';
import Alert from '@mui/material/Alert';
import Divider from '@mui/material/Divider';
import LinearProgress from '@mui/material/LinearProgress';
import Chip from '@mui/material/Chip';
import Grid from '@mui/material/Grid';
import MicIcon from '@mui/icons-material/Mic';
import MicOffIcon from '@mui/icons-material/MicOff';
import ReplayIcon from '@mui/icons-material/Replay';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import CancelIcon from '@mui/icons-material/Cancel';
import HourglassEmptyIcon from '@mui/icons-material/HourglassEmpty';
import { uploadAndPredict, uploadAndCompare } from '../services/recordingService';
import ModelManager from './ModelManager';
import Recorder from 'recorder-js';

const RECORDING_TIME_LIMIT = 4584;

const RecordingScreen = ({ studentId, itemToRecord, currentRun, totalRuns, progress, onRecordingSaved, onRedo }) => {
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isPredicting, setIsPredicting] = useState(false);
  const [error, setError] = useState('');
  const [info, setInfo] = useState('');

  // Prediction results
  const [currentPrediction, setCurrentPrediction] = useState(null);
  const [predictionReady, setPredictionReady] = useState(false);
  const [comparisonResult, setComparisonResult] = useState(null); // for compare mode

  // Model manager state
  const [compareMode, setCompareMode] = useState(false);
  const [compareModelIds, setCompareModelIds] = useState({ a: 'default', b: 'base-whisper' });

  const [audioStream, setAudioStream] = useState(null);
  const recorderInstanceRef = useRef(null);
  const recordingTimerRef = useRef(null);

  // Reset prediction when item changes
  useEffect(() => {
    setCurrentPrediction(null);
    setPredictionReady(false);
    setComparisonResult(null);
  }, [itemToRecord]);

  useEffect(() => {
    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
    recorderInstanceRef.current = new Recorder(audioContext, {});
  }, []);

  const stopMicrophone = useCallback(() => {
    if (audioStream) {
      audioStream.getTracks().forEach(track => track.stop());
      setAudioStream(null);
    }
  }, [audioStream]);

  const stopRecording = useCallback(async () => {
    if (recordingTimerRef.current) clearTimeout(recordingTimerRef.current);
    const recorder = recorderInstanceRef.current;
    if (!recorder || !isRecording) return;

    setIsProcessing(true);
    setInfo('Saving recording...');

    try {
      const { blob } = await recorder.stop();
      setIsRecording(false);
      stopMicrophone();

      setIsPredicting(true);

      if (compareMode) {
        // ── COMPARISON MODE ──
        setInfo('Running both models simultaneously...');
        const result = await uploadAndCompare(
          blob, studentId, itemToRecord, currentRun,
          compareModelIds.a, compareModelIds.b
        );
        setComparisonResult(result);
        setPredictionReady(true);
        setIsPredicting(false);
        setInfo('Comparison complete!');
        // Advance after 3 seconds so user can read both results
        setTimeout(() => onRecordingSaved(result?.modelA?.prediction), 3000);
      } else {
        // ── SINGLE MODEL MODE ──
        setInfo('Analysing with Whisper model...');
        const result = await uploadAndPredict(blob, studentId, itemToRecord, currentRun);
        setCurrentPrediction(result);
        setPredictionReady(true);
        setIsPredicting(false);
        setInfo(result.prediction ? `Model heard: "${result.prediction}"` : 'Saved! (prediction unavailable)');
        setTimeout(() => onRecordingSaved(result.prediction), 2000);
      }

    } catch (err) {
      console.error('Error:', err);
      setError('Failed to process or save recording.');
      stopMicrophone();
      setIsPredicting(false);
    } finally {
      setIsProcessing(false);
    }
  }, [isRecording, studentId, itemToRecord, currentRun, onRecordingSaved, stopMicrophone, compareMode, compareModelIds]);

  const startRecording = useCallback(async () => {
    if (isRecording) return;
    stopMicrophone();
    setIsProcessing(true);
    setError('');
    setCurrentPrediction(null);
    setComparisonResult(null);
    setPredictionReady(false);

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      setAudioStream(stream);
      const recorder = recorderInstanceRef.current;
      await recorder.init(stream);
      await recorder.start();
      setIsRecording(true);
      setIsProcessing(false);
      setInfo('Recording in progress...');
      recordingTimerRef.current = setTimeout(stopRecording, RECORDING_TIME_LIMIT);
    } catch (err) {
      setError(err.message || 'Microphone access denied.');
      setIsProcessing(false);
      stopMicrophone();
    }
  }, [isRecording, stopRecording, stopMicrophone]);

  const handleRedoClick = () => { onRedo(); };
  const handleCloseError = useCallback(() => setError(''), []);
  const handleCloseInfo = useCallback(() => setInfo(''), []);

  useEffect(() => {
    return () => {
      if (recordingTimerRef.current) clearTimeout(recordingTimerRef.current);
      stopMicrophone();
    };
  }, [stopMicrophone]);

  // ── Single model prediction result ──
  const renderSinglePrediction = () => {
    if (!predictionReady && !isPredicting) return null;

    if (isPredicting) {
      return (
        <Box sx={{ mt: 3, p: 2, borderRadius: 2, bgcolor: 'grey.100', textAlign: 'center' }}>
          <HourglassEmptyIcon sx={{ fontSize: 32, color: 'text.secondary' }} />
          <Typography variant="body1" color="textSecondary" sx={{ mt: 1 }}>
            Analysing with Whisper model...
          </Typography>
          <CircularProgress size={24} sx={{ mt: 1 }} />
        </Box>
      );
    }

    if (!currentPrediction?.prediction) {
      return (
        <Box sx={{ mt: 3, p: 2, borderRadius: 2, bgcolor: 'warning.light', textAlign: 'center' }}>
          <Typography variant="body1">Model prediction unavailable. Recording saved.</Typography>
        </Box>
      );
    }

    const isCorrect = currentPrediction.isCorrect;
    return (
      <Box sx={{
        mt: 3, p: 3, borderRadius: 2, textAlign: 'center',
        bgcolor: isCorrect ? 'success.light' : 'error.light',
        border: '2px solid', borderColor: isCorrect ? 'success.main' : 'error.main',
      }}>
        <Typography variant="overline" sx={{ color: isCorrect ? 'success.dark' : 'error.dark', fontWeight: 'bold' }}>
          {currentPrediction.modelName} heard
        </Typography>
        <Typography variant="h2" sx={{ fontFamily: 'monospace', fontWeight: 'bold', my: 1, color: isCorrect ? 'success.dark' : 'error.dark' }}>
          {currentPrediction.prediction}
        </Typography>
        <Chip
          icon={isCorrect ? <CheckCircleIcon /> : <CancelIcon />}
          label={isCorrect ? 'Correct!' : `Expected: ${itemToRecord}`}
          color={isCorrect ? 'success' : 'error'}
          variant="filled"
          sx={{ fontWeight: 'bold', fontSize: '0.9rem' }}
        />
      </Box>
    );
  };

  // ── Comparison mode result ──
  const renderComparisonResult = () => {
    if (!predictionReady && !isPredicting) return null;

    if (isPredicting) {
      return (
        <Box sx={{ mt: 3, p: 2, borderRadius: 2, bgcolor: 'grey.100', textAlign: 'center' }}>
          <HourglassEmptyIcon sx={{ fontSize: 32, color: 'text.secondary' }} />
          <Typography variant="body1" color="textSecondary" sx={{ mt: 1 }}>
            Running both models simultaneously...
          </Typography>
          <CircularProgress size={24} sx={{ mt: 1 }} />
        </Box>
      );
    }

    if (!comparisonResult) return null;

    const { modelA, modelB } = comparisonResult;

    const renderModelBox = (model, label, color) => {
      if (!model) return null;
      const isCorrect = model.isCorrect;
      return (
        <Box sx={{
          p: 2, borderRadius: 2, textAlign: 'center',
          bgcolor: isCorrect ? 'success.light' : 'error.light',
          border: '2px solid', borderColor: isCorrect ? 'success.main' : 'error.main',
          flex: 1,
        }}>
          <Chip label={label} color={color} size="small" sx={{ mb: 1, fontWeight: 'bold' }} />
          <Typography variant="caption" display="block" color="textSecondary" noWrap>
            {model.name}
          </Typography>
          <Typography variant="h3" sx={{ fontFamily: 'monospace', fontWeight: 'bold', my: 1, color: isCorrect ? 'success.dark' : 'error.dark' }}>
            {model.prediction || '?'}
          </Typography>
          <Chip
            icon={isCorrect ? <CheckCircleIcon /> : <CancelIcon />}
            label={isCorrect ? 'Correct!' : `Expected: ${itemToRecord}`}
            color={isCorrect ? 'success' : 'error'}
            size="small"
            variant="filled"
          />
        </Box>
      );
    };

    return (
      <Box sx={{ mt: 3 }}>
        <Typography variant="overline" color="textSecondary" display="block" textAlign="center" sx={{ mb: 1 }}>
          Model Comparison Results
        </Typography>
        <Box sx={{ display: 'flex', gap: 2 }}>
          {renderModelBox(modelA, 'Model A', 'primary')}
          {renderModelBox(modelB, 'Model B', 'secondary')}
        </Box>
      </Box>
    );
  };

  const isDisabled = isProcessing || isPredicting;

  return (
    <Container maxWidth="md" sx={{ mt: 4, mb: 6 }}>
      <Snackbar open={!!error} autoHideDuration={6000} onClose={handleCloseError} anchorOrigin={{ vertical: 'top', horizontal: 'center' }} sx={{ mt: 6 }}>
        <Alert onClose={handleCloseError} severity="error" elevation={6} variant="filled">{error}</Alert>
      </Snackbar>
      <Snackbar open={!!info} autoHideDuration={4000} onClose={handleCloseInfo} anchorOrigin={{ vertical: 'top', horizontal: 'center' }} sx={{ mt: 6 }}>
        <Alert onClose={handleCloseInfo} severity="info" elevation={6} variant="filled">{info}</Alert>
      </Snackbar>

      {/* ── Model Manager Panel ── */}
      <ModelManager
        onModelChange={(model) => setInfo(`Now using: ${model.name}`)}
        compareMode={compareMode}
        onCompareModeChange={setCompareMode}
        compareModelIds={compareModelIds}
        onCompareModelChange={(slot, modelId) => setCompareModelIds(prev => ({ ...prev, [slot]: modelId }))}
      />

      {/* ── Recording Panel ── */}
      <Paper elevation={3} sx={{ p: { xs: 2, md: 4 }, mb: 4 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Typography variant="h5" component="h1" sx={{ fontWeight: 'medium' }}>
            {compareMode ? 'Voice to Text — Comparison Mode' : 'Voice to Text'}
          </Typography>
          {compareMode && <Chip label="COMPARE MODE" color="secondary" size="small" />}
        </Box>
        <Typography variant="body2" color="textSecondary" sx={{ mb: 2 }}>Student ID: {studentId}</Typography>
        <Divider />

        <Box sx={{ textAlign: 'center', my: 4, minHeight: 200, display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
          {isProcessing && !isPredicting ? (
            <Box>
              <CircularProgress size={60} />
              <Typography variant="h6" sx={{ mt: 2 }}>Processing...</Typography>
            </Box>
          ) : (
            <>
              <Typography variant="h1" component="div" sx={{ fontSize: { xs: '4rem', sm: '6rem' }, fontWeight: 'bold', color: 'primary.main', fontFamily: 'monospace' }}>
                {itemToRecord}
              </Typography>

              <Box sx={{ mt: 3 }}>
                <IconButton
                  onClick={isRecording ? stopRecording : startRecording}
                  color={isRecording ? 'error' : 'primary'}
                  disabled={isDisabled}
                  sx={{ width: 80, height: 80, border: '2px solid' }}
                >
                  {isRecording ? <MicOffIcon sx={{ fontSize: 40 }} /> : <MicIcon sx={{ fontSize: 40 }} />}
                </IconButton>
                <Typography variant="h6" sx={{ mt: 1 }}>
                  {isRecording ? 'Recording... (tap to stop)' : 'Tap to Record'}
                </Typography>
              </Box>

              {/* Prediction results */}
              {compareMode ? renderComparisonResult() : renderSinglePrediction()}
            </>
          )}
        </Box>

        <Divider />
        <Box sx={{ mt: 3 }}>
          <Typography variant="body2">Overall Progress: {Math.round(progress)}%</Typography>
          <LinearProgress variant="determinate" value={progress} sx={{ height: 8, borderRadius: 4, mt: 1 }} />
        </Box>
      </Paper>

      <Box sx={{ display: 'flex', gap: 2, mb: 4 }}>
        <Button
          variant="outlined"
          onClick={handleRedoClick}
          disabled={isDisabled || isRecording || (currentRun === 1 && itemToRecord === 'A')}
          startIcon={<ReplayIcon />}
          size="large"
        >
          Redo Previous
        </Button>
      </Box>
    </Container>
  );
};

export default RecordingScreen;