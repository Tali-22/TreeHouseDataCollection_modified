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
import MicIcon from '@mui/icons-material/Mic';
import MicOffIcon from '@mui/icons-material/MicOff';
import ReplayIcon from '@mui/icons-material/Replay';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import CancelIcon from '@mui/icons-material/Cancel';
import HourglassEmptyIcon from '@mui/icons-material/HourglassEmpty';
import { uploadAndPredict } from '../services/recordingService';
import Recorder from 'recorder-js';

// Constants
const RECORDING_TIME_LIMIT = 5000; // 5 seconds

const RecordingScreen = ({ studentId, itemToRecord, currentRun, totalRuns, progress, onRecordingSaved, onRedo, lastPrediction }) => {
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isPredicting, setIsPredicting] = useState(false);
  const [error, setError] = useState('');
  const [info, setInfo] = useState('');

  // ✅ Local prediction result for the CURRENT recording (shown until next item)
  const [currentPrediction, setCurrentPrediction] = useState(null);
  const [predictionReady, setPredictionReady] = useState(false);

  const [audioStream, setAudioStream] = useState(null);
  const recorderInstanceRef = useRef(null);
  const recordingTimerRef = useRef(null);

  // Reset prediction display when item changes
  useEffect(() => {
    setCurrentPrediction(null);
    setPredictionReady(false);
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

      // ✅ Now call upload-and-predict instead of plain upload
      setIsPredicting(true);
      setInfo('Analysing with Whisper model...');

      const result = await uploadAndPredict(blob, studentId, itemToRecord, currentRun);

      const prediction = result.prediction;
      setCurrentPrediction(prediction);
      setPredictionReady(true);
      setIsPredicting(false);

      if (prediction) {
        setInfo(`Saved! Model heard: "${prediction}"`);
      } else {
        setInfo('Saved! (Model prediction unavailable)');
      }

    // Wait 2 seconds so user can see the prediction before moving on
      setTimeout(() => {
        onRecordingSaved(prediction);
      }, 2000);

    } catch (err) {
      console.error('Error stopping/saving recording:', err);
      setError('Failed to process or save recording.');
      stopMicrophone();
      setIsPredicting(false);
    } finally {
      setIsProcessing(false);
    }
  }, [isRecording, studentId, itemToRecord, currentRun, onRecordingSaved, stopMicrophone]);

  const startRecording = useCallback(async () => {
    if (isRecording) return;
    stopMicrophone();
    setIsProcessing(true);
    setError('');
    setCurrentPrediction(null);
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
      console.error('Error starting recording:', err);
      setError(err.message || 'Microphone access denied or failed to start.');
      setIsProcessing(false);
      stopMicrophone();
    }
  }, [isRecording, stopRecording, stopMicrophone]);

  const handleRedoClick = () => { onRedo(); };
  const handleCloseError = useCallback(() => { setError(''); }, []);
  const handleCloseInfo = useCallback(() => { setInfo(''); }, []);

  useEffect(() => {
    return () => {
      if (recordingTimerRef.current) clearTimeout(recordingTimerRef.current);
      stopMicrophone();
    };
  }, [stopMicrophone]);

  // ✅ Helper: render the prediction result box
  const renderPredictionResult = () => {
    if (!predictionReady && !isPredicting) return null;

    if (isPredicting) {
      return (
        <Box sx={{ mt: 3, p: 2, borderRadius: 2, bgcolor: 'grey.100', textAlign: 'center' }}>
          <HourglassEmptyIcon sx={{ fontSize: 32, color: 'text.secondary', mb: 1 }} />
          <Typography variant="body1" color="textSecondary">
            Analysing with Whisper model...
          </Typography>
          <CircularProgress size={24} sx={{ mt: 1 }} />
        </Box>
      );
    }

    if (!currentPrediction) {
      return (
        <Box sx={{ mt: 3, p: 2, borderRadius: 2, bgcolor: 'warning.light', textAlign: 'center' }}>
          <Typography variant="body1" color="warning.contrastText">
            Model prediction unavailable. Recording saved.
          </Typography>
        </Box>
      );
    }

    const isCorrect = currentPrediction.toUpperCase() === itemToRecord.toUpperCase();

    return (
      <Box
        sx={{
          mt: 3, p: 3, borderRadius: 2, textAlign: 'center',
          bgcolor: isCorrect ? 'success.light' : 'error.light',
          border: '2px solid',
          borderColor: isCorrect ? 'success.main' : 'error.main',
        }}
      >
        <Typography variant="overline" sx={{ color: isCorrect ? 'success.dark' : 'error.dark', fontWeight: 'bold' }}>
          Whisper Model Heard
        </Typography>
        <Typography
          variant="h2"
          component="div"
          sx={{
            fontFamily: 'monospace', fontWeight: 'bold', my: 1,
            color: isCorrect ? 'success.dark' : 'error.dark',
          }}
        >
          {currentPrediction}
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

  return (
    <Container maxWidth="md" sx={{ mt: 4, mb: 6 }}>
      <Snackbar open={!!error} autoHideDuration={6000} onClose={handleCloseError} anchorOrigin={{ vertical: 'top', horizontal: 'center' }} sx={{ mt: 6 }}>
        <Alert onClose={handleCloseError} severity="error" elevation={6} variant="filled" sx={{ width: '100%' }}>{error}</Alert>
      </Snackbar>
      <Snackbar open={!!info} autoHideDuration={4000} onClose={handleCloseInfo} anchorOrigin={{ vertical: 'top', horizontal: 'center' }} sx={{ mt: 6 }}>
        <Alert onClose={handleCloseInfo} severity="info" elevation={6} variant="filled" sx={{ width: '100%' }}>{info}</Alert>
      </Snackbar>

      <Paper elevation={3} sx={{ p: { xs: 2, md: 4 }, mb: 4 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Typography variant="h5" component="h1" sx={{ fontWeight: 'medium' }}>Voice to Text</Typography>
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
              {/* The letter/word to say */}
              <Typography
                variant="h1"
                component="div"
                sx={{ fontSize: { xs: '4rem', sm: '6rem' }, fontWeight: 'bold', color: 'primary.main', fontFamily: 'monospace' }}
              >
                {itemToRecord}
              </Typography>

              {/* Mic button */}
              <Box sx={{ mt: 3 }}>
                <IconButton
                  onClick={isRecording ? stopRecording : startRecording}
                  color={isRecording ? 'error' : 'primary'}
                  disabled={isProcessing || isPredicting}
                  sx={{ width: 80, height: 80, border: '2px solid' }}
                >
                  {isRecording ? <MicOffIcon sx={{ fontSize: 40 }} /> : <MicIcon sx={{ fontSize: 40 }} />}
                </IconButton>
                <Typography variant="h6" sx={{ mt: 1 }}>
                  {isRecording ? 'Recording... (tap to stop)' : 'Tap to Record'}
                </Typography>
              </Box>

              {/* ✅ Prediction result shown here */}
              {renderPredictionResult()}
            </>
          )}
        </Box>

        <Divider />
        <Box sx={{ mt: 3 }}>
          <Typography variant="body2">Overall Progress: {Math.round(progress)}%</Typography>
          <LinearProgress variant="determinate" value={progress} sx={{ height: 8, borderRadius: 4, mt: 1 }} />
        </Box>
      </Paper>

      <Box sx={{ display: 'flex', justifyContent: 'flex-start', flexWrap: 'wrap', gap: 2, mb: 4 }}>
        <Button
          variant="outlined"
          onClick={handleRedoClick}
          disabled={isRecording || isProcessing || isPredicting || (currentRun === 1 && itemToRecord === 'A')}
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