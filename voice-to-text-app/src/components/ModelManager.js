import React, { useState, useEffect, useCallback } from 'react';
import Box from '@mui/material/Box';
import Paper from '@mui/material/Paper';
import Typography from '@mui/material/Typography';
import Button from '@mui/material/Button';
import IconButton from '@mui/material/IconButton';
import TextField from '@mui/material/TextField';
import Chip from '@mui/material/Chip';
import Divider from '@mui/material/Divider';
import CircularProgress from '@mui/material/CircularProgress';
import Alert from '@mui/material/Alert';
import Tooltip from '@mui/material/Tooltip';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableContainer from '@mui/material/TableContainer';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import Dialog from '@mui/material/Dialog';
import DialogTitle from '@mui/material/DialogTitle';
import DialogContent from '@mui/material/DialogContent';
import DialogActions from '@mui/material/DialogActions';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import CancelIcon from '@mui/icons-material/Cancel';
import DeleteIcon from '@mui/icons-material/Delete';
import AddIcon from '@mui/icons-material/Add';
import SwapHorizIcon from '@mui/icons-material/SwapHoriz';
import HistoryIcon from '@mui/icons-material/History';
import ModelTrainingIcon from '@mui/icons-material/ModelTraining';
import CompareIcon from '@mui/icons-material/Compare';
import {
  getModels, addModel, switchModel, removeModel,
  getPredictionHistory, clearPredictionHistory
} from '../services/recordingService';

const ModelManager = ({ onModelChange, compareMode, onCompareModeChange, compareModelIds, onCompareModelChange }) => {
  const [models, setModels] = useState([]);
  const [activeModelId, setActiveModelId] = useState('default');
  const [history, setHistory] = useState([]);
  const [accuracyByModel, setAccuracyByModel] = useState({});
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [showAddDialog, setShowAddDialog] = useState(false);
  const [newModelName, setNewModelName] = useState('');
  const [newModelPath, setNewModelPath] = useState('');
  const [activeTab, setActiveTab] = useState('models'); // 'models' | 'history'

  const fetchModels = useCallback(async () => {
    try {
      const data = await getModels();
      setModels(data.models);
      setActiveModelId(data.activeModelId);
    } catch (err) {
      setError('Failed to load models.');
    }
  }, []);

  const fetchHistory = useCallback(async () => {
    try {
      const data = await getPredictionHistory(null, 100);
      setHistory(data.history || []);
      setAccuracyByModel(data.accuracyByModel || {});
    } catch (err) {
      setError('Failed to load history.');
    }
  }, []);

  useEffect(() => {
    fetchModels();
    fetchHistory();
  }, [fetchModels, fetchHistory]);

  const handleSwitchModel = async (modelId) => {
    setIsLoading(true);
    setError('');
    try {
      const model = await switchModel(modelId);
      setActiveModelId(modelId);
      setSuccess(`Switched to: ${model.name}`);
      if (onModelChange) onModelChange(model);
      setTimeout(() => setSuccess(''), 3000);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const handleAddModel = async () => {
    if (!newModelName.trim() || !newModelPath.trim()) {
      setError('Please fill in both name and path.');
      return;
    }
    setIsLoading(true);
    setError('');
    try {
      await addModel(newModelName.trim(), newModelPath.trim());
      await fetchModels();
      setShowAddDialog(false);
      setNewModelName('');
      setNewModelPath('');
      setSuccess('Model added successfully!');
      setTimeout(() => setSuccess(''), 3000);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const handleRemoveModel = async (modelId) => {
    setIsLoading(true);
    try {
      await removeModel(modelId);
      await fetchModels();
      setSuccess('Model removed.');
      setTimeout(() => setSuccess(''), 3000);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const handleClearHistory = async () => {
    try {
      await clearPredictionHistory();
      setHistory([]);
      setAccuracyByModel({});
      setSuccess('History cleared.');
      setTimeout(() => setSuccess(''), 3000);
    } catch (err) {
      setError(err.message);
    }
  };

  const getAccuracyColor = (accuracy) => {
    if (accuracy >= 80) return 'success';
    if (accuracy >= 60) return 'warning';
    return 'error';
  };

  return (
    <Paper elevation={2} sx={{ p: 2, mb: 3, border: '1px solid', borderColor: 'divider' }}>
      {/* Header */}
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <ModelTrainingIcon color="primary" />
          <Typography variant="h6" sx={{ fontWeight: 'bold' }}>Model Testing Panel</Typography>
        </Box>
        <Box sx={{ display: 'flex', gap: 1 }}>
          <Tooltip title={compareMode ? 'Exit comparison mode' : 'Compare two models side by side'}>
            <Button
              size="small"
              variant={compareMode ? 'contained' : 'outlined'}
              color="secondary"
              startIcon={<CompareIcon />}
              onClick={() => onCompareModeChange(!compareMode)}
            >
              {compareMode ? 'Comparing' : 'Compare'}
            </Button>
          </Tooltip>
        </Box>
      </Box>

      {/* Tabs */}
      <Box sx={{ display: 'flex', gap: 1, mb: 2 }}>
        <Button
          size="small"
          variant={activeTab === 'models' ? 'contained' : 'outlined'}
          startIcon={<ModelTrainingIcon />}
          onClick={() => setActiveTab('models')}
        >
          Models
        </Button>
        <Button
          size="small"
          variant={activeTab === 'history' ? 'contained' : 'outlined'}
          startIcon={<HistoryIcon />}
          onClick={() => { setActiveTab('history'); fetchHistory(); }}
        >
          History
        </Button>
      </Box>

      {error && <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError('')}>{error}</Alert>}
      {success && <Alert severity="success" sx={{ mb: 2 }}>{success}</Alert>}

      {/* MODELS TAB */}
      {activeTab === 'models' && (
        <Box>
          {/* Accuracy summary */}
          {Object.keys(accuracyByModel).length > 0 && (
            <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap', mb: 2 }}>
              {Object.entries(accuracyByModel).map(([mId, acc]) => {
                const model = models.find(m => m.id === mId);
                const pct = acc.total > 0 ? Math.round((acc.correct / acc.total) * 100) : 0;
                return (
                  <Chip
                    key={mId}
                    label={`${model?.name || mId}: ${pct}% (${acc.correct}/${acc.total})`}
                    color={getAccuracyColor(pct)}
                    size="small"
                    variant="outlined"
                  />
                );
              })}
            </Box>
          )}

          {/* Model list */}
          {models.map((model) => (
            <Box
              key={model.id}
              sx={{
                p: 1.5, mb: 1, borderRadius: 1, border: '1px solid',
                borderColor: activeModelId === model.id ? 'primary.main' : 'divider',
                bgcolor: activeModelId === model.id ? 'primary.50' : 'transparent',
                display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 1,
              }}
            >
              <Box sx={{ flex: 1, minWidth: 0 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <Typography variant="body2" sx={{ fontWeight: 'bold' }} noWrap>
                    {model.name}
                  </Typography>
                  {activeModelId === model.id && (
                    <Chip label="ACTIVE" size="small" color="primary" />
                  )}
                </Box>
                <Typography variant="caption" color="textSecondary" noWrap>
                  {model.path}
                </Typography>
                {accuracyByModel[model.id] && (
                  <Typography variant="caption" color={`${getAccuracyColor(Math.round((accuracyByModel[model.id].correct / accuracyByModel[model.id].total) * 100))}.main`}>
                    {' '} · Accuracy: {Math.round((accuracyByModel[model.id].correct / accuracyByModel[model.id].total) * 100)}%
                  </Typography>
                )}
              </Box>

              <Box sx={{ display: 'flex', gap: 0.5, flexShrink: 0 }}>
                {/* Compare mode: select A or B */}
                {compareMode && (
                  <>
                    <Button
                      size="small"
                      variant={compareModelIds?.a === model.id ? 'contained' : 'outlined'}
                      color="primary"
                      onClick={() => onCompareModelChange('a', model.id)}
                      sx={{ minWidth: 36, px: 1 }}
                    >
                      A
                    </Button>
                    <Button
                      size="small"
                      variant={compareModelIds?.b === model.id ? 'contained' : 'outlined'}
                      color="secondary"
                      onClick={() => onCompareModelChange('b', model.id)}
                      sx={{ minWidth: 36, px: 1 }}
                    >
                      B
                    </Button>
                  </>
                )}

                {/* Switch button (normal mode) */}
                {!compareMode && activeModelId !== model.id && (
                  <Tooltip title="Use this model">
                    <IconButton
                      size="small"
                      color="primary"
                      onClick={() => handleSwitchModel(model.id)}
                      disabled={isLoading}
                    >
                      <SwapHorizIcon fontSize="small" />
                    </IconButton>
                  </Tooltip>
                )}

                {/* Delete (only non-built-in models) */}
                {!model.isDefault && model.id !== 'base-whisper' && (
                  <Tooltip title="Remove model">
                    <IconButton
                      size="small"
                      color="error"
                      onClick={() => handleRemoveModel(model.id)}
                      disabled={isLoading}
                    >
                      <DeleteIcon fontSize="small" />
                    </IconButton>
                  </Tooltip>
                )}
              </Box>
            </Box>
          ))}

          <Button
            startIcon={<AddIcon />}
            variant="outlined"
            size="small"
            onClick={() => setShowAddDialog(true)}
            sx={{ mt: 1 }}
          >
            Add New Model Version
          </Button>

          {compareMode && (
            <Alert severity="info" sx={{ mt: 2 }} icon={<CompareIcon />}>
              Select <strong>A</strong> and <strong>B</strong> above, then record — both models will analyse the same audio simultaneously.
            </Alert>
          )}
        </Box>
      )}

      {/* HISTORY TAB */}
      {activeTab === 'history' && (
        <Box>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
            <Typography variant="body2" color="textSecondary">
              {history.length} recent predictions
            </Typography>
            <Button size="small" color="error" onClick={handleClearHistory}>
              Clear History
            </Button>
          </Box>

          {history.length === 0 ? (
            <Typography variant="body2" color="textSecondary" sx={{ textAlign: 'center', py: 3 }}>
              No predictions yet. Start recording to see results here!
            </Typography>
          ) : (
            <TableContainer sx={{ maxHeight: 300 }}>
              <Table size="small" stickyHeader>
                <TableHead>
                  <TableRow>
                    <TableCell>Expected</TableCell>
                    <TableCell>Heard</TableCell>
                    <TableCell>Result</TableCell>
                    <TableCell>Model</TableCell>
                    <TableCell>Time</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {history.map((entry) => (
                    <TableRow key={entry.id} sx={{ bgcolor: entry.isCorrect ? 'success.50' : entry.isCorrect === false ? 'error.50' : 'transparent' }}>
                      <TableCell><strong>{entry.expectedLabel || '—'}</strong></TableCell>
                      <TableCell><strong>{entry.prediction || '—'}</strong></TableCell>
                      <TableCell>
                        {entry.isCorrect === true && <CheckCircleIcon color="success" fontSize="small" />}
                        {entry.isCorrect === false && <CancelIcon color="error" fontSize="small" />}
                        {entry.isCorrect === null && '—'}
                      </TableCell>
                      <TableCell>
                        <Typography variant="caption" noWrap sx={{ maxWidth: 120, display: 'block' }}>
                          {entry.modelName}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Typography variant="caption">
                          {new Date(entry.timestamp).toLocaleTimeString()}
                        </Typography>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          )}
        </Box>
      )}

      {/* Add Model Dialog */}
      <Dialog open={showAddDialog} onClose={() => setShowAddDialog(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Add New Model Version</DialogTitle>
        <DialogContent>
          <Typography variant="body2" color="textSecondary" sx={{ mb: 2 }}>
            Enter a name and the full path to your model folder on this Mac, or a HuggingFace model ID (e.g. <code>openai/whisper-tiny</code>).
          </Typography>
          <TextField
            fullWidth label="Model Name" placeholder="e.g. whisper-kids-v2"
            value={newModelName} onChange={(e) => setNewModelName(e.target.value)}
            sx={{ mb: 2 }} autoFocus
          />
          <TextField
            fullWidth label="Model Path or HuggingFace ID"
            placeholder="/Users/you/Downloads/whisper_cmd_az_v2  or  openai/whisper-tiny"
            value={newModelPath} onChange={(e) => setNewModelPath(e.target.value)}
            helperText="For a local folder: paste the full path. For HuggingFace: type the repo ID."
          />
          {error && <Alert severity="error" sx={{ mt: 1 }}>{error}</Alert>}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => { setShowAddDialog(false); setError(''); }}>Cancel</Button>
          <Button onClick={handleAddModel} variant="contained" disabled={isLoading}>
            {isLoading ? <CircularProgress size={20} /> : 'Add Model'}
          </Button>
        </DialogActions>
      </Dialog>
    </Paper>
  );
};

export default ModelManager;