const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:3001';
const API_TIMEOUT = 60000; // 60 seconds — longer for model comparison

// ─── RECORDING UPLOAD ─────────────────────────────────────────────────────────

/**
 * Upload a recording and get a prediction from the active model
 */
const uploadAndPredict = async (file, studentId, item, repetition) => {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), API_TIMEOUT);

  const formData = new FormData();
  const filename = `${studentId}_${repetition}${item}.wav`;
  formData.append('audio', file, filename);
  formData.append('expectedLabel', item); // So server can calculate isCorrect

  try {
    const response = await fetch(`${API_BASE_URL}/api/upload-and-predict`, {
      method: 'POST',
      body: formData,
      signal: controller.signal,
    });

    clearTimeout(timeoutId);
    const data = await response.json();

    if (!response.ok) throw new Error(data.message || `Upload failed: ${response.status}`);

    return {
      success: true,
      filename: data.data.filename,
      prediction: data.data.prediction || null,
      isCorrect: data.data.isCorrect,
      modelId: data.data.modelId,
      modelName: data.data.modelName,
    };
  } catch (error) {
    clearTimeout(timeoutId);
    if (error.name === 'AbortError') throw new Error('Upload timed out. Please check your connection.');
    throw new Error(error.message || 'An error occurred while uploading.');
  }
};

/**
 * Upload a recording and compare TWO models side by side
 */
const uploadAndCompare = async (file, studentId, item, repetition, modelIdA, modelIdB) => {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), API_TIMEOUT);

  const formData = new FormData();
  const filename = `${studentId}_${repetition}${item}.wav`;
  formData.append('audio', file, filename);
  formData.append('expectedLabel', item);
  formData.append('modelIdA', modelIdA);
  formData.append('modelIdB', modelIdB);

  try {
    const response = await fetch(`${API_BASE_URL}/api/upload-and-compare`, {
      method: 'POST',
      body: formData,
      signal: controller.signal,
    });

    clearTimeout(timeoutId);
    const data = await response.json();

    if (!response.ok) throw new Error(data.message || `Comparison failed: ${response.status}`);

    return {
      success: true,
      filename: data.data.filename,
      expectedLabel: data.data.expectedLabel,
      modelA: data.data.modelA,
      modelB: data.data.modelB,
    };
  } catch (error) {
    clearTimeout(timeoutId);
    if (error.name === 'AbortError') throw new Error('Request timed out.');
    throw new Error(error.message || 'An error occurred during comparison.');
  }
};

/**
 * Original upload without prediction (kept for compatibility)
 */
const uploadRecording = async (file, studentId, item, repetition) => {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), API_TIMEOUT);

  const formData = new FormData();
  const filename = `${studentId}_${repetition}${item}.wav`;
  formData.append('audio', file, filename);

  try {
    const response = await fetch(`${API_BASE_URL}/api/upload`, {
      method: 'POST',
      body: formData,
      signal: controller.signal,
    });

    clearTimeout(timeoutId);
    const data = await response.json();

    if (!response.ok) throw new Error(data.message || `Upload failed: ${response.status}`);

    return {
      success: true,
      data: { filename: data.data.filename, size: data.data.size, timestamp: new Date().toISOString() },
    };
  } catch (error) {
    clearTimeout(timeoutId);
    if (error.name === 'AbortError') throw new Error('Upload timed out.');
    throw new Error(error.message || 'An error occurred while uploading.');
  }
};

// ─── MODEL MANAGEMENT ─────────────────────────────────────────────────────────

/**
 * Get all registered models and the active model ID
 */
const getModels = async () => {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 5000);

  try {
    const response = await fetch(`${API_BASE_URL}/api/models`, {
      signal: controller.signal,
    });
    clearTimeout(timeoutId);
    const data = await response.json();
    if (!response.ok) throw new Error(data.message);
    return data.data;
  } catch (error) {
    clearTimeout(timeoutId);
    throw new Error(error.message || 'Failed to fetch models.');
  }
};

/**
 * Add a new model to the registry
 */
const addModel = async (name, modelPath) => {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 5000);

  try {
    const response = await fetch(`${API_BASE_URL}/api/models`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, path: modelPath }),
      signal: controller.signal,
    });
    clearTimeout(timeoutId);
    const data = await response.json();
    if (!response.ok) throw new Error(data.message);
    return data.data.model;
  } catch (error) {
    clearTimeout(timeoutId);
    throw new Error(error.message || 'Failed to add model.');
  }
};

/**
 * Switch the active model
 */
const switchModel = async (modelId) => {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 5000);

  try {
    const response = await fetch(`${API_BASE_URL}/api/models/active`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ modelId }),
      signal: controller.signal,
    });
    clearTimeout(timeoutId);
    const data = await response.json();
    if (!response.ok) throw new Error(data.message);
    return data.data.activeModel;
  } catch (error) {
    clearTimeout(timeoutId);
    throw new Error(error.message || 'Failed to switch model.');
  }
};

/**
 * Remove a model from the registry
 */
const removeModel = async (modelId) => {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 5000);

  try {
    const response = await fetch(`${API_BASE_URL}/api/models/${modelId}`, {
      method: 'DELETE',
      signal: controller.signal,
    });
    clearTimeout(timeoutId);
    const data = await response.json();
    if (!response.ok) throw new Error(data.message);
    return true;
  } catch (error) {
    clearTimeout(timeoutId);
    throw new Error(error.message || 'Failed to remove model.');
  }
};

// ─── PREDICTION HISTORY ───────────────────────────────────────────────────────

/**
 * Get prediction history (optionally filtered by modelId)
 */
const getPredictionHistory = async (modelId = null, limit = 50) => {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 5000);

  try {
    const params = new URLSearchParams({ limit });
    if (modelId) params.append('modelId', modelId);

    const response = await fetch(`${API_BASE_URL}/api/history?${params}`, {
      signal: controller.signal,
    });
    clearTimeout(timeoutId);
    const data = await response.json();
    if (!response.ok) throw new Error(data.message);
    return data.data;
  } catch (error) {
    clearTimeout(timeoutId);
    throw new Error(error.message || 'Failed to fetch history.');
  }
};

/**
 * Clear prediction history
 */
const clearPredictionHistory = async () => {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 5000);

  try {
    const response = await fetch(`${API_BASE_URL}/api/history`, {
      method: 'DELETE',
      signal: controller.signal,
    });
    clearTimeout(timeoutId);
    const data = await response.json();
    if (!response.ok) throw new Error(data.message);
    return true;
  } catch (error) {
    clearTimeout(timeoutId);
    throw new Error(error.message || 'Failed to clear history.');
  }
};

// ─── SERVER STATUS ────────────────────────────────────────────────────────────

const checkServerStatus = async () => {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 5000);

  try {
    const response = await fetch(`${API_BASE_URL}/api/health`, {
      signal: controller.signal,
      headers: { 'Cache-Control': 'no-cache' },
    });
    clearTimeout(timeoutId);
    if (!response.ok) return false;
    const data = await response.json();
    return data.status === 'Server is running';
  } catch (error) {
    clearTimeout(timeoutId);
    return false;
  }
};

const getRecordings = async (studentId) => {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 5000);

  try {
    if (!studentId || !/^\d+$/.test(studentId)) throw new Error('Invalid student ID format');
    const response = await fetch(`${API_BASE_URL}/api/recordings/${studentId}`, { signal: controller.signal });
    clearTimeout(timeoutId);
    const data = await response.json();
    if (!response.ok) throw new Error(data.message);
    return data.data.recordings || [];
  } catch (error) {
    clearTimeout(timeoutId);
    throw new Error(error.message || 'Failed to fetch recordings.');
  }
};

const deleteRecording = async (filename) => {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 5000);

  try {
    const response = await fetch(`${API_BASE_URL}/api/recordings/${encodeURIComponent(filename)}`, {
      method: 'DELETE',
      signal: controller.signal,
    });
    clearTimeout(timeoutId);
    const data = await response.json();
    if (!response.ok) throw new Error(data.message);
    return { success: true };
  } catch (error) {
    clearTimeout(timeoutId);
    throw new Error(error.message || 'Failed to delete recording.');
  }
};

export {
  uploadRecording,
  uploadAndPredict,
  uploadAndCompare,
  getModels,
  addModel,
  switchModel,
  removeModel,
  getPredictionHistory,
  clearPredictionHistory,
  checkServerStatus,
  getRecordings,
  deleteRecording,
};