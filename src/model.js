// model.js
// Model logic separated from main.js with enhanced persistency

// ML Model State
export let tfModel = null;
export let modelTrained = false;
export let modelTraining = false;
export let modelTrainingPromise = null;
export let MODEL_TOPICS = [];
export let MODEL_HASHTAGS = [];

// New storage for previous arrays
export let previousTopicsResults = [];  // Store previous topic analysis results
export let previousHashtagsResults = []; // Store previous hashtag analysis results
export const MAX_STORED_RESULTS = 5;    // Maximum number of previous results to store

// Preference weighting constants
export const WEIGHT_LIKED = 3.0;
export const WEIGHT_INTERESTED = 2.0;
export const WEIGHT_NOT_INTERESTED = -4.0;
export const WEIGHT_COMMENTED = 1.5;
export const MIN_INTERACTIONS = 10;

// Load model with improved architecture
export async function loadModel(force = false) {
  try {
    if (modelTraining && modelTrainingPromise) {
      await modelTrainingPromise;
    }
    if (!tfModel || force) {
      if (MODEL_TOPICS.length === 0) {
        console.log('❌ No topics available, cannot create model');
        tfModel = null;
        modelTrained = false;
        return;
      }
      
      // Check if tf is available
      if (typeof tf === 'undefined') {
        throw new Error('TensorFlow.js not loaded. Make sure tf is available globally.');
      }
      
      console.log(`🏗️ Creating model with ${MODEL_TOPICS.length} input features...`);
      
      // Store the current training state before recreating the model
      const wasModelTrained = modelTrained && !force;
      
      if (tfModel) {
        try { 
          tfModel.dispose(); 
          console.log('🗑️ Disposed previous model');
        } catch (e) { 
          console.log('⚠️ Error disposing model:', e);
        }
      }
      
      // Create model architecture
      tfModel = tf.sequential();
      
      const inputUnits = Math.max(16, MODEL_TOPICS.length);
      console.log(`Adding layers: input(${MODEL_TOPICS.length}) -> dense(${inputUnits}) -> dense(8) -> output(3)`);
      
      tfModel.add(tf.layers.dense({
        inputShape: [MODEL_TOPICS.length],
        units: inputUnits,
        activation: "relu",
        kernelRegularizer: tf.regularizers.l2({ l2: 0.001 })
      }));
      
      tfModel.add(tf.layers.dense({
        units: 8,
        activation: "relu",
        kernelRegularizer: tf.regularizers.l2({ l2: 0.001 })
      }));
      
      tfModel.add(tf.layers.dense({
        units: 3,
        activation: "softmax"
      }));
      
      // Compile the model
      const optimizer = tf.train.adam(0.001);
      tfModel.compile({
        optimizer: optimizer,
        loss: "categoricalCrossentropy",
        metrics: ["accuracy"]
      });
      
      console.log(`✅ Model created and compiled successfully with ${tfModel.layers.length} layers`);
      
      // Note: tfModel.compiled property may not be reliable in all TensorFlow.js versions
      // The successful compilation is confirmed by reaching this point without errors
      
      // Only reset modelTrained if we're forcing a reload (during training)
      // If we're just checking/loading existing model, preserve training state
      if (force) {
        modelTrained = false;
      } else {
        modelTrained = wasModelTrained;
      }
      
      console.log(`Model state: layers=${tfModel.layers.length}, trained=${modelTrained}`);
    }
  } catch (error) {
    console.error('❌ Error in loadModel:', error);
    tfModel = null;
    modelTrained = false;
    throw error; // Re-throw to handle in calling function
  }
}

// Store results to persistent storage
export function storeResults(topics, hashtags) {
  // Add current results to the storage arrays
  if (topics && topics.length > 0) {
    previousTopicsResults.push([...topics]);
    // Keep only the last MAX_STORED_RESULTS
    if (previousTopicsResults.length > MAX_STORED_RESULTS) {
      previousTopicsResults.shift();
    }
  }
  
  if (hashtags && hashtags.length > 0) {
    previousHashtagsResults.push([...hashtags]);
    // Keep only the last MAX_STORED_RESULTS
    if (previousHashtagsResults.length > MAX_STORED_RESULTS) {
      previousHashtagsResults.shift();
    }
  }
  
  // Optional: Store to localStorage for persistence between sessions
  try {
    localStorage.setItem('modelPreviousTopics', JSON.stringify(previousTopicsResults));
    localStorage.setItem('modelPreviousHashtags', JSON.stringify(previousHashtagsResults));
  } catch (e) {
    console.warn('Could not save model results to localStorage', e);
  }
}

// Load previously stored results
export function loadStoredResults() {
  try {
    const storedTopics = localStorage.getItem('modelPreviousTopics');
    const storedHashtags = localStorage.getItem('modelPreviousHashtags');
    
    if (storedTopics) {
      previousTopicsResults = JSON.parse(storedTopics);
    }
    
    if (storedHashtags) {
      previousHashtagsResults = JSON.parse(storedHashtags);
    }
  } catch (e) {
    console.warn('Could not load stored model results', e);
  }
}

// Export model to downloads folder with improved user feedback
export async function exportModelToDownloads() {
  if (!tfModel || !modelTrained) {
    const message = 'No trained model to export. Train the model first by interacting with posts!';
    console.warn(message);
    alert(message);
    return false;
  }

  try {
    // Create a timestamp for the filename
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5);
    const modelName = `recommender-model-${timestamp}`;
    
    // Create model metadata
    const metadata = {
      exportTime: new Date().toISOString(),
      modelTopics: [...MODEL_TOPICS],
      modelHashtags: [...MODEL_HASHTAGS],
      previousTopicsResults: [...previousTopicsResults],
      previousHashtagsResults: [...previousHashtagsResults],
      modelArchitecture: {
        inputShape: tfModel.layers[0].inputShape,
        layers: tfModel.layers.map(layer => ({
          type: layer.constructor.name,
          units: layer.units || null,
          activation: layer.activation?.name || null
        }))
      }
    };

    // Show user what's happening
    console.log(`🚀 Exporting model as ${modelName}...`);
    alert(`🚀 Exporting model! Check your Downloads folder for:\n• ${modelName}.json (model)\n• ${modelName}.weights.bin (weights)\n• ${modelName}-metadata.json (metadata)`);
    
    // Save the model to downloads (triggers browser download)
    await tfModel.save(`downloads://${modelName}`);
    
    // Also save metadata as a JSON file
    const metadataBlob = new Blob([JSON.stringify(metadata, null, 2)], { type: 'application/json' });
    const metadataUrl = URL.createObjectURL(metadataBlob);
    const metadataLink = document.createElement('a');
    metadataLink.href = metadataUrl;
    metadataLink.download = `${modelName}-metadata.json`;
    metadataLink.style.display = 'none';
    document.body.appendChild(metadataLink);
    metadataLink.click();
    
    // Clean up
    setTimeout(() => {
      document.body.removeChild(metadataLink);
      URL.revokeObjectURL(metadataUrl);
    }, 100);

    const successMessage = `✅ Model exported successfully!\n\nFiles saved to Downloads:\n• ${modelName}.json\n• ${modelName}.weights.bin\n• ${modelName}-metadata.json`;
    console.log(successMessage);
    alert(successMessage);
    return true;
  } catch (error) {
    console.error('❌ Failed to export model:', error);
    
    // Try alternative export method if downloads:// fails
    try {
      console.log('Attempting alternative export method...');
      const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5);
      const modelName = `recommender-model-${timestamp}`;
      
      // Use IndexedDB as fallback
      await tfModel.save(`indexeddb://${modelName}`);
      const fallbackMessage = `✅ Model saved to browser storage as fallback: ${modelName}`;
      console.log(fallbackMessage);
      alert(fallbackMessage);
      return true;
    } catch (fallbackError) {
      const errorMessage = `❌ Export failed completely: ${fallbackError.message}`;
      console.error(errorMessage);
      alert(errorMessage);
      return false;
    }
  }
}

// Manual export function for testing
export function manualExportModel() {
  console.log('🔧 Manual export requested...');
  console.log(`Model status: ${modelTrained ? 'Trained' : 'Not trained'}`);
  console.log(`Topics available: ${MODEL_TOPICS.length}`);
  console.log(`Hashtags available: ${MODEL_HASHTAGS.length}`);
  
  if (modelTrained) {
    return exportModelToDownloads();
  } else {
    const message = `❌ Cannot export: Model not trained yet!\n\nStatus:\n• Topics: ${MODEL_TOPICS.length}\n• Hashtags: ${MODEL_HASHTAGS.length}\n• Model: ${modelTrained ? 'Trained' : 'Not trained'}\n\nInteract with posts to train the model first!`;
    console.log(message);
    alert(message);
    return false;
  }
}

// Debug function to check TensorFlow availability
export function checkTensorFlowStatus() {
  const status = {
    tfAvailable: typeof tf !== 'undefined',
    tfVersion: typeof tf !== 'undefined' ? tf.version.tfjs : 'N/A',
    modelExists: !!tfModel,
    modelLayers: tfModel?.layers?.length || 0,
    modelIsCompiled: tfModel?.layers?.length > 0 && tfModel.optimizer != null,
    modelTrained,
    topicsCount: MODEL_TOPICS.length,
    hashtagsCount: MODEL_HASHTAGS.length
  };
  
  console.log('🔍 TensorFlow Status:', status);
  return status;
}

// Force train model with dummy data for testing
export async function forceTrainModelForTesting() {
  console.log('🧪 Force training model with dummy data for testing...');
  
  // Ensure we have some topics and hashtags
  if (MODEL_TOPICS.length === 0) {
    MODEL_TOPICS.push('technology', 'ai', 'science', 'programming', 'javascript', 'web', 'mobile', 'data');
    console.log('📝 Added topics:', MODEL_TOPICS);
  }
  if (MODEL_HASHTAGS.length === 0) {
    MODEL_HASHTAGS.push('#tech', '#ai', '#science', '#programming', '#js', '#web', '#mobile', '#data');
    console.log('📝 Added hashtags:', MODEL_HASHTAGS);
  }
  
  // Create dummy interactions
  const dummyInteractions = [];
  for (let i = 0; i < 15; i++) {
    const topicSelection = MODEL_TOPICS.slice(0, Math.floor(Math.random() * 3) + 1);
    const hashtagSelection = MODEL_HASHTAGS.slice(0, Math.floor(Math.random() * 2) + 1);
    
    const interaction = {
      postId: `dummy-${i}`,
      topics: topicSelection,
      hashtags: hashtagSelection,
      liked: Math.random() > 0.7,
      interested: Math.random() > 0.6,
      not_interested: Math.random() > 0.8,
      commented: Math.random() > 0.9,
      timeSpentMs: Math.floor(Math.random() * 10000) + 1000,
      timestamp: Date.now() + i * 1000
    };
    
    dummyInteractions.push(interaction);
  }
  
  console.log(`Created ${dummyInteractions.length} dummy interactions`);
  console.log('Sample interaction:', dummyInteractions[0]);
  
  // Train the model
  console.log('🚀 Starting training...');
  await trainModel(dummyInteractions);
  
  console.log(`Training completed. Model trained: ${modelTrained}`);
  
  // Additional verification
  console.log('🔍 Final verification:', {
    modelTrained,
    hasModel: !!tfModel,
    modelLayers: tfModel ? tfModel.layers.length : 0,
    topicsCount: MODEL_TOPICS.length,
    hashtagsCount: MODEL_HASHTAGS.length
  });
  
  return modelTrained;
}

// Train model with option to use previous data
export async function trainModel(interactionsArr, usePreviousData = true) {
  console.log('🔄 trainModel called with:', {
    interactionsLength: interactionsArr?.length,
    modelTraining,
    topicsLength: MODEL_TOPICS.length,
    minInteractions: MIN_INTERACTIONS,
    tfAvailable: typeof tf !== 'undefined'
  });
  
  // Check if TensorFlow is available
  if (typeof tf === 'undefined') {
    const error = 'TensorFlow.js not available. Make sure tf is loaded globally.';
    console.error('❌', error);
    return false;
  }
  
  if (modelTraining || MODEL_TOPICS.length === 0) {
    console.log('❌ Early return:', { modelTraining, topicsLength: MODEL_TOPICS.length });
    return;
  }
  const interactions = interactionsArr;
  if (interactions.length < MIN_INTERACTIONS) {
    console.log('❌ Not enough interactions:', interactions.length, 'needed:', MIN_INTERACTIONS);
    return;
  }
  
  let resolveTraining;
  try {
    console.log('🏗️ Loading model...');
    await loadModel(true);
    
    // Verify model was created successfully
    if (!tfModel || tfModel.layers.length === 0) {
      throw new Error('Model creation failed - no model or no layers');
    }
    
    console.log(`✅ Model verification passed: ${tfModel.layers.length} layers created`);
    
    modelTraining = true;
    modelTrainingPromise = new Promise((r) => (resolveTraining = r));
    
    // Initialize with current interactions
    const recentInteractions = interactions.slice(-100);
    const xs = [], ys = [];
    
    console.log(`📊 Processing ${recentInteractions.length} interactions...`);
    
    // Process current interactions
    for (const inter of recentInteractions) {
      if (!inter.topics || !Array.isArray(inter.topics)) {
        console.log('⚠️ Skipping interaction with invalid topics:', inter);
        continue;
      }
      let preferenceScore = 0;
      
      // If not_interested is active, force score to -4
      if (inter.not_interested) {
        preferenceScore = WEIGHT_NOT_INTERESTED;
      } else {
        // Otherwise, calculate score normally
        if (inter.liked) preferenceScore += WEIGHT_LIKED;
        if (inter.interested) preferenceScore += WEIGHT_INTERESTED;
        if (inter.commented) preferenceScore += WEIGHT_COMMENTED;
        if (!inter.liked && !inter.interested && !inter.not_interested && !inter.commented) {
          const time_watched = inter.timeSpentMs || 0;
          const duration = inter.duration || 10000;
          const timeRatio = time_watched / duration;
          const timeScore = (timeRatio - 0.5) * 2;
          preferenceScore += timeScore;
        }
      }
      let engaged;
      if (preferenceScore <= -1.5) engaged = 0;
      else if (preferenceScore >= 1.5) engaged = 2;
      else engaged = 1;
      const x = MODEL_TOPICS.map((t) => (inter.topics || []).includes(t) ? 1 : 0);
      const y = [0, 0, 0];
      y[engaged] = 1;
      xs.push(x);
      ys.push(y);
    }
    
    console.log(`📈 Created training data: ${xs.length} samples with ${xs[0]?.length || 0} features`);
    
    // Add previous learned patterns if enabled
    if (usePreviousData && previousTopicsResults.length > 0) {
      console.log('🔄 Adding previous learned patterns...');
      // Use previous topic results to augment training
      for (const prevTopics of previousTopicsResults) {
        const prevTopicNames = prevTopics.map(t => t.name);
        
        // Generate synthetic data points based on previous high-weight topics
        for (const topic of prevTopics.filter(t => t.weight > 0.3)) {
          const x = MODEL_TOPICS.map((t) => t === topic.name ? 1 : 
            (prevTopicNames.includes(t) ? 0.5 : 0));
          const y = [0, 0, 0];
          // Create a more positive signal for previously important topics
          y[2] = topic.weight > 0.6 ? 1 : 0.8;
          xs.push(x);
          ys.push(y);
        }
      }
      console.log(`📈 Total training data after augmentation: ${xs.length} samples`);
    }
    
    if (xs.length > 0 && xs[0].length > 0) {
      console.log('🚀 Starting model training...');
      console.log(`Model compiled: ${tfModel.compiled}, layers: ${tfModel.layers.length}`);
      
      const xsTensor = tf.tensor2d(xs);
      const ysTensor = tf.tensor2d(ys);
      try {
        const history = await tfModel.fit(xsTensor, ysTensor, {
          epochs: 25,
          batchSize: 8,
          validationSplit: 0.2,
        });
        
        console.log('✅ Training completed successfully!');
        console.log('📊 Final loss:', history.history.loss[history.history.loss.length - 1]);
        
        modelTrained = true;
        console.log('✅ Model trained flag set to:', modelTrained);
        
        // Automatically export the model after successful training
        console.log('Training completed successfully, exporting model...');
        const exportSuccess = await exportModelToDownloads();
        if (exportSuccess) {
          console.log('Model exported to downloads folder');
        } else {
          console.warn('Model training succeeded but export failed');
        }
      } finally {
        xsTensor.dispose();
        ysTensor.dispose();
      }
    } else {
      console.log('❌ No valid training data created');
    }
  } catch (error) {
    console.error('❌ Training failed with error:', error);
    modelTrained = false;
  } finally {
    modelTraining = false;
    console.log('🏁 Training process finished. Final state:', {
      modelTrained,
      modelTraining,
      hasModel: !!tfModel,
      modelLayers: tfModel?.layers?.length || 0
    });
    if (resolveTraining) resolveTraining();
    modelTrainingPromise = null;
  }
}

// Analyze interactions with option to incorporate previous results
export async function analyzeInteractions(interactionsArr, blendPreviousResults = true) {
  const interactions = interactionsArr;
  if (MODEL_TOPICS.length === 0) return { topics: [], hashtags: [] };
  if (interactions.length < MIN_INTERACTIONS) return { topics: [], hashtags: [] };
  
  let results;
  if (!modelTrained) {
    results = analyzeWithoutModel(interactions);
  } else {
    await loadModel(false);
    const topicInputs = MODEL_TOPICS.map((t, i) =>
      MODEL_TOPICS.map((_, j) => (i === j ? 1 : 0))
    );
    const topicTensor = tf.tensor2d(topicInputs);
    const topicPreds = tfModel.predict(topicTensor).arraySync();
    topicTensor.dispose();
    const scoredTopics = MODEL_TOPICS.map((t, i) => ({
      name: t,
      weight: topicPreds[i][2] - topicPreds[i][0],
    }));
    scoredTopics.sort((a, b) => b.weight - a.weight);
    const splitIdx = findNaturalSplit(scoredTopics, 0.1, 'weight');
    const topTopics = scoredTopics.slice(0, splitIdx).filter((e) => e.weight > 0.1);
    
    // Hashtag analysis
    const hashtagInputs = MODEL_HASHTAGS.map((h, i) =>
      MODEL_TOPICS.map((t) =>
        interactions.some(
          (inter) => (inter.hashtags || []).includes(h) && (inter.topics || []).includes(t)
        ) ? 1 : 0
      )
    );
    const hashtagTensor = tf.tensor2d(hashtagInputs);
    const hashtagPreds = tfModel.predict(hashtagTensor).arraySync();
    hashtagTensor.dispose();
    const scoredHashtags = MODEL_HASHTAGS.map((h, i) => ({
      name: h,
      weight: hashtagPreds[i][2] - hashtagPreds[i][0],
    }));
    scoredHashtags.sort((a, b) => b.weight - a.weight);
    const splitHIdx = findNaturalSplit(scoredHashtags, 0.1, 'weight');
    const topHashtags = scoredHashtags.slice(0, splitHIdx).filter((e) => e.weight > 0.1);
    
    results = { topics: topTopics, hashtags: topHashtags };
  }
  
  // Blend with previous results if requested
  if (blendPreviousResults && previousTopicsResults.length > 0) {
    results = blendWithPreviousResults(results);
  }
  
  // Store the current results
  storeResults(results.topics, results.hashtags);
  
  return results;
}

// Blend current results with previous ones for stability
export function blendWithPreviousResults(currentResults) {
  if (previousTopicsResults.length === 0) return currentResults;
  
  const blendedTopics = [...currentResults.topics];
  const blendedHashtags = [...currentResults.hashtags];
  
  // Create maps for quick lookup
  const topicMap = new Map(blendedTopics.map(t => [t.name, t]));
  const hashtagMap = new Map(blendedHashtags.map(h => [h.name, h]));
  
  // Calculate recency weights (more recent = higher weight)
  const recencyWeights = Array(previousTopicsResults.length)
    .fill(0)
    .map((_, i) => 0.8 ** (previousTopicsResults.length - i - 1));
  
  // Add topics from previous results with decaying importance
  for (let i = 0; i < previousTopicsResults.length; i++) {
    const recencyWeight = recencyWeights[i];
    const prevTopics = previousTopicsResults[i];
    
    for (const prevTopic of prevTopics) {
      if (topicMap.has(prevTopic.name)) {
        // Blend with existing topic
        const current = topicMap.get(prevTopic.name);
        current.weight = current.weight * 0.7 + prevTopic.weight * 0.3 * recencyWeight;
      } else if (prevTopic.weight > 0.3) {
        // Add previous topic with decayed weight if it was important
        blendedTopics.push({
          name: prevTopic.name,
          weight: prevTopic.weight * 0.4 * recencyWeight
        });
        topicMap.set(prevTopic.name, blendedTopics[blendedTopics.length - 1]);
      }
    }
  }
  
  // Do the same for hashtags if we have previous results
  if (previousHashtagsResults.length > 0) {
    const hashtagRecencyWeights = Array(previousHashtagsResults.length)
      .fill(0)
      .map((_, i) => 0.8 ** (previousHashtagsResults.length - i - 1));
      
    for (let i = 0; i < previousHashtagsResults.length; i++) {
      const recencyWeight = hashtagRecencyWeights[i];
      const prevHashtags = previousHashtagsResults[i];
      
      for (const prevHashtag of prevHashtags) {
        if (hashtagMap.has(prevHashtag.name)) {
          const current = hashtagMap.get(prevHashtag.name);
          current.weight = current.weight * 0.7 + prevHashtag.weight * 0.3 * recencyWeight;
        } else if (prevHashtag.weight > 0.3) {
          blendedHashtags.push({
            name: prevHashtag.name,
            weight: prevHashtag.weight * 0.4 * recencyWeight
          });
          hashtagMap.set(prevHashtag.name, blendedHashtags[blendedHashtags.length - 1]);
        }
      }
    }
  }
  
  // Resort and filter by weight threshold
  blendedTopics.sort((a, b) => b.weight - a.weight);
  blendedHashtags.sort((a, b) => b.weight - a.weight);
  
  return {
    topics: blendedTopics.filter(t => t.weight > 0.1),
    hashtags: blendedHashtags.filter(h => h.weight > 0.1)
  };
}

// Simple analysis method when model isn't ready
export function analyzeWithoutModel(interactions) {
  const topicScores = {};
  const hashtagScores = {};
  MODEL_TOPICS.forEach(topic => {
    topicScores[topic] = { positive: 0, negative: 0, count: 0 };
  });
  MODEL_HASHTAGS.forEach(hashtag => {
    hashtagScores[hashtag] = { positive: 0, negative: 0, count: 0 };
  });
  for (const inter of interactions) {
    let score = 0;
    
    // If not_interested is active, force score to -4
    if (inter.not_interested) {
      score = WEIGHT_NOT_INTERESTED;
    } else {
      // Otherwise, calculate score normally
      if (inter.liked) score += WEIGHT_LIKED;
      if (inter.interested) score += WEIGHT_INTERESTED;
      if (inter.commented) score += WEIGHT_COMMENTED;
      const timeRatio = inter.timeSpentMs / (inter.duration || 10000);
      const timeScore = (timeRatio - 0.5) * 2;
      score += timeScore;
    }
    (inter.topics || []).forEach(topic => {
      if (topicScores[topic]) {
        if (score > 0) topicScores[topic].positive += score;
        if (score < 0) topicScores[topic].negative += Math.abs(score);
        topicScores[topic].count++;
      }
    });
    (inter.hashtags || []).forEach(hashtag => {
      if (hashtagScores[hashtag]) {
        if (score > 0) hashtagScores[hashtag].positive += score;
        if (score < 0) hashtagScores[hashtag].negative += Math.abs(score);
        hashtagScores[hashtag].count++;
      }
    });
  }
  const scoredTopics = Object.entries(topicScores)
    .map(([topic, data]) => {
      const avg = data.count > 0 ? (data.positive - data.negative) / data.count : 0;
      const total = data.positive - data.negative;
      // Composite: 60% average, 40% total (normalize total by dividing by max total)
      return {
        name: topic,
        weight: avg * 0.6 + (total / Math.max(1, interactions.length)) * 0.4
      };
    })
    .filter(item => item.weight !== 0)
    .sort((a, b) => b.weight - a.weight);

  const scoredHashtags = Object.entries(hashtagScores)
    .map(([hashtag, data]) => {
      const avg = data.count > 0 ? (data.positive - data.negative) / data.count : 0;
      const total = data.positive - data.negative;
      return {
        name: hashtag,
        weight: avg * 0.6 + (total / Math.max(1, interactions.length)) * 0.4
      };
    })
    .filter(item => item.weight !== 0)
    .sort((a, b) => b.weight - a.weight);

  const topTopics = scoredTopics.filter(item => item.weight > 0.1).slice(0, 5);
  const topHashtags = scoredHashtags.filter(item => item.weight > 0.1).slice(0, 5);
  return { topics: topTopics, hashtags: topHashtags };
}

// Helper to find natural split
export function findNaturalSplit(items, minThreshold = 0.1, scoreKey = "score") {
  if (items.length <= 3) return items.length;
  let splitIdx = items.length;
  let maxGap = 0;
  let gapThreshold = Math.max(0.1, items[0][scoreKey] * 0.25);
  for (let i = 0; i < Math.min(items.length - 1, 10); i++) {
    const gap = items[i][scoreKey] - items[i + 1][scoreKey];
    const relativeGap = gap / items[i][scoreKey];
    if ((gap > maxGap && gap > gapThreshold) || relativeGap > 0.4) {
      maxGap = gap;
      splitIdx = i + 1;
      if (relativeGap > 0.6) break;
    }
  }
  if (splitIdx === items.length) {
    return Math.min(5, items.filter(item => item[scoreKey] > minThreshold).length);
  }
  return splitIdx;
}

// Initialize the system
export function initializeModelSystem() {
  // Load any previously stored results
  loadStoredResults();
  
  // Return initial state
  return {
    hasPreviousData: previousTopicsResults.length > 0,
    previousTopicsCount: previousTopicsResults.length,
    previousHashtagsCount: previousHashtagsResults.length
  };
}