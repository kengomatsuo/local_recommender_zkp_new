// Test script to verify model export functionality
// This script simulates the model training and export process

import { exportModelToDownloads, trainModel, MODEL_TOPICS, MODEL_HASHTAGS } from './src/model.js';

// Mock some sample data for testing
const mockInteractions = [
  {
    postId: 'test1',
    topics: ['technology', 'ai'],
    hashtags: ['#tech', '#ai'],
    liked: true,
    interested: true,
    not_interested: false,
    commented: false,
    timeSpentMs: 5000,
    timestamp: Date.now()
  },
  {
    postId: 'test2',
    topics: ['science', 'research'],
    hashtags: ['#science', '#research'],
    liked: false,
    interested: false,
    not_interested: true,
    commented: false,
    timeSpentMs: 1000,
    timestamp: Date.now()
  }
];

// Populate MODEL_TOPICS and MODEL_HASHTAGS
MODEL_TOPICS.push('technology', 'ai', 'science', 'research', 'programming', 'javascript');
MODEL_HASHTAGS.push('#tech', '#ai', '#science', '#research', '#programming', '#js');

async function testModelExport() {
  console.log('🧪 Testing model export functionality...');
  
  try {
    // Add more interactions to meet minimum requirements
    const extendedInteractions = [];
    for (let i = 0; i < 15; i++) {
      extendedInteractions.push({
        ...mockInteractions[i % 2],
        postId: `test${i}`,
        timestamp: Date.now() + i * 1000
      });
    }
    
    console.log('📚 Training model with test data...');
    await trainModel(extendedInteractions);
    
    console.log('✅ Test completed! Check console for export results.');
  } catch (error) {
    console.error('❌ Test failed:', error);
  }
}

// Run the test
testModelExport();
