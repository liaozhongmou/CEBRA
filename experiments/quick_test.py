"""
Quick test script for CEBRA synthetic data experiment
This is a minimal version for quick testing with reduced iterations.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cebra
from cebra import CEBRA
import cebra.datasets

print("Loading synthetic dataset...")
dataset = cebra.datasets.init('continuous-label-poisson', download=True)
neural_data = dataset.neural.numpy()
continuous_labels = dataset.continuous_index.numpy()

print(f"Neural data shape: {neural_data.shape}")
print(f"Labels shape: {continuous_labels.shape}")

print("\nInitializing CEBRA model...")
cebra_model = CEBRA(
    model_architecture='offset10-model',
    batch_size=512,
    learning_rate=3e-4,
    temperature=1.0,
    output_dimension=3,
    max_iterations=1000,  # Quick test
    distance='cosine',
    conditional='time_delta',
    device='cuda',
    verbose=True,
    time_offsets=10
)

print("\nTraining CEBRA model...")
cebra_model.fit(neural_data, continuous_labels)

print("\nGenerating embedding...")
embedding = cebra_model.transform(neural_data)
print(f"Embedding shape: {embedding.shape}")

print("\nCreating visualization...")
plt.figure(figsize=(12, 4))

plt.subplot(131)
plt.scatter(embedding[:, 0], embedding[:, 1], c=continuous_labels, 
           cmap='viridis', s=1, alpha=0.5)
plt.xlabel('CEBRA Dim 1')
plt.ylabel('CEBRA Dim 2')
plt.title('2D Embedding')
plt.colorbar(label='Continuous Label')

plt.subplot(132)
time_idx = np.arange(min(500, len(embedding)))
for i in range(3):
    plt.plot(time_idx, embedding[time_idx, i], label=f'Dim {i+1}', alpha=0.7)
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Embedding Over Time')
plt.legend()

plt.subplot(133)
plt.plot(time_idx, continuous_labels[time_idx], color='purple')
plt.xlabel('Time')
plt.ylabel('Label')
plt.title('Continuous Label')

plt.tight_layout()
plt.savefig('quick_test_result.png', dpi=150)
print("Visualization saved to: quick_test_result.png")

# Simple decoding test
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    embedding, continuous_labels, test_size=0.2, random_state=42
)
decoder = Ridge(alpha=1.0)
decoder.fit(X_train, y_train)
score = decoder.score(X_test, y_test)

print(f"\nDecoding R² score: {score:.4f}")
print("\nQuick test completed!")
