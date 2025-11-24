"""
Quick test script for CEBRA synthetic data experiment
This is a minimal version for quick testing with reduced iterations.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

# Provide legacy NumPy aliases expected by older SciPy/CEBRA code paths.
if not hasattr(np, "typeDict"):
    np.typeDict = np.sctypeDict

if not hasattr(np, "dtypes"):
    class _CompatDTypes:
        Float64DType = np.float64
        Int64DType = np.int64

    np.dtypes = _CompatDTypes()

if not hasattr(np, "int"):
    np.int = int

if not hasattr(np, "bool"):
    np.bool = np.bool_

import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cebra
from cebra import CEBRA
import cebra.datasets


def generate_local_synthetic(num_samples=6000, num_neurons=64, seed=7):
    rng = np.random.default_rng(seed)
    time = np.linspace(0.0, 4 * np.pi, num_samples, dtype=np.float32)
    latent = np.stack((
        np.sin(time),
        np.cos(time),
        np.sin(2.0 * time),
    ), axis=1).astype(np.float32)
    mixing = rng.standard_normal(size=(latent.shape[1], num_neurons)).astype(np.float32)
    neural = latent @ mixing
    neural += 0.1 * rng.standard_normal(size=neural.shape).astype(np.float32)
    labels = ((time - time.min()) / (time.max() - time.min())).astype(np.float32)
    return neural, labels


def load_synthetic_dataset(dataset_name='continuous-label-poisson'):
    print('Loading synthetic dataset...')
    try:
        dataset = cebra.datasets.init(dataset_name, download=True)
    except Exception as exc:
        print(f"Failed to load dataset {dataset_name!r}: {exc}")
        print('Generating fallback synthetic dataset locally.')
        neural_data, continuous_labels = generate_local_synthetic()
    else:
        neural_data = np.asarray(dataset.neural.numpy(), dtype=np.float32)
        continuous_labels = np.asarray(dataset.continuous_index.numpy(), dtype=np.float32).reshape(-1)
    neural_data = np.asarray(neural_data, dtype=np.float32)
    continuous_labels = np.asarray(continuous_labels, dtype=np.float32).reshape(-1)
    return neural_data, continuous_labels

neural_data, continuous_labels = load_synthetic_dataset()
print(f"Neural data shape: {neural_data.shape}")
print(f"Labels shape: {continuous_labels.shape}")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\nInitializing CEBRA model on {device.upper()}...")
cebra_model = CEBRA(
    model_architecture='offset10-model',
    batch_size=1024,
    learning_rate=3e-4,
    temperature=0.8,
    output_dimension=3,
    max_iterations=10000,  # Quick test
    distance='cosine',
    conditional='time_delta',
    device=device,
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
