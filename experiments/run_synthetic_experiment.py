"""
CEBRA Synthetic Data Experiment Script
This script runs a complete CEBRA experiment on synthetic data, including:
- Loading synthetic dataset
- Training CEBRA model
- Evaluating embedding quality
- Visualizing results
- Saving outputs
"""

import os
import sys
import argparse
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
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import torch
import joblib
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cebra
from cebra import CEBRA
import cebra.datasets


def setup_directories(output_dir):
    """Create output directories for saving results."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (output_path / 'models').mkdir(exist_ok=True)
    (output_path / 'embeddings').mkdir(exist_ok=True)
    (output_path / 'figures').mkdir(exist_ok=True)
    (output_path / 'logs').mkdir(exist_ok=True)
    
    return output_path


def load_synthetic_data(dataset_name='continuous-label-poisson'):
    """
    Load synthetic dataset from CEBRA's built-in datasets.
    
    Available datasets:
    - continuous-label-t
    - continuous-label-uniform
    - continuous-label-laplace
    - continuous-label-poisson
    - continuous-label-gaussian
    """
    print(f"Loading synthetic dataset: {dataset_name}")
    try:
        dataset = cebra.datasets.init(dataset_name, download=True)
    except TypeError:
        dataset = cebra.datasets.init(dataset_name)
    
    # Get neural data and labels
    neural_data = dataset.neural.numpy()
    continuous_labels = dataset.continuous_index.numpy()
    
    print(f"Neural data shape: {neural_data.shape}")
    print(f"Continuous labels shape: {continuous_labels.shape}")
    
    return neural_data, continuous_labels, dataset


def train_cebra_model(neural_data, continuous_labels, config):
    """Train CEBRA model on synthetic data."""
    print("\n" + "="*60)
    print("Training CEBRA Model")
    print("="*60)
    
    # Initialize CEBRA model
    cebra_model = CEBRA(
        model_architecture=config['model_architecture'],
        batch_size=config['batch_size'],
        learning_rate=config['learning_rate'],
        temperature=config['temperature'],
        output_dimension=config['output_dimension'],
        max_iterations=config['max_iterations'],
        distance=config['distance'],
        conditional=config['conditional'],
        device=config['device'],
        verbose=True,
        time_offsets=config['time_offsets']
    )
    
    print(f"\nModel configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Train model
    print(f"\nStarting training for {config['max_iterations']} iterations...")
    cebra_model.fit(neural_data, continuous_labels)
    
    print("Training completed!")
    
    return cebra_model


def evaluate_embedding(cebra_model, neural_data, continuous_labels, output_path):
    """Evaluate and visualize the learned embedding."""
    print("\n" + "="*60)
    print("Evaluating Embedding")
    print("="*60)
    
    # Transform data to embedding space
    embedding = cebra_model.transform(neural_data)
    print(f"Embedding shape: {embedding.shape}")
    
    # Save embedding
    embedding_file = output_path / 'embeddings' / 'embedding.npy'
    np.save(embedding_file, embedding)
    print(f"Embedding saved to: {embedding_file}")
    
    # Create visualizations
    create_visualizations(embedding, continuous_labels, output_path)
    
    # Calculate embedding quality metrics
    metrics = calculate_metrics(embedding, continuous_labels)
    print("\nEmbedding Quality Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    return embedding, metrics


def create_visualizations(embedding, continuous_labels, output_path):
    """Create and save visualization plots."""
    print("\nCreating visualizations...")
    
    fig = plt.figure(figsize=(20, 5))
    
    label_array = np.asarray(continuous_labels)
    if label_array.ndim == 1:
        label_2d = label_array.reshape(-1, 1)
    else:
        label_2d = label_array
    color_values = label_2d[:, 0]
    
    # Plot 1: 2D embedding colored by (first) continuous label dimension
    ax1 = fig.add_subplot(141)
    scatter1 = ax1.scatter(embedding[:, 0], embedding[:, 1], 
                          c=color_values, cmap='viridis', 
                          s=1, alpha=0.5)
    ax1.set_xlabel('CEBRA Dimension 1')
    ax1.set_ylabel('CEBRA Dimension 2')
    ax1.set_title('2D CEBRA Embedding')
    plt.colorbar(scatter1, ax=ax1, label='Continuous Label (dim 1)')
    
    # Plot 2: 3D embedding if available
    if embedding.shape[1] >= 3:
        ax2 = fig.add_subplot(142, projection='3d')
        scatter2 = ax2.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2],
                              c=color_values, cmap='viridis', 
                              s=1, alpha=0.5)
        ax2.set_xlabel('Dim 1')
        ax2.set_ylabel('Dim 2')
        ax2.set_zlabel('Dim 3')
        ax2.set_title('3D CEBRA Embedding')
        plt.colorbar(scatter2, ax=ax2, label='Continuous Label (dim 1)')
    
    # Plot 3: Embedding dimensions over time
    ax3 = fig.add_subplot(143)
    time_indices = np.arange(min(1000, len(embedding)))
    for dim in range(min(3, embedding.shape[1])):
        ax3.plot(time_indices, embedding[time_indices, dim], 
                label=f'Dim {dim+1}', alpha=0.7, linewidth=0.5)
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Embedding Value')
    ax3.set_title('Embedding Dimensions Over Time')
    ax3.legend()
    
    # Plot 4: Continuous label over time
    ax4 = fig.add_subplot(144)
    max_label_dims = min(3, label_2d.shape[1])
    for idx in range(max_label_dims):
        ax4.plot(time_indices, label_2d[time_indices, idx], linewidth=0.8, label=f'Label Dim {idx+1}')
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Continuous Label')
    ax4.set_title('Continuous Label Over Time')
    if label_2d.shape[1] > 1:
        ax4.legend()
    
    plt.tight_layout()
    
    # Save figure
    fig_path = output_path / 'figures' / 'embedding_visualization.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {fig_path}")
    plt.close()


def calculate_metrics(embedding, continuous_labels):
    """Calculate quality metrics for the embedding."""
    from sklearn.metrics import r2_score
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        embedding, continuous_labels, test_size=0.2, random_state=42
    )
    
    # Train a simple decoder to predict labels from embedding
    decoder = Ridge(alpha=1.0)
    decoder.fit(X_train, y_train)
    
    # Evaluate
    train_score = decoder.score(X_train, y_train)
    test_score = decoder.score(X_test, y_test)
    
    # Calculate variance explained
    y_pred = decoder.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    
    metrics = {
        'train_r2': train_score,
        'test_r2': test_score,
        'prediction_r2': r2
    }
    
    return metrics


def save_results(cebra_model, embedding, metrics, config, output_path):
    """Save all results including model, embeddings, and metrics."""
    print("\n" + "="*60)
    print("Saving Results")
    print("="*60)
    
    # Save CEBRA model
    model_file = output_path / 'models' / 'cebra_model.pt'
    cebra_model.save(model_file)
    print(f"Model saved to: {model_file}")
    
    # Save configuration and metrics
    results = {
        'config': config,
        'metrics': metrics,
        'embedding_shape': embedding.shape,
        'timestamp': datetime.now().isoformat()
    }
    
    results_file = output_path / 'logs' / 'experiment_results.jl'
    joblib.dump(results, results_file)
    print(f"Results saved to: {results_file}")
    
    # Save text summary
    summary_file = output_path / 'logs' / 'summary.txt'
    with open(summary_file, 'w') as f:
        f.write("CEBRA Synthetic Data Experiment Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Timestamp: {results['timestamp']}\n\n")
        f.write("Configuration:\n")
        for key, value in config.items():
            f.write(f"  {key}: {value}\n")
        f.write("\nMetrics:\n")
        for key, value in metrics.items():
            f.write(f"  {key}: {value:.4f}\n")
        f.write(f"\nEmbedding shape: {embedding.shape}\n")
    
    print(f"Summary saved to: {summary_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Run CEBRA experiment on synthetic data'
    )
    
    # Data arguments
    parser.add_argument('--dataset', type=str, 
                       default='demo-continuous',
                       choices=[
                            'continuous-label-t', 'continuous-label-uniform',
                            'continuous-label-laplace', 'continuous-label-poisson',
                            'continuous-label-gaussian', 'demo-continuous',
                            'demo-discrete', 'demo-mixed'
    ],
                       help='Synthetic dataset to use')
    
    # Model arguments
    parser.add_argument('--model-architecture', type=str, default='offset10-model',
                       help='Model architecture')
    parser.add_argument('--output-dimension', type=int, default=3,
                       help='Output embedding dimension')
    parser.add_argument('--max-iterations', type=int, default=10000,
                       help='Maximum training iterations')
    parser.add_argument('--batch-size', type=int, default=1024,
                       help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--temperature', type=float, default=0.8,
                       help='Temperature for InfoNCE loss')
    parser.add_argument('--time-offsets', type=int, default=10,
                       help='Time offset for positive pairs')
    parser.add_argument('--distance', type=str, default='cosine',
                       choices=['cosine', 'euclidean'],
                       help='Distance metric')
    parser.add_argument('--conditional', type=str, default='time_delta',
                       choices=['time', 'time_delta'],
                       help='Conditional distribution type')
    
    # Output arguments
    parser.add_argument('--output-dir', type=str, 
                       default='./results/synthetic_experiment',
                       help='Directory to save results')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device for training')
    
    args = parser.parse_args()
    
    # Check device availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, switching to CPU")
        args.device = 'cpu'
    
    print("="*60)
    print("CEBRA Synthetic Data Experiment")
    print("="*60)
    
    # Setup
    output_path = setup_directories(args.output_dir)
    
    # Configuration
    config = {
        'model_architecture': args.model_architecture,
        'output_dimension': args.output_dimension,
        'max_iterations': args.max_iterations,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'temperature': args.temperature,
        'time_offsets': args.time_offsets,
        'distance': args.distance,
        'conditional': args.conditional,
        'device': args.device,
        'dataset': args.dataset
    }
    
    # Load data
    neural_data, continuous_labels, dataset = load_synthetic_data(args.dataset)
    
    # Train model
    cebra_model = train_cebra_model(neural_data, continuous_labels, config)
    
    # Evaluate
    embedding, metrics = evaluate_embedding(cebra_model, neural_data, 
                                           continuous_labels, output_path)
    
    # Save everything
    save_results(cebra_model, embedding, metrics, config, output_path)
    
    print("\n" + "="*60)
    print("Experiment completed successfully!")
    print(f"All results saved to: {output_path}")
    print("="*60)


if __name__ == '__main__':
    main()
