#!/usr/bin/env python3
"""
KNN Parallelized for Digits Dataset using MPI
Based on knn_digits_sec.py and knn_hpc_class_vis.py structure

Usage:
    mpirun -np <num_processes> python knn_digits_parallel.py [scale_factor]
    
    scale_factor: optional, multiplies the training data size (default: 1)
"""

from mpi4py import MPI
import numpy as np
from collections import Counter
import sys
import time

# Only rank 0 will import these for dataset and visualization
if MPI.COMM_WORLD.Get_rank() == 0:
    from sklearn.datasets import load_digits, make_classification
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt

def euclidean_distance(a, b):
    """Compute euclidean distance between arrays a and b"""
    return np.sqrt(np.sum((a - b) ** 2, axis=1))

# MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Parameters
k = 3  # Number of neighbors
scale_factor = int(sys.argv[1]) if len(sys.argv) >= 2 else 1  # Data scaling factor
use_synthetic = False  # Set to True to use make_classification instead of digits

# Load and prepare data (only on rank 0)
if rank == 0:
    print(f"[MPI KNN] Running with {size} processes")
    print(f"[MPI KNN] Data scale factor: {scale_factor}")
    
    if use_synthetic:
        # Option 1: Use synthetic data for variable size
        n_samples = 1437 * scale_factor + 360  # Similar to digits dataset size
        X, y = make_classification(
            n_samples=n_samples,
            n_features=64,  # Same as digits
            n_informative=50,
            n_redundant=0,
            n_classes=10,  # Same as digits
            random_state=42,
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    else:
        # Option 2: Use digits dataset and replicate for scaling
        digits = load_digits()
        X_train, X_test, y_train, y_test = train_test_split(
            digits.data, digits.target, test_size=0.2, random_state=42
        )
        
        # Scale training data by replicating
        if scale_factor > 1:
            X_train = np.tile(X_train, (scale_factor, 1))
            y_train = np.tile(y_train, scale_factor)
    
    train_size = len(X_train)
    test_size = len(X_test)
    num_features = X_train.shape[1]
    
    print(f"[MPI KNN] Training size: {train_size}, Test size: {test_size}")
    print(f"[MPI KNN] Features: {num_features}")
else:
    X_train = y_train = X_test = y_test = None
    train_size = test_size = num_features = None

# Broadcast dimensions
train_size = comm.bcast(train_size, root=0)
test_size = comm.bcast(test_size, root=0)
num_features = comm.bcast(num_features, root=0)

# Broadcast test data and labels to all processes
if rank == 0:
    X_test = X_test.astype('float64')
    y_test = y_test.astype('int')
else:
    X_test = np.empty((test_size, num_features), dtype='float64')
    y_test = np.empty(test_size, dtype='int')

X_test = comm.bcast(X_test, root=0)
y_test = comm.bcast(y_test, root=0)

# Calculate local training data size
local_train_size = train_size // size
remainder = train_size % size

# Handle uneven distribution
if rank < remainder:
    local_train_size += 1
    offset = rank * local_train_size
else:
    offset = remainder * (local_train_size + 1) + (rank - remainder) * local_train_size

# Prepare local arrays
local_X = np.empty((local_train_size, num_features), dtype='float64')
local_y = np.empty(local_train_size, dtype='int')

# Timing starts here
t_start = MPI.Wtime()

# Scatter training data
if rank == 0:
    # Prepare data for scattering
    X_train = X_train.astype('float64')
    y_train = y_train.astype('int')
    
    # Send data to each process
    for i in range(size):
        if i == 0:
            local_X = X_train[offset:offset+local_train_size]
            local_y = y_train[offset:offset+local_train_size]
        else:
            proc_size = train_size // size
            proc_remainder = train_size % size
            if i < proc_remainder:
                proc_size += 1
                proc_offset = i * proc_size
            else:
                proc_offset = proc_remainder * (proc_size + 1) + (i - proc_remainder) * proc_size
            
            comm.Send(X_train[proc_offset:proc_offset+proc_size].astype('float64'), dest=i, tag=11)
            comm.Send(y_train[proc_offset:proc_offset+proc_size].astype('int'), dest=i, tag=12)
else:
    comm.Recv(local_X, source=0, tag=11)
    comm.Recv(local_y, source=0, tag=12)

t_dist = MPI.Wtime()

# Local distance computation and prediction
local_predictions = []
for x in X_test:
    # Compute distances to all local training points
    dists = euclidean_distance(local_X, x)
    # Get k nearest neighbors locally
    k_indices = dists.argsort()[:min(k, len(dists))]
    k_labels = local_y[k_indices]
    k_dists = dists[k_indices]
    local_predictions.append((k_dists, k_labels))

t_comp = MPI.Wtime()

# Gather all partial predictions
all_predictions = comm.gather(local_predictions, root=0)

t_gather = MPI.Wtime()

# Final prediction on rank 0
if rank == 0:
    final_preds = []
    
    for i in range(test_size):
        # Combine all neighbors from all processes
        all_neighbors = []
        for proc_preds in all_predictions:
            if proc_preds and i < len(proc_preds):
                all_neighbors.extend(zip(proc_preds[i][0], proc_preds[i][1]))
        
        # Sort by distance and get top k
        all_neighbors.sort(key=lambda x: x[0])
        top_k = [label for _, label in all_neighbors[:k]]
        
        # Vote for final prediction
        final_pred = Counter(top_k).most_common(1)[0][0]
        final_preds.append(final_pred)
    
    final_preds = np.array(final_preds)
    accuracy = np.mean(final_preds == y_test)
    
    # Print results
    print("\n" + "="*60)
    print(f"[RESULTS] Process Count: {size}")
    print(f"[RESULTS] Training Size: {train_size}")
    print(f"[RESULTS] Test Size: {test_size}")
    print("-"*60)
    print(f"[TIMING] Total Time        : {t_gather - t_start:.4f} sec")
    print(f"[TIMING]   - Distribution  : {t_dist - t_start:.4f} sec")
    print(f"[TIMING]   - Computation   : {t_comp - t_dist:.4f} sec")
    print(f"[TIMING]   - Gathering     : {t_gather - t_comp:.4f} sec")
    print("-"*60)
    print(f"[ACCURACY] Accuracy: {accuracy:.4f}")
    print("="*60)
    
    # Visualize sample predictions (similar to original)
    if not use_synthetic:  # Only for digits dataset
        fig, axes = plt.subplots(2, 5, figsize=(12, 6))
        for i, ax in enumerate(axes.flat):
            if i < len(X_test):
                ax.imshow(X_test[i].reshape(8, 8), cmap='gray')
                ax.set_title(f"Pred: {final_preds[i]}\nTrue: {y_test[i]}")
                ax.axis('off')
        plt.suptitle(f"Sample Predictions (Parallel KNN, {size} processes)")
        plt.tight_layout()
        plt.savefig(f'knn_parallel_predictions_p{size}_scale{scale_factor}.png')
        plt.show()