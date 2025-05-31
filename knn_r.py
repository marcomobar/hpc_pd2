#!/usr/bin/env python
import argparse
from mpi4py import MPI
import numpy as np
from collections import Counter
import sys

# -------------------------------
# 1. MPI setup and imports
# -------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--scale', type=int, default=1,
                    help='Factor para replicar el dataset original')
args = parser.parse_args()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Solo el rank 0 importa y divide el dataset
if rank == 0:
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split

# -------------------------------
# 2. Funciones auxiliares
# -------------------------------
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2, axis=1))

def knn_predict(test_point, X_subset, y_subset, k):
    distances = euclidean_distance(X_subset, test_point)  # shape: (N_subset,)
    k_indices = np.argsort(distances)[:k]                  # índices de los k menores
    distancias_k = distances[k_indices]                     # (k,)
    labels_k = y_subset[k_indices]                          # (k,)
    return distancias_k, labels_k

# -------------------------------
# 3. Parámetros K-NN
# -------------------------------
k = 3

# -------------------------------
# 4. Cargar y dividir datos (solo rank 0)
# -------------------------------
if rank == 0:
    
    X_orig, y_orig = load_digits(return_X_y=True)
    X_train0, X_test, y_train0, y_test = train_test_split(X_orig, y_orig, test_size=0.2, random_state=42)
    
    #digits = load_digits()
    X_train = np.tile(X_train0, (args.scale, 1))
    y_train = np.tile(y_train0, args.scale)

    #X_train, X_test, y_train, y_test = train_test_split(
    #    X, y, test_size=0.2, random_state=42
    #)
    # Recortar para que sea divisible por size
    n = X_train.shape[0] - (X_train.shape[0] % size)
    X_train = X_train[:n]
    y_train = y_train[:n]
    train_size = n
    test_size  = X_test.shape[0]
else:
    X_train = y_train = X_test = y_test = None
    train_size = None
    test_size  = None

# -------------------------------
# 5. Broadcast de tamaños
# -------------------------------
train_size = comm.bcast(train_size, root=0)
test_size  = comm.bcast(test_size,  root=0)

# -------------------------------
# 6. Broadcast del conjunto de test
# -------------------------------
X_test = comm.bcast(X_test, root=0)
y_test = comm.bcast(y_test, root=0)

# -------------------------------
# 7. Preparar buffers locales y Scatter
# -------------------------------
num_features = X_test.shape[1]               # número de columnas
local_train_size = train_size // size        # filas por proceso

local_X = np.empty((local_train_size, num_features), dtype='float64')
local_y = np.empty(local_train_size, dtype='int')

# -------------------------------
# 8. Scatter de X_train y y_train
# -------------------------------
t_start = MPI.Wtime()
comm.Scatter([X_train, MPI.DOUBLE], local_X, root=0)
comm.Scatter([y_train, MPI.INT],    local_y, root=0)
t_dist = MPI.Wtime()

# -------------------------------
# 9. Computación local usando knn_predict
# -------------------------------
local_neighbors = []  # Cada elemento: lista de k tuplas (distancia, etiqueta)
for x in X_test:
    dist_k, labels_k = knn_predict(x, local_X, local_y, k)
    local_neighbors.append(list(zip(dist_k, labels_k)))
t_comp = MPI.Wtime()

# -------------------------------
# 10. Gather de vecinos parciales en rank 0
# -------------------------------
all_neighbors = comm.gather(local_neighbors, root=0)
t_gather = MPI.Wtime()

# -------------------------------
# 11. Predicción final y métricas (solo rank 0)
# -------------------------------
if rank == 0:
    final_preds = []
    for i in range(test_size):
        neighs = []
        for proc_list in all_neighbors:
            neighs.extend(proc_list[i])
        neighs.sort(key=lambda x: x[0])
        top_k = [label for _, label in neighs[:k]]
        final_preds.append(Counter(top_k).most_common(1)[0][0])

    final_preds = np.array(final_preds)
    accuracy = np.mean(final_preds == y_test)

    print(f"[Process Count: {size}] Dataset Size: {train_size}")
    print(f"Total Time       : {t_gather - t_start:.4f} sec")
    print(f"  - Distribution : {t_dist - t_start:.4f} sec")
    print(f"  - Computation  : {t_comp - t_dist:.4f} sec")
    print(f"  - Gathering    : {t_gather - t_comp:.4f} sec")
    print(f"Accuracy         : {accuracy:.4f}")