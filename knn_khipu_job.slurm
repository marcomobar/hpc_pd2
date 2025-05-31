#!/bin/bash
#SBATCH --job-name=knn_parallel_test
#SBATCH --output=knn_parallel_%j.out
#SBATCH --error=knn_parallel_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --time=00:30:00
#SBATCH --partition=general
#SBATCH --account=your_account  # Cambiar por tu cuenta en Khipu

# Cargar módulos necesarios
module load python/3.9
module load openmpi/4.1.1

# Activar entorno virtual (si es necesario)
# source /path/to/your/venv/bin/activate

# Instalar dependencias si no están instaladas
pip install --user mpi4py scikit-learn numpy matplotlib

# Ejecutar pruebas de escalabilidad
echo "Iniciando pruebas de KNN paralelo en Khipu"
echo "Fecha: $(date)"
echo "Nodo: $(hostname)"
echo ""

# Ejecutar el script de pruebas
bash run_scalability_tests.sh

# Analizar resultados
python analyze_results.py results/scalability_results_*.txt

echo ""
echo "Pruebas completadas: $(date)"