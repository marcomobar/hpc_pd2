# KNN Paralelo con MPI - Documentación

## Descripción General

Este proyecto implementa una versión paralelizada del algoritmo k-Nearest Neighbors (KNN) usando MPI (Message Passing Interface) para el dataset de dígitos de scikit-learn. La paralelización sigue la estructura del ejemplo `knn_hpc_class_vis.py` e incluye medición de tiempos y análisis de escalabilidad.

## Estructura del Código

### 1. **Inicialización MPI y Configuración**
```python
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
```
- Establece el comunicador MPI
- Identifica el proceso (rank) y el número total de procesos (size)

### 2. **Carga y Distribución de Datos**

#### Proceso 0 (Master):
- Carga el dataset de dígitos o genera datos sintéticos
- Divide los datos en entrenamiento y prueba
- Calcula la distribución de datos para cada proceso

#### Comunicación de Datos:
```python
# Broadcast de datos de prueba
X_test = comm.bcast(X_test, root=0)
y_test = comm.bcast(y_test, root=0)

# Distribución de datos de entrenamiento
comm.Send(X_train[...], dest=i, tag=11)
comm.Send(y_train[...], dest=i, tag=12)
```

### 3. **Cómputo Local**
Cada proceso:
1. Recibe su porción de datos de entrenamiento
2. Para cada punto de prueba:
   - Calcula distancias euclidianas a sus datos locales
   - Selecciona los k vecinos más cercanos localmente
   - Almacena distancias y etiquetas

### 4. **Gathering y Predicción Final**
```python
all_predictions = comm.gather(local_predictions, root=0)
```
- Todos los procesos envían sus predicciones locales al proceso 0
- El proceso 0 combina todos los vecinos, los ordena por distancia y realiza la votación final

### 5. **Medición de Tiempos**
El código mide tres componentes principales:
- **Distribución**: Tiempo para distribuir los datos de entrenamiento
- **Cómputo**: Tiempo de cálculo de distancias y vecinos locales
- **Gathering**: Tiempo para recolectar y combinar resultados

## Uso del Código

### Ejecución Básica
```bash
mpirun -np 4 python knn_digits_parallel.py
```

### Ejecución con Factor de Escala
```bash
mpirun -np 4 python knn_digits_parallel.py 10
```
Donde `10` es el factor que multiplica el tamaño del dataset de entrenamiento.

### Ejecución en Khipu
```bash
sbatch knn_khipu_job.slurm
```

## Scripts de Prueba

### 1. **run_scalability_tests.sh**
Ejecuta tres tipos de pruebas de escalabilidad:

- **Escalabilidad Fuerte**: Datos fijos, varía el número de procesos
- **Escalabilidad Débil**: Datos proporcionales al número de procesos
- **Variación de Datos**: Procesos fijos, varía el tamaño de datos

### 2. **analyze_results.py**
Analiza los resultados y genera gráficos de:
- Tiempo de ejecución vs. número de procesos
- Speedup
- Distribución de tiempos (cómputo vs. comunicación)
- Precisión vs. número de procesos

## Diferencias con el Código Secuencial

| Aspecto | Secuencial | Paralelo |
|---------|------------|----------|
| Distribución de datos | N/A | `comm.bcast`, `comm.Send/Recv` |
| Cálculo de distancias | Todos los datos en un proceso | Distribuido entre procesos |
| Selección de vecinos | Global directa | Local + gathering + merge |
| Medición de tiempos | Solo tiempo total | Desglose por componentes |

## Consideraciones de Escalabilidad

### Escalabilidad Fuerte
- Al aumentar procesos con datos fijos, el tiempo de cómputo disminuye
- La comunicación puede convertirse en cuello de botella
- Existe un punto óptimo de procesos

### Escalabilidad Débil
- Mantiene la carga de trabajo por proceso constante
- Ideal para evaluar la eficiencia del paralelismo
- Muestra el overhead de comunicación

### Factores que Afectan el Rendimiento
1. **Tamaño del dataset**: Más datos justifican más procesos
2. **Valor de k**: Afecta la cantidad de datos a comunicar
3. **Arquitectura del cluster**: Latencia y ancho de banda de la red
4. **Balance de carga**: Distribución equitativa de datos

## Optimizaciones Implementadas

1. **Manejo de datos no divisibles**: El código maneja correctamente cuando el número de datos no es divisible entre procesos
2. **Tipos de datos eficientes**: Uso de `float64` e `int` para comunicación MPI
3. **Minimización de comunicación**: Solo se comunican los k vecinos más cercanos locales

## Resultados Esperados

- **Accuracy**: Debe mantenerse igual que la versión secuencial (~0.96-0.98)
- **Speedup**: Cercano a lineal para datasets grandes
- **Eficiencia**: Disminuye con más procesos debido al overhead de comunicación

## Troubleshooting

### Error: "No module named 'mpi4py'"
```bash
pip install mpi4py
```

### Error en Khipu
Verificar módulos cargados:
```bash
module load python/3.9
module load openmpi/4.1.1
```

### Resultados incorrectos
- Verificar que todos los procesos reciban los datos correctamente
- Revisar que el gathering combine correctamente todos los vecinos

## Referencias

- [MPI4PY Documentation](https://mpi4py.readthedocs.io/)
- [Scikit-learn Digits Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html)
- Material del curso sobre paralelización con MPI