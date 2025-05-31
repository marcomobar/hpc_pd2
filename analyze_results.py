import re
import matplotlib.pyplot as plt
import sys

def parse_results(filename):
    with open(filename, 'r') as f:
        content = f.read()
    
    # Expresiones regulares para extraer datos
    procs_pattern = r'\[RESULTS\] Process Count: (\d+)'
    size_pattern = r'\[RESULTS\] Training Size: (\d+)'
    total_time_pattern = r'\[TIMING\] Total Time\s+: ([\d.]+) sec'
    comp_time_pattern = r'\[TIMING\]\s+- Computation\s+: ([\d.]+) sec'
    comm_time_pattern = r'\[TIMING\]\s+- Distribution\s+: ([\d.]+) sec'
    gather_time_pattern = r'\[TIMING\]\s+- Gathering\s+: ([\d.]+) sec'
    accuracy_pattern = r'\[ACCURACY\] Accuracy: ([\d.]+)'
    
    results = []
    
    # Dividir por pruebas
    tests = content.split('Procesos:')
    
    for test in tests[1:]:  # Skip first empty split
        procs_match = re.search(procs_pattern, test)
        size_match = re.search(size_pattern, test)
        total_time_match = re.search(total_time_pattern, test)
        comp_time_match = re.search(comp_time_pattern, test)
        comm_time_match = re.search(comm_time_pattern, test)
        gather_time_match = re.search(gather_time_pattern, test)
        accuracy_match = re.search(accuracy_pattern, test)
        
        if all([procs_match, total_time_match, accuracy_match]):
            results.append({
                'procs': int(procs_match.group(1)),
                'size': int(size_match.group(1)) if size_match else 0,
                'total_time': float(total_time_match.group(1)),
                'comp_time': float(comp_time_match.group(1)) if comp_time_match else 0,
                'comm_time': float(comm_time_match.group(1)) if comm_time_match else 0,
                'gather_time': float(gather_time_match.group(1)) if gather_time_match else 0,
                'accuracy': float(accuracy_match.group(1))
            })
    
    return results

# Analizar resultados si se proporciona archivo
if len(sys.argv) > 1:
    results = parse_results(sys.argv[1])
    
    # Crear gráficos
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Separar resultados por tipo de prueba (estimación básica)
    strong_scaling = [r for r in results if r['size'] > 14000 and r['size'] < 15000]
    
    if strong_scaling:
        procs = [r['procs'] for r in strong_scaling]
        times = [r['total_time'] for r in strong_scaling]
        speedup = [times[0] / t for t in times]
        
        ax1.plot(procs, times, 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('Número de Procesos')
        ax1.set_ylabel('Tiempo Total (s)')
        ax1.set_title('Escalabilidad Fuerte - Tiempo de Ejecución')
        ax1.grid(True)
        
        ax2.plot(procs, speedup, 'ro-', linewidth=2, markersize=8)
        ax2.plot(procs, procs, 'k--', label='Speedup ideal')
        ax2.set_xlabel('Número de Procesos')
        ax2.set_ylabel('Speedup')
        ax2.set_title('Escalabilidad Fuerte - Speedup')
        ax2.legend()
        ax2.grid(True)
    
    # Gráfico de tiempos por componente
    if results:
        procs = [r['procs'] for r in results[:5]]
        comp_times = [r['comp_time'] for r in results[:5]]
        comm_times = [r['comm_time'] + r['gather_time'] for r in results[:5]]
        
        width = 0.35
        x = range(len(procs))
        
        ax3.bar([i - width/2 for i in x], comp_times, width, label='Cómputo')
        ax3.bar([i + width/2 for i in x], comm_times, width, label='Comunicación')
        ax3.set_xlabel('Número de Procesos')
        ax3.set_ylabel('Tiempo (s)')
        ax3.set_title('Distribución de Tiempos')
        ax3.set_xticks(x)
        ax3.set_xticklabels(procs)
        ax3.legend()
        ax3.grid(True, axis='y')
    
    # Accuracy vs procesos
    if results:
        procs = [r['procs'] for r in results]
        accuracies = [r['accuracy'] for r in results]
        
        ax4.plot(procs[:5], accuracies[:5], 'go-', linewidth=2, markersize=8)
        ax4.set_xlabel('Número de Procesos')
        ax4.set_ylabel('Precisión')
        ax4.set_title('Precisión vs Número de Procesos')
        ax4.set_ylim([0.9, 1.0])
        ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig('scalability_analysis.png')
    print("Gráficos guardados en: scalability_analysis.png")
else:
    print("Uso: python analyze_results.py <archivo_resultados>")
