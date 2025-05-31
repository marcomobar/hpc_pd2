import glob
import re
import numpy as np
import matplotlib.pyplot as plt

# Baseline dataset sizes from digits dataset
TRAIN_SIZE = 1437  # original training samples
TEST_SIZE = 360    # original test samples
OPS_PER_PAIR = 193  # flops per distance computation

PAIRS_BASE = TRAIN_SIZE * TEST_SIZE
FLOPS_BASE = PAIRS_BASE * OPS_PER_PAIR

# parse time files
speedup_data = {}
flop_data = {}

pattern = re.compile(r"times_N(\d+)\.txt")
files = sorted(glob.glob("times_N*.txt"))
for filename in files:
    m = pattern.search(filename)
    if not m:
        continue
    scale = int(m.group(1))
    dataset_flops = FLOPS_BASE * scale
    procs = []
    total = []
    comp = []
    with open(filename) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            p = int(parts[0])
            t_total = float(parts[1])
            t_comp = float(parts[3])
            procs.append(p)
            total.append(t_total)
            comp.append(t_comp)
    total = np.array(total)
    comp = np.array(comp)
    procs = np.array(procs)
    speedup = total[0] / total
    flops = dataset_flops / comp
    speedup_data[scale] = (procs, speedup)
    flop_data[scale] = (procs, flops)

# -------- Speedup plot for largest dataset --------
max_scale = max(speedup_data.keys())
p, s = speedup_data[max_scale]
plt.figure()
plt.plot(p, s, 'o-', label=f'N={max_scale}')
plt.plot(p, p, 'k--', label='Ideal')
plt.xlabel('Procesos (p)')
plt.ylabel('Speedup')
plt.title('Speedup vs p')
plt.grid(True)
plt.legend()
plt.savefig('speedup_from_data.png')

# -------- FLOPs/s vs processes for largest dataset --------
p, f = flop_data[max_scale]
plt.figure()
plt.plot(p, f/1e9, 'o-')
plt.xlabel('Procesos (p)')
plt.ylabel('GFLOP/s')
plt.title(f'Rendimiento (N={max_scale})')
plt.grid(True)
plt.savefig('flops_vs_p.png')

# -------- FLOPs/s vs dataset size at p=16 --------
scales = sorted(flop_data.keys())
flops_p16 = []
for sc in scales:
    p, f = flop_data[sc]
    if 16 in p:
        idx = list(p).index(16)
        flops_p16.append(f[idx]/1e9)
    else:
        flops_p16.append(np.nan)

plt.figure()
plt.plot(scales, flops_p16, 'o-')
plt.xlabel('Factor de escala (n)')
plt.ylabel('GFLOP/s a p=16')
plt.title('Escalabilidad con n creciente')
plt.grid(True)
plt.savefig('flops_vs_n.png')
