import os
import numpy as np
import pandas as pd

INPUT_FILE = os.path.join(os.path.dirname(__file__), "kalipoints_real",
                          "rosbag2_2026_02_24-17_08_39_kalipoints.csv")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "max_distance_results.csv")

# --- Punkte laden ---
df = pd.read_csv(INPUT_FILE, sep=';')
points = df[['X', 'Y', 'Z']].values  # shape (N, 3)
N = len(points)
print(f"Punkte geladen: {N}")

# --- Grössten Abstand je Punkt berechnen ---
max_distances = []

for i in range(N):
    d_max = 0.0
    p_i = points[i]
    for j in range(N):
        if i == j:
            continue
        diff = p_i - points[j]
        dist = np.sqrt(np.dot(diff, diff))
        if dist > d_max:
            d_max = dist
    max_distances.append(d_max)
    if (i + 1) % 100 == 0 or (i + 1) == N:
        print(f"  Fortschritt: {i + 1}/{N}", end='\r')

print()

max_distances = np.array(max_distances)

# --- Statistiken ---
mean_val   = np.mean(max_distances)
median_val = np.median(max_distances)
std_val    = np.std(max_distances)

print(f"\nMittelwert:                {mean_val:.10f}")
print(f"Median:                    {median_val:.10f}")
print(f"Standardabweichung:        {std_val:.10f}")

# --- CSV schreiben ---
with open(OUTPUT_FILE, 'w', newline='') as f:
    f.write(f"# SOURCE: {os.path.basename(INPUT_FILE)}\n")
    f.write(f"# N_POINTS: {N}\n")
    f.write(f"# MEAN: {mean_val:.10f}\n")
    f.write(f"# MEDIAN: {median_val:.10f}\n")
    f.write(f"# STD: {std_val:.10f}\n")
    f.write("max_distance\n")
    for d in max_distances:
        f.write(f"{d:.10f}\n")

print(f"\nErgebnisse gespeichert: {OUTPUT_FILE}")
