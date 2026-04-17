import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load the CSV file
csv_file = Path(__file__).parent / "rosbag2_2026_02_24-16_31_07.csv"
df = pd.read_csv(csv_file)

#"rosbag2_2026_02_24-16_31_07.csv"
#"rosbag2_2026_02_24-17_01_26.csv" 5s

# Calculate magnitude from x, y, z values
df['magnitude'] = np.sqrt(df['x']**2 + df['y']**2 + df['z']**2) * 1000000000

# Convert timestamp_ns to seconds from start
start_time_ns = df['timestamp_ns'].iloc[0]
df['time_seconds'] = (df['timestamp_ns'] - start_time_ns) / 1e9

# Create list with timestamp and magnitude
data_list = list(zip(df['timestamp_ns'], df['time_seconds'], df['magnitude']))

# Font sizes for both plots
TITLE_FONT_SIZE = 20
LABEL_FONT_SIZE = 16
TICK_FONT_SIZE = 14

# Print the list
# print("Timestamp (ns), Time (s), Magnitude:")
# print("-" * 50)
# for timestamp_ns, time_s, magnitude in data_list:
#     print(f"{timestamp_ns}, {time_s:.6f}, {magnitude:.10e}")

# Plot 1: First minute
one_minute_data = [item for item in data_list if item[1] <= 60]

times = [item[1] for item in one_minute_data]
magnitudes = [item[2] for item in one_minute_data]

plt.figure(figsize=(12, 6))
plt.plot(times, magnitudes, linewidth=1, alpha=0.8)
plt.xlabel('Zeit (Sekunden)', fontsize=LABEL_FONT_SIZE)
plt.ylabel('Magnitude (Nanotesla)', fontsize=LABEL_FONT_SIZE)
plt.title('Magnetometer Magnitude - Erste 60 Sekunden von Versuch 1', fontsize=TITLE_FONT_SIZE)
plt.xticks(fontsize=TICK_FONT_SIZE)
plt.yticks(fontsize=TICK_FONT_SIZE)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Plot 2: From 7.5 to 12.5 seconds
plot2_data = [item for item in data_list if 8 <= item[1] <= 12]

times_8_12 = [item[1] for item in plot2_data]
magnitudes_8_12 = [item[2] for item in plot2_data]

plt.figure(figsize=(12, 6))
plt.plot(times_8_12, magnitudes_8_12, linewidth=1, alpha=0.8, color='orange')
plt.xlabel('Zeit (Sekunden)', fontsize=LABEL_FONT_SIZE)
plt.ylabel('Magnitude (Nanotesla)', fontsize=LABEL_FONT_SIZE)
plt.title('Magnetometer Magnitude - Sekunden 8 bis 12', fontsize=TITLE_FONT_SIZE)
plt.xticks(fontsize=TICK_FONT_SIZE)
plt.yticks(fontsize=TICK_FONT_SIZE)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Calculate statistics for 8 to 12 seconds
stats_data = [item for item in data_list if 8 <= item[1] <= 12]
stats_magnitudes = [item[2] for item in stats_data]

mean_value = np.mean(stats_magnitudes)
median_value = np.median(stats_magnitudes)
std_dev = np.std(stats_magnitudes)
max_deviation = np.max(np.abs(stats_magnitudes - mean_value))

print("\n" + "="*50)
print("Statistiken für den Zeitbereich 8-12 Sekunden:")
print("="*50)
print(f"Anzahl der Messungen: {len(stats_magnitudes)}")
print(f"Durchschnittswert:    {mean_value:.10e}")
print(f"Median:               {median_value:.10e}")
print(f"Standardabweichung:   {std_dev:.10e}")
print(f"Maximale Abweichung:  {max_deviation:.10e}")
print("="*50)

# Show all plots simultaneously
plt.show()
