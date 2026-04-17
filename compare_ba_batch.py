import os
import pandas as pd


def parse_three_values(raw_value: str):
    """Parse comma-separated triple values and return exactly 3 strings."""
    parts = [part.strip() for part in raw_value.split(',')]
    if len(parts) < 3:
        parts.extend([''] * (3 - len(parts)))
    return parts[:3]


def read_file_metadata(file_path: str):
    """Read metadata from the commented header block of an export CSV file."""
    metadata = {
        'Eingestellte_Punkteanzahl(POINT_AMOUNT)': '',
        'Achseneinschränkung(AXIS_CONSTRAINT)': '',
        'Winkeleinschränkung(ANGULAR_CONSTRAINT-DEG)': '',
        'HI-x': '',
        'HI-y': '',
        'HI-z': '',
        'SIV-x': '',
        'SIV-y': '',
        'SIV-z': '',
        'SIR-x': '',
        'SIR-y': '',
        'SIR-z': ''
    }

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if not line.startswith('#'):
                    # Metadata block is at the top; stop when data section starts.
                    if line.startswith('X;Y;Z'):
                        break
                    continue

                content = line[1:].strip()
                if ':' not in content:
                    continue

                key, value = content.split(':', 1)
                key = key.strip()
                value = value.strip()

                if key == 'POINT_AMOUNT':
                    metadata['Eingestellte_Punkteanzahl(POINT_AMOUNT)'] = value
                elif key == 'AXIS_CONSTRAINT':
                    metadata['Achseneinschränkung(AXIS_CONSTRAINT)'] = value
                elif key == 'ANGULAR_CONSTRAINT-DEG':
                    metadata['Winkeleinschränkung(ANGULAR_CONSTRAINT-DEG)'] = value
                elif key == 'HI-X-Y-Z-OFFSET':
                    hi_x, hi_y, hi_z = parse_three_values(value)
                    metadata['HI-x'] = hi_x
                    metadata['HI-y'] = hi_y
                    metadata['HI-z'] = hi_z
                elif key == 'SI-X-Y-Z-DISTORTION':
                    siv_x, siv_y, siv_z = parse_three_values(value)
                    metadata['SIV-x'] = siv_x
                    metadata['SIV-y'] = siv_y
                    metadata['SIV-z'] = siv_z
                elif key == 'SI-X-Y-Z-ROTATION-DEG':
                    sir_x, sir_y, sir_z = parse_three_values(value)
                    metadata['SIR-x'] = sir_x
                    metadata['SIR-y'] = sir_y
                    metadata['SIR-z'] = sir_z
    except Exception as exc:
        print(f"Warnung: Metadaten konnten nicht gelesen werden für {os.path.basename(file_path)}: {exc}")

    return metadata

# Pfade definieren
export_dir = r"c:\Users\Sven\Desktop\GitHub\BA\datasets\1-kalipoints_exports\BA-Batch"
calib_dir = r"c:\Users\Sven\Desktop\GitHub\BA\datasets\3-calibration_results\BA-Batch"
output_file = r"c:\Users\Sven\Desktop\GitHub\BA\datasets\Not-Kalibrated_BA-Batch.csv"

# Dateien auslesen
export_files = set(f for f in os.listdir(export_dir) if f.endswith('.csv'))
calib_files = set(f for f in os.listdir(calib_dir) if f.endswith('.csv'))

# Nicht kalibrierte Dateien finden
not_calibrated = sorted(export_files - calib_files)

print(f"Exportierte Dateien: {len(export_files)}")
print(f"Kalibrierte Dateien: {len(calib_files)}")
print(f"Nicht kalibrierte Dateien: {len(not_calibrated)}")

# Originale Spalten definieren
columns = ['Name', 'Eingestellte_Punkteanzahl(POINT_AMOUNT)', 
           'Achseneinschränkung(AXIS_CONSTRAINT)', 
           'Winkeleinschränkung(ANGULAR_CONSTRAINT-DEG)',
           'HI-x', 'HI-y', 'HI-z', 'SIV-x', 'SIV-y', 'SIV-z', 
           'SIR-x', 'SIR-y', 'SIR-z']

# DataFrame mit leeren Werten für andere Spalten erstellen
data = []
for name in sorted(not_calibrated):
    source_file = os.path.join(export_dir, name)
    file_metadata = read_file_metadata(source_file)
    row = [
        name,
        file_metadata['Eingestellte_Punkteanzahl(POINT_AMOUNT)'],
        file_metadata['Achseneinschränkung(AXIS_CONSTRAINT)'],
        file_metadata['Winkeleinschränkung(ANGULAR_CONSTRAINT-DEG)'],
        file_metadata['HI-x'],
        file_metadata['HI-y'],
        file_metadata['HI-z'],
        file_metadata['SIV-x'],
        file_metadata['SIV-y'],
        file_metadata['SIV-z'],
        file_metadata['SIR-x'],
        file_metadata['SIR-y'],
        file_metadata['SIR-z']
    ]
    data.append(row)

df = pd.DataFrame(data, columns=columns)
df.to_csv(output_file, index=False)
print(f"\n✓ {output_file} aktualisiert mit {len(not_calibrated)} nicht kalibrierten Dateien")
print(f"✓ Spalten: {', '.join(columns[:5])} ... ({len(columns)} insgesamt)")
