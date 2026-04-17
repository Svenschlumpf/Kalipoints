import numpy as np

EPSILON = 1e-12

# Manuelle Eingabe der beiden Punkte (nT):
# A entspricht test_point, B entspricht ref_point.
A = np.array([933.14, 3004.91, 0.0], dtype=float)
B = np.array([2199.45, 2070.29, 0.0], dtype=float)


def compute_azimuth_error_deg_for_A_B(A, B):
	# Entspricht der aktuell verwendeten Azimut-Fehlerberechnung im Programm.
	test_point = np.asarray(A, dtype=float)
	ref_point = np.asarray(B, dtype=float)

	test_xy = test_point[:2]
	ref_xy = ref_point[:2]
	norm_test = float(np.linalg.norm(test_xy))
	norm_ref = float(np.linalg.norm(ref_xy))
	if norm_test <= EPSILON or norm_ref <= EPSILON:
		return None

	dot_uv = float(ref_xy[0] * test_xy[0] + ref_xy[1] * test_xy[1])
	cos_value = dot_uv / (norm_ref * norm_test)
	cos_value = max(-1.0, min(1.0, cos_value))
	return float(np.degrees(np.arccos(cos_value)))


if __name__ == "__main__":
	azimuth_error_deg = compute_azimuth_error_deg_for_A_B(A, B)
	print(azimuth_error_deg)