"""
Generic pressure-map visualiser
───────────────────────────────
Reads an arbitrary CSV exported from a pressure-sensor grid, converts the raw
sensor values to engineering units (kPa, N, kg), crops away blank margins,
and shows the calibrated map in a single colour plot.

Author : max mccormack
Date   : 2025-06-03
"""

import csv
import numpy as np
import matplotlib.pyplot as plt


# ─── USER SETTINGS ────────────────────────────────────────────────────────────
CSV_FILEPATH            = r"soft/white toe 0.csv"  # Path to your CSV log
CAL_COEFFICIENT         = 224.8            # Device-specific calibration factor
MAX_DISPLAY_PRESSURE    = 120              # kPa – top of colour scale
SENSOR_AREA_CM2         = 0.702579         # Area of one sensel (any unit ok)
PADDING_CELLS           = 3                # Blank border to keep around ROI
# -----------------------------------------------------------------------------


# ─── UNIT-CONVERSION HELPERS ─────────────────────────────────────────────────
def raw_to_pressure(raw):
    """Convert raw ADC value (0-255) to pressure in kPa."""
    return (raw * CAL_COEFFICIENT) / 255.0


def raw_to_mass(raw):
    """Convert raw ADC value to kg‐equivalent load per sensel."""
    # 1 kPa acting on 1 cm² = 0.1 N. 1 N = 1/9.81 kgf.
    return raw_to_pressure(raw) * (SENSOR_AREA_CM2 * 0.1) / 9.81


def contact_area(matrix):
    """Return the area (same units as SENSOR_AREA_CM2) under non-zero load."""
    return np.count_nonzero(matrix) * SENSOR_AREA_CM2


# ─── CSV INGEST ──────────────────────────────────────────────────────────────
def load_sensor_csv(filepath):
    """
    Read a CSV produced by the sensor grid.
    * Non-numeric rows (headers/footers) are skipped.
    * Ragged rows are padded with NaN.
    Returns the calibrated pressure matrix (kPa) and total load estimate (kg).
    """
    numeric_rows = []
    with open(filepath, newline='', encoding='utf-8') as f:
        for row in csv.reader(f):
            if not row:
                continue
            try:
                int(row[0].strip())                       # data row?
                numeric_rows.append(
                    [float(c.strip()) if c.strip() else np.nan for c in row]
                )
            except ValueError:
                # Header/footer – ignore
                continue

    if not numeric_rows:
        raise RuntimeError("No data rows found in CSV.")

    # Pad to rectangular matrix
    max_len = max(len(r) for r in numeric_rows)
    padded  = [r + [np.nan]*(max_len-len(r)) for r in numeric_rows]
    raw_mat = np.array(padded, dtype=float)

    pressure_mat = raw_to_pressure(raw_mat)
    total_load_kg = np.nansum(raw_to_mass(raw_mat))
    return pressure_mat, total_load_kg


# ─── CROPPING + VISUALISATION ────────────────────────────────────────────────
def show_pressure_map(matrix,
                      title="Calibrated Pressure Map",
                      padding=PADDING_CELLS,
                      vmax=MAX_DISPLAY_PRESSURE):
    """
    Display a single colour map of the sensor grid, cropped to non-zero region
    plus a configurable padding border. Zero cells are white.
    Axes are scaled to show millimeter distances.
    """
    CELL_SIZE_MM = 4  # Each cell is 4x4 mm

    if matrix.size == 0 or np.all((matrix == 0) | np.isnan(matrix)):
        print("Nothing to display – all cells are zero/NaN.")
        return

    # Find bounding box of active sensels
    active = ~np.isnan(matrix) & (matrix != 0)
    r_min, r_max = np.where(np.any(active, axis=1))[0][[0, -1]]
    c_min, c_max = np.where(np.any(active, axis=0))[0][[0, -1]]

    r0, r1 = max(0, r_min - padding), min(matrix.shape[0], r_max + padding + 1)
    c0, c1 = max(0, c_min - padding), min(matrix.shape[1], c_max + padding + 1)
    r1 = r0 + 35
	
    cropped = matrix[r0:r1, c0:c1]
    masked  = np.ma.masked_where(cropped == 0, cropped)

    cmap = plt.cm.get_cmap("turbo").copy()
    cmap.set_bad(color="white")

    fig, ax = plt.subplots(figsize=(8, 7))
    img = ax.imshow(masked, cmap=cmap, vmin=0, vmax=vmax,
                    interpolation="nearest", aspect="equal",
                    extent=[c0*CELL_SIZE_MM, c1*CELL_SIZE_MM,
                            r1*CELL_SIZE_MM, r0*CELL_SIZE_MM])  # Flip y-axis

    ax.set_title(title)
    ax.set_xlabel("X position (mm)")
    ax.set_ylabel("Y position (mm)")
    plt.colorbar(img, label="Pressure (kPa)", ax=ax)

    # Peak annotation
    peak_val = np.nanmax(cropped)
    peak_val = 0
    if peak_val > 0:
        pr, pc = np.unravel_index(np.nanargmax(cropped), cropped.shape)
        ax.annotate(f"Peak: {peak_val:.2f} kPa",
                    xy=((c0 + pc) * CELL_SIZE_MM, (r0 + pr) * CELL_SIZE_MM),
                    xytext=((c0 + pc + 2) * CELL_SIZE_MM, (r0 + pr + 2) * CELL_SIZE_MM),
                    arrowprops=dict(facecolor="grey", edgecolor="grey", width=2),
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=0.5))

    plt.tight_layout()
    plt.show()


# ─── MAIN WORKFLOW ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    pressure_matrix, total_load = load_sensor_csv(CSV_FILEPATH)

    print("── Sensor-grid summary ───────────────────────────────")
    print(f"Matrix shape           : {pressure_matrix.shape}")
    print(f"Estimated total load   : {total_load:.2f} kg")
    print(f"Max pressure           : {np.nanmax(pressure_matrix):.2f} kPa")
    print(f"Contact area           : {contact_area(pressure_matrix):.2f} cm²")
    if np.any(pressure_matrix > 0):
        mean_p = np.nanmean(pressure_matrix[pressure_matrix > 0])
        print(f"Mean pressure          : {mean_p:.2f} kPa")
    print("──────────────────────────────────────────────────────\n")

    show_pressure_map(pressure_matrix,
                      title="Calibrated Pressure Map (cropped)")