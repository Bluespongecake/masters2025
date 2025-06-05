import os
import sys
import csv
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# Optional drag-and-drop support
try:
    from tkinterdnd2 import TkinterDnD, DND_FILES  # pip install tkinterdnd2
except ImportError:  # continue without DnD
    TkinterDnD = None
    DND_FILES = None

import numpy as np
import matplotlib.pyplot as plt

# ────────────────────────────────────────────────
#  Calibration constants (unchanged)
# ────────────────────────────────────────────────
CALIBRATION_COEFFICIENT = 215
CELL_AREA_CM2 = 0.702579
CELL_AREA_M2 = CELL_AREA_CM2 * 1e-4
CELL_AREA_N_PER_KPA = CELL_AREA_M2 * 1000  # kPa→Pa × area → N
SCALE_MAX = 200  # heat-map colour scale

# ────────────────────────────────────────────────
#  Mapping helpers
# ────────────────────────────────────────────────

def map_raw_to_pressure(raw):
    """Convert raw sensor value (0-255) to calibrated pressure in kPa."""
    return (raw * CALIBRATION_COEFFICIENT) / 255.0


def map_raw_to_force(raw):
    """Convert raw sensor value (0-255) to force in newtons for the cell area."""
    return (raw * CALIBRATION_COEFFICIENT * CELL_AREA_N_PER_KPA) / 255.0

# ────────────────────────────────────────────────
#  CSV loader (raw matrix + total force, N)
# ────────────────────────────────────────────────

def load_csv(path: str):
    """Load Tekscan CSV and return raw matrix (float, NaN for blanks) and total force."""
    raw_rows = []
    with open(path, newline="", encoding="utf-8") as f:
        rdr = csv.reader(f)
        for row in rdr:
            if not row:
                continue
            try:
                int(row[0].strip())  # strip potential label/header rows
                raw_rows.append([float(c) if c.strip() else np.nan for c in row])
            except ValueError:
                continue  # header/footer lines
    if not raw_rows:
        raise ValueError("No numeric data rows found in CSV.")

    max_len = max(len(r) for r in raw_rows)
    raw_np = np.array(
        [r + [np.nan] * (max_len - len(r)) for r in raw_rows], dtype=float
    )
    total_force_csv = np.nansum(map_raw_to_force(raw_np))
    return raw_np, total_force_csv

# ────────────────────────────────────────────────
#  Split heuristic – zero column or middle
# ────────────────────────────────────────────────

def find_split_column(raw):
    """Locate a separating column: a column of all zeros or the centre column."""
    rows, cols = raw.shape
    if cols <= 1:
        return cols, False
    zeros = [c for c in range(cols) if np.all(raw[:, c] == 0)]
    if zeros:
        return min(zeros, key=lambda c: abs(c - (cols - 1) / 2)), True
    return cols // 2, False

# ────────────────────────────────────────────────
#  Cluster labelling (4-connectivity)
# ────────────────────────────────────────────────
from collections import deque

def label_clusters(binary):
    visited = np.zeros_like(binary, bool)
    labels = np.zeros_like(binary, int)
    areas = []
    h, w = binary.shape
    cid = 0
    for r in range(h):
        for c in range(w):
            if binary[r, c] and not visited[r, c]:
                cid += 1
                q, area = deque([(r, c)]), 0
                visited[r, c] = True
                while q:
                    x, y = q.popleft()
                    labels[x, y] = cid
                    area += 1
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nx, ny = x + dx, y + dy
                        if (
                            0 <= nx < h
                            and 0 <= ny < w
                            and binary[nx, ny]
                            and not visited[nx, ny]
                        ):
                            visited[nx, ny] = True
                            q.append((nx, ny))
                areas.append(area)
    return labels, areas

# ────────────────────────────────────────────────
#  Plot helpers (regular Matplotlib windows)
# ────────────────────────────────────────────────

def make_heatmap_fig(data, labels, title="Pressure map"):
    fig, ax = plt.subplots(figsize=(6, 5))
    masked = np.ma.masked_where(data == 0, data)
    cmap = plt.cm.get_cmap("turbo").copy()
    cmap.set_bad("white")
    im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=SCALE_MAX, aspect="equal")
    fig.colorbar(im, ax=ax, label="Pressure (kPa)")

    # Draw bounding boxes around labelled toe clusters
    if labels.max() > 0:
        colours = plt.cm.get_cmap("tab10", max(10, labels.max()))
        for cid in range(1, labels.max() + 1):
            rows, cols = np.where(labels == cid)
            if not rows.size:
                continue
            r0, r1 = rows.min(), rows.max()
            c0, c1 = cols.min(), cols.max()
            ax.add_patch(
                plt.Rectangle(
                    (c0 - 0.5, r0 - 0.5),
                    (c1 - c0) + 1,
                    (r1 - r0) + 1,
                    linewidth=2,
                    edgecolor=colours(cid - 1),
                    facecolor="none",
                )
            )
            ax.text(
                c0 - 0.4,
                r0 - 0.6,
                f"Toe {cid}",
                color=colours(cid - 1),
                fontsize=8,
                weight="bold",
            )
    ax.set(title=title, xlabel="Col", ylabel="Row")
    fig.tight_layout()
    return fig


def make_area_fig(area_cm2, title="Contact area per toe"):
    toes = np.arange(len(area_cm2))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(toes, area_cm2, width=0.6)
    ax.set_ylabel("Area (cm²)")
    ax.set_xticks(toes)
    ax.set_xticklabels([f"Toe {i + 1}" for i in toes])
    ax.set_title(title)
    fig.tight_layout()
    return fig


def make_force_fig(force_N, title="Force per toe"):
    toes = np.arange(len(force_N))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(toes, force_N, width=0.6)
    ax.set_ylabel("Force (N)")
    ax.set_xticks(toes)
    ax.set_xticklabels([f"Toe {i + 1}" for i in toes])
    ax.set_title(title)
    fig.tight_layout()
    return fig

# NEW: pressure distribution helpers
# ────────────────────────────────────────────────

def calc_row_pressure_sums(data):
    """Return 1-D array with sum of pressure (kPa) for each row, excluding zero cells."""
    # Replace NaN with 0 for summation and mask out zeros afterwards
    safe = np.nan_to_num(data, nan=0.0)
    row_sums = np.sum(np.where(safe > 0, safe, 0.0), axis=1)
    return row_sums


def make_pressure_distribution_fig(row_sums, title="Pressure distribution along rows"):
    rows = np.arange(len(row_sums))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(rows, row_sums, marker="o")
    ax.set_xlabel("Row")
    ax.set_ylabel("Sum of pressure (kPa)")
    ax.set_title(title)
    fig.tight_layout()
    return fig

# ────────────────────────────────────────────────
#  GUI app
# ────────────────────────────────────────────────


class FootScanApp((TkinterDnD.Tk if TkinterDnD else tk.Tk)):
    def __init__(self):
        super().__init__()
        self.title("Foot-scan Viewer")
        self.geometry("540x300")
        self.resizable(False, False)

        # Drop / browse frame
        frame = ttk.Frame(self, padding=20, relief="groove")
        frame.pack(fill="both", expand=True, padx=20, pady=15)
        prompt = "📂  Drop a Tekscan CSV here" if TkinterDnD else ""
        prompt += "\n…or click Browse"
        self.prompt = ttk.Label(frame, text=prompt, anchor="center", font=("Segoe UI", 11))
        self.prompt.pack(expand=True)

        if TkinterDnD:
            frame.drop_target_register(DND_FILES)
            frame.dnd_bind("<<Drop>>", self._on_drop)

        ttk.Button(self, text="Browse…", command=self._browse).pack()
        # stats label
        self.stats = ttk.Label(self, text="No file loaded", anchor="w", justify="left")
        self.stats.pack(fill="x", padx=10, pady=(10, 5))

    # ── Helpers ─────────────────────────────────────────────────────────
    def _browse(self):
        path = filedialog.askopenfilename(
            title="Choose Tekscan CSV", filetypes=[("CSV", "*.csv")]
        )
        if path:
            self._process(path)

    def _on_drop(self, event):
        p = self.tk.splitlist(event.data)[0].strip("{}")
        if os.path.isfile(p):
            self._process(p)

    # ── Main processing ────────────────────────────────────────────────
    def _process(self, path):
        plt.close("all")
        self.prompt.configure(text="Processing…")
        self.update_idletasks()
        try:
            raw_np, total_force_csv = load_csv(path)
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to load CSV:\n{exc}")
            self.prompt.configure(text="Drop CSV here / Browse…")
            self.stats.configure(text="No file loaded")
            return

        split_col, is_sep = find_split_column(raw_np)
        left_raw = raw_np[:, :split_col] if split_col else np.empty((raw_np.shape[0], 0))
        right_raw = raw_np[:, split_col + 1 :] if is_sep else raw_np[:, split_col:]

        vmap = np.vectorize(map_raw_to_pressure)
        left_kpa = vmap(left_raw) if left_raw.size else None
        right_kpa = vmap(right_raw) if right_raw.size else None

        # Calibrated pressure matrices (kPa)
        data = left_kpa if left_kpa is not None and np.count_nonzero(left_kpa) else right_kpa
        side = "Left" if data is left_kpa else "Right"

        labels, areas = label_clusters(data > 0)
        area_cm2 = [a * CELL_AREA_CM2 for a in areas]
        force_N = [
            np.nansum(data[labels == i + 1] * CELL_AREA_N_PER_KPA) for i in range(len(areas))
        ]

        # NEW: row-wise pressure distribution
        row_sums = calc_row_pressure_sums(data)

        # Build Matplotlib windows (heat-map + three separate charts)
        fig_heat = make_heatmap_fig(data, labels, f"{side} – heat-map")
        fig_area = make_area_fig(area_cm2, "Toe contact area")
        fig_force = make_force_fig(force_N, "Toe force")
        fig_distribution = make_pressure_distribution_fig(
            row_sums, "Pressure distribution (row sum)"
        )

        # Show all figures without blocking the GUI
        for fig in (fig_heat, fig_area, fig_force, fig_distribution):
            fig.show()
        plt.show(block=False)

        stats_txt = (
            f"File: {os.path.basename(path)}\n"
            f"Side analysed: {side}\n"
            f"Contact area: {sum(area_cm2):.2f} cm²\n"
            f"Max pressure: {np.nanmax(data):.2f} kPa\n"
            f"Average pressure (contact): {np.nanmean(data[data > 0]):.2f} kPa\n"
            f"Total force (CSV): {total_force_csv:.1f} N  (~{total_force_csv / 9.81:.2f} kg)\n"
            f"Sum of toe forces: {sum(force_N):.1f} N"
        )
        self.stats.configure(text=stats_txt)
        self.prompt.configure(text="Drop another CSV or Browse…")


# ─────────────────────────────────────────────────────────────────────
#  Run
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    FootScanApp().mainloop()
