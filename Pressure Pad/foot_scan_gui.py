import os
import csv
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# Optional drag-and-drop support
try:
    from tkinterdnd2 import TkinterDnD, DND_FILES  # pip install tkinterdnd2 for drag-and-drop
except ImportError:  # continue gracefully without drag-and-drop
    TkinterDnD = None
    DND_FILES = None

import numpy as np
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from collections import deque

# Ensure Tk backend for Matplotlib (important on macOS/WSL)
matplotlib.use("TkAgg")

# ────────────────────────────────────────────────
#  Calibration constants (unchanged)
# ────────────────────────────────────────────────
CALIBRATION_COEFFICIENT = 215
CELL_AREA_CM2 = 0.702579
CELL_AREA_M2 = CELL_AREA_CM2 * 1e-4
CELL_AREA_N_PER_KPA = CELL_AREA_M2 * 1000  # kPa→Pa × area → N
SCALE_MAX = 120  # heat-map colour scale

# CSV summary file
OUTPUT_CSV = "foot pressure distributions.csv"

# ────────────────────────────────────────────────
#  Mapping helpers
# ────────────────────────────────────────────────

def map_raw_to_pressure(raw: np.ndarray) -> np.ndarray:
    """Convert raw sensor (0-255) to calibrated pressure (kPa)."""
    return (raw * CALIBRATION_COEFFICIENT) / 255.0


def map_raw_to_force(raw: np.ndarray) -> np.ndarray:
    """Convert raw sensor (0-255) to force (N) given cell area."""
    return (raw * CALIBRATION_COEFFICIENT * CELL_AREA_N_PER_KPA) / 255.0

# ────────────────────────────────────────────────
#  CSV loader
# ────────────────────────────────────────────────

def load_csv(path: str):
    """Load Tekscan CSV → (raw matrix, total force)."""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        rdr = csv.reader(f)
        for row in rdr:
            if not row:
                continue
            try:
                int(row[0].strip())  # numeric test – ignore header/footer
                rows.append([float(c) if c.strip() else np.nan for c in row])
            except ValueError:
                continue
    if not rows:
        raise ValueError("No numeric data rows found in CSV.")
    max_len = max(len(r) for r in rows)
    raw_np = np.array([r + [np.nan] * (max_len - len(r)) for r in rows], dtype=float)
    total_force = np.nansum(map_raw_to_force(raw_np))
    return raw_np, total_force

# ────────────────────────────────────────────────
#  Split heuristic – zero column or centre column
# ────────────────────────────────────────────────

def find_split_column(raw):
    rows, cols = raw.shape
    if cols <= 1:
        return cols, False
    zero_cols = [c for c in range(cols) if np.all(raw[:, c] == 0)]
    if zero_cols:
        return min(zero_cols, key=lambda c: abs(c - (cols - 1) / 2)), True
    return cols // 2, False

# ────────────────────────────────────────────────
#  Cluster labelling (4-connectivity)
# ────────────────────────────────────────────────

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
                    x, y = q.popleft(); labels[x, y] = cid; area += 1
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < h and 0 <= ny < w and binary[nx, ny] and not visited[nx, ny]:
                            visited[nx, ny] = True; q.append((nx, ny))
                areas.append(area)
    return labels, areas

# ────────────────────────────────────────────────
#  Plot helpers
# ────────────────────────────────────────────────

def make_heatmap_fig(data, labels, title="Pressure map"):
    fig, ax = plt.subplots(figsize=(6, 5))
    masked = np.ma.masked_where(data == 0, data)
    cmap = plt.cm.get_cmap("turbo").copy(); cmap.set_bad("white")
    im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=SCALE_MAX, aspect="equal")
    fig.colorbar(im, ax=ax, label="Pressure (kPa)")
    if labels.max() > 0:
        colours = plt.cm.get_cmap("tab10", max(10, labels.max()))
        for cid in range(1, labels.max() + 1):
            rows, cols = np.where(labels == cid)
            if not rows.size:
                continue
            r0, r1 = rows.min(), rows.max(); c0, c1 = cols.min(), cols.max()
            ax.add_patch(plt.Rectangle((c0 - 0.5, r0 - 0.5), (c1 - c0) + 1, (r1 - r0) + 1,
                                        linewidth=2, edgecolor=colours(cid - 1), facecolor="none"))
            ax.text(c0 - 0.4, r0 - 0.6, f"Toe {cid}", color=colours(cid - 1), fontsize=8, weight="bold")
    ax.set(title=title, xlabel="Col", ylabel="Row"); fig.tight_layout(); return fig


def make_bar_fig(values, ylabel, title):
    toes = np.arange(len(values))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(toes, values, width=0.6)
    ax.set_ylabel(ylabel)
    ax.set_xticks(toes)
    ax.set_xticklabels([f"Toe {i + 1}" for i in toes])
    ax.set_title(title)
    fig.tight_layout(); return fig


def make_force_distribution_fig(row_force, title="Row-wise force distribution"):
    rows = np.arange(len(row_force))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(rows, row_force, marker="o")
    ax.set_xlabel("Row"); ax.set_ylabel("Force (N)"); ax.set_title(title)
    fig.tight_layout(); return fig

# ────────────────────────────────────────────────
#  Row-wise force (equal-area assumption)
# ────────────────────────────────────────────────

def calc_row_force_equal_area(data):
    safe = np.nan_to_num(data, nan=0.0)
    nnz = np.count_nonzero(safe > 0, axis=1)
    row_sum_kpa = np.sum(np.where(safe > 0, safe, 0.0), axis=1)
    active = nnz > 0
    if not np.any(active):
        return np.zeros_like(row_sum_kpa)
    avg_width = int(round(nnz[active].mean()))
    row_area = avg_width * CELL_AREA_M2  # m²
    with np.errstate(divide="ignore", invalid="ignore"):
        avg_kpa = np.where(nnz > 0, row_sum_kpa / nnz, 0.0)
    row_force = avg_kpa * 1000 * row_area  # kPa→Pa * area
    return row_force

# ────────────────────────────────────────────────
#  Summary CSV writer
# ────────────────────────────────────────────────
def append_summary_csv(
        file_name: str,
        toe_forces_n: list[float],   # expected length = 4
        toe_areas_cm2: list[float],  # expected length = 4
        total_force_n: float,
        total_area_cm2: float
):
    """
    Append a single line to the summary CSV with:
        filename,
        Toe-1 … Toe-4 forces (N),
        Toe-1 … Toe-4 areas (cm²),
        total force (N),
        total contact area (cm²)
    """
    # Make sure we always have four items (pad with zeros if fewer)
    toe_forces_n  = (toe_forces_n  + [0.0]*4)[:4]
    toe_areas_cm2 = (toe_areas_cm2 + [0.0]*4)[:4]

    write_header = not os.path.isfile(OUTPUT_CSV)

    with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # ── header (once) ──────────────────────────
        if write_header:
            header = (
                ["Filename"]
                + [f"Toe {i} force (N)" for i in range(1, 5)]
                + [f"Toe {i} area (cm²)" for i in range(1, 5)]
                + ["Total force (N)", "Total area (cm²)"]
            )
            writer.writerow(header)

        # ── data row ───────────────────────────────
        row = (
            [file_name]
            + [f"{f:.1f}" for f in toe_forces_n]
            + [f"{a:.2f}" for a in toe_areas_cm2]
            + [f"{total_force_n:.1f}", f"{total_area_cm2:.2f}"]
        )
        writer.writerow(row)

# ────────────────────────────────────────────────
#  GUI Application
# ────────────────────────────────────────────────

class FootScanApp((TkinterDnD.Tk if TkinterDnD else tk.Tk)):
    """Simple viewer for Tekscan CSV files with optional embedded charts."""

    def __init__(self):
        super().__init__()
        self.title("Foot-scan Viewer")
        self.geometry("620x420")
        self.minsize(560, 360)

        # ── Control bar ───────────────────────────
        ctrl = ttk.Frame(self, padding=8); ctrl.pack(fill="x")
        ttk.Button(ctrl, text="Open CSV…", command=self._browse).pack(side="left")
        self.embed_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(ctrl, text="Embed plots in window", variable=self.embed_var).pack(side="left", padx=10)

        # ── Main frame ────────────────────────────
        self.main_frame = ttk.Frame(self, padding=15, relief="groove")
        self.main_frame.pack(fill="both", expand=True, padx=15, pady=(0, 10))

        prompt_txt = "📂  Drop a Tekscan CSV here" if TkinterDnD else ""
        prompt_txt += "\n…or click Open CSV"
        self.prompt_lbl = ttk.Label(self.main_frame, text=prompt_txt, anchor="center", font=("Segoe UI", 11))
        self.prompt_lbl.pack(expand=True)

        if TkinterDnD:
            self.main_frame.drop_target_register(DND_FILES)
            self.main_frame.dnd_bind("<<Drop>>", self._on_drop)

        self.stats_lbl = ttk.Label(self, text="No file loaded", anchor="w", justify="left")
        self.stats_lbl.pack(fill="x", padx=10, pady=(0, 10))

        # Container for embedded canvases (created lazily)
        self.canvas_container = None

    # ────────────────────────────────────────────
    #  Drag-and-drop / file dialogs
    # ────────────────────────────────────────────
    def _browse(self):
        path = filedialog.askopenfilename(title="Choose Tekscan CSV", filetypes=[("CSV", "*.csv")])
        if path:
            self._process(path)

    def _on_drop(self, event):
        p = self.tk.splitlist(event.data)[0].strip("{}")
        if os.path.isfile(p):
            self._process(p)

    # ────────────────────────────────────────────
    #  Embedded-figure utilities
    # ────────────────────────────────────────────
    def _clear_embedded(self):
        if self.canvas_container is not None:
            for child in self.canvas_container.winfo_children():
                child.destroy()
            self.canvas_container.destroy()
            self.canvas_container = None

    def _embed_figures(self, figs):
        self._clear_embedded()
        self.canvas_container = ttk.Frame(self.main_frame)
        self.canvas_container.pack(fill="both", expand=True)
        for i, fig in enumerate(figs):
            canvas = FigureCanvasTkAgg(fig, master=self.canvas_container)
            widget = canvas.get_tk_widget(); widget.grid(row=i // 2, column=i % 2, padx=5, pady=5, sticky="nsew")
            canvas.draw()
        # Responsive grid
        for c in (0, 1):
            self.canvas_container.columnconfigure(c, weight=1)
        rows = (len(figs) + 1) // 2
        for r in range(rows):
            self.canvas_container.rowconfigure(r, weight=1)

    # ────────────────────────────────────────────
    #  Core processing
    # ────────────────────────────────────────────
    def _process(self, path):
        """
        Analyse one Tekscan CSV, display the usual plots, and append the new
        summary line (filename, Toe-1…4 forces & areas, totals) to
        foot pressure distributions.csv.
        """
        # ── housekeeping ────────────────────────────────────────────────────────
        plt.close("all")
        self.prompt_lbl.configure(text="Processing…")
        self.update_idletasks()

        # ── 1. Load raw data ────────────────────────────────────────────────────
        try:
            raw_np, _ = load_csv(path)               # we'll recompute forces per foot
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to load CSV: {exc}")
            self.prompt_lbl.configure(text="Drop CSV here / Open CSV")
            self.stats_lbl.configure(text="No file loaded")
            return

        # ── 2. Split left/right and pick the active side (unchanged logic) ─────
        split_col, is_sep = find_split_column(raw_np)
        left_raw  = raw_np[:, :split_col] if split_col else np.empty((raw_np.shape[0], 0))
        right_raw = raw_np[:, split_col + 1:] if is_sep else raw_np[:, split_col:]

        vmap = np.vectorize(map_raw_to_pressure)
        left_kpa  = vmap(left_raw)  if left_raw.size  else None
        right_kpa = vmap(right_raw) if right_raw.size else None
        data = left_kpa if left_kpa is not None and np.count_nonzero(left_kpa) else right_kpa
        side = "Left" if data is left_kpa else "Right"

        # ── 3. Label toe clusters ──────────────────────────────────────────────
        labels, cluster_areas = label_clusters(data > 0)
        toe_areas_cm2 = [a * CELL_AREA_CM2 for a in cluster_areas]
        toe_forces_n  = [np.nansum(data[labels == i + 1] * CELL_AREA_N_PER_KPA)
                     for i in range(len(cluster_areas))]

        # Ensure exactly four entries (pad with zeros or truncate)
        toe_areas_cm2 = (toe_areas_cm2 + [0.0] * 4)[:4]
        toe_forces_n  = (toe_forces_n  + [0.0] * 4)[:4]

        # ── 4. Totals for the analysed side ─────────────────────────────────────
        total_area_cm2 = float(np.count_nonzero(data > 0) * CELL_AREA_CM2)
        total_force_n  = float(np.nansum(data * CELL_AREA_N_PER_KPA))

        # ── 5. Plots (as before) ────────────────────────────────────────────────
        row_force = calc_row_force_equal_area(data)
        figs = [
            make_heatmap_fig(data, labels, f"{side} – heat-map"),
            make_bar_fig(toe_areas_cm2, "Area (cm²)", "Toe contact area"),
            make_bar_fig(toe_forces_n, "Force (N)", "Toe force"),
            make_force_distribution_fig(row_force, "Row-wise force distribution (equal area)")
        ]

        if self.embed_var.get():
            self._embed_figures(figs)
        else:
            self._clear_embedded()
            for fig in figs:
                fig.show()
            plt.show(block=False)

        # ── 6. Write summary line using the new CSV schema ──────────────────────
        append_summary_csv(
            file_name      = os.path.basename(path),
            toe_forces_n   = toe_forces_n,
            toe_areas_cm2  = toe_areas_cm2,
            total_force_n  = total_force_n,
            total_area_cm2 = total_area_cm2,
        )

        # ── 7. GUI stats read-out ───────────────────────────────────────────────
        stats = (
            f"File: {os.path.basename(path)}\n"
            f"Side analysed: {side}\n"
            f"Contact area: {total_area_cm2:.2f} cm²\n"
            f"Max pressure: {np.nanmax(data):.2f} kPa\n"
            f"Average pressure (contact): {np.nanmean(data[data > 0]):.2f} kPa\n"
            f"Total force: {total_force_n:.1f} N  (~{total_force_n / 9.81:.2f} kg)\n"
            f"Toe-1…4 forces: {', '.join(f'{f:.1f}' for f in toe_forces_n)} N"
        )
        self.stats_lbl.configure(text=stats)
        self.prompt_lbl.configure(text="Drop another CSV or Open CSV")

# ────────────────────────────────────────────────
#  Entrypoint
# ────────────────────────────────────────────────
if __name__ == "__main__":
    FootScanApp().mainloop()
