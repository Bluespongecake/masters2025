import os
import csv
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# Optional drag-and-drop support
try:
    from tkinterdnd2 import TkinterDnD, DND_FILES  # pip install tkinterdnd2
except ImportError:
    TkinterDnD = None
    DND_FILES = None

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import deque

# ────────────────────────────────────────────────
#  Calibration constants (unchanged)
# ────────────────────────────────────────────────
CALIBRATION_COEFFICIENT = 215
CELL_AREA_CM2 = 0.702579
CELL_AREA_M2 = CELL_AREA_CM2 * 1e-4
CELL_AREA_N_PER_KPA = CELL_AREA_M2 * 1000  # kPa→Pa × area → N
SCALE_MAX = 200  # heat‑map colour scale

# ────────────────────────────────────────────────
#  Mapping helpers
# ────────────────────────────────────────────────

def map_raw_to_pressure(raw: np.ndarray) -> np.ndarray:
    return (raw * CALIBRATION_COEFFICIENT) / 255.0


def map_raw_to_force(raw: np.ndarray) -> np.ndarray:
    return (raw * CALIBRATION_COEFFICIENT * CELL_AREA_N_PER_KPA) / 255.0

# ────────────────────────────────────────────────
#  CSV loader
# ────────────────────────────────────────────────

def load_csv(path: str):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        rdr = csv.reader(f)
        for row in rdr:
            if not row:
                continue
            try:
                int(row[0].strip())
                rows.append([float(c) if c.strip() else np.nan for c in row])
            except ValueError:
                continue
    if not rows:
        raise ValueError("No numeric data rows found in CSV.")
    max_len = max(len(r) for r in rows)
    raw_np = np.array([r + [np.nan] * (max_len - len(r)) for r in rows], dtype=float)
    total_force_csv = np.nansum(map_raw_to_force(raw_np))
    return raw_np, total_force_csv

# ────────────────────────────────────────────────
#  Split heuristic
# ────────────────────────────────────────────────

def find_split_column(raw):
    _, cols = raw.shape
    if cols <= 1:
        return cols, False
    zeros = [c for c in range(cols) if np.all(raw[:, c] == 0)]
    if zeros:
        return min(zeros, key=lambda c: abs(c - (cols - 1) / 2)), True
    return cols // 2, False

# ────────────────────────────────────────────────
#  Cluster labelling (4‑connectivity)
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

def make_heatmap_fig(data, labels, title):
    fig, ax = plt.subplots(figsize=(6, 5))
    cmap = plt.cm.get_cmap("turbo").copy(); cmap.set_bad("white")
    masked = np.ma.masked_where(data == 0, data)
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
    ax.set(title=title, xlabel="Col", ylabel="Row")
    fig.tight_layout(); return fig


def make_bar_fig(values, ylabel, title):
    toes = np.arange(len(values))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(toes, values, width=0.6)
    ax.set_ylabel(ylabel)
    ax.set_xticks(toes)
    ax.set_xticklabels([f"Toe {i + 1}" for i in toes])
    ax.set_title(title)
    fig.tight_layout(); return fig


def make_force_distribution_fig(row_force, title):
    rows = np.arange(len(row_force))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(rows, row_force, marker="o")
    ax.set_xlabel("Row"); ax.set_ylabel("Force (N)"); ax.set_title(title)
    fig.tight_layout(); return fig

# ────────────────────────────────────────────────
#  Row‑wise force (equal‑area assumption)
# ────────────────────────────────────────────────

def calc_row_force_equal_area(data):
    safe = np.nan_to_num(data, nan=0.0)
    nnz = np.count_nonzero(safe > 0, axis=1)
    row_pressure_sum_kpa = np.sum(np.where(safe > 0, safe, 0.0), axis=1)
    active_rows = nnz > 0
    if not np.any(active_rows):
        return np.zeros_like(row_pressure_sum_kpa)
    avg_width_cells = int(round(nnz[active_rows].mean()))
    row_area_m2 = avg_width_cells * CELL_AREA_M2
    with np.errstate(divide="ignore", invalid="ignore"):
        avg_pressure_kpa = np.where(nnz > 0, row_pressure_sum_kpa / nnz, 0.0)
    row_force_n = avg_pressure_kpa * 1000 * row_area_m2
    return row_force_n

# ────────────────────────────────────────────────
#  GUI Application
# ────────────────────────────────────────────────

class FootScanApp((TkinterDnD.Tk if TkinterDnD else tk.Tk)):
    def __init__(self):
        super().__init__()
        self.title("Foot‑scan Viewer"); self.geometry("560x360"); self.resizable(True, False)

        # Top frame for controls
        ctrl = ttk.Frame(self, padding=(10, 8)); ctrl.pack(fill="x")
        ttk.Button(ctrl, text="Open CSV…", command=self._browse).pack(side="left")
        self.embed_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(ctrl, text="Embed plots in window", variable=self.embed_var).pack(side="left", padx=10)

        # Main frame for drag‑and‑drop and (optional) embedded charts
        self.main_frame = ttk.Frame(self, padding=15, relief="groove")
        self.main_frame.pack(fill="both", expand=True, padx=15, pady=(0, 10))

        prompt = "📂  Drop a Tekscan CSV here" if TkinterDnD else ""
        prompt += "\n…or click Open CSV"
        self.prompt_lbl = ttk.Label(self.main_frame, text=prompt, anchor="center", font=("Segoe UI", 11))
        self.prompt_lbl.pack(expand=True)

        if TkinterDnD:
            self.main_frame.drop_target_register(DND_FILES)
            self.main_frame.dnd_bind("<<Drop>>", self._on_drop)

        # Stats label
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
        plt.close("all")  # Close old external windows
        self.prompt_lbl.configure(text="Processing…"); self.update_idletasks()

        try:
            raw_np, total_force_csv = load_csv(path)
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to load CSV:\n{exc}")
            self.prompt_lbl.configure(text="Drop CSV here / Open CSV")
            self.stats_lbl.configure(text="No file loaded")
            return

        split_col, is_sep = find_split_column(raw_np)
        left_raw  = raw_np[:, :split_col] if split_col else np.empty((raw_np.shape[0], 0))
        right_raw = raw_np[:, split_col + 1:] if is_sep else raw_np[:, split_col:]

        vmap = np.vectorize(map_raw_to_pressure)
        left_kpa  = vmap(left_raw)  if left_raw.size  else None
        right_kpa = vmap(right_raw) if right_raw.size else None
        data = left_kpa if left_kpa is not None and np.count_nonzero(left_kpa) else right_kpa
        side = "Left" if data is left_kpa else "Right"

        labels, cluster_areas = label_clusters(data > 0)
        area_cm2 = [a * CELL_AREA_CM2 for a in cluster_areas]
        force_n  = [np.nansum(data[labels == i + 1] * CELL_AREA_N_PER_KPA) for i in range(len(cluster_areas))]
        row_force = calc_row_force_equal_area(data)

        figs = [
            make_heatmap_fig(data, labels, f"{side} – heat-map"),
            make_bar_fig(area_cm2, "Area (cm²)", "Toe contact area"),
            make_bar_fig(force_n, "Force (N)", "Toe force"),
            make_force_distribution_fig(row_force, "Row-wise force distribution (equal area)")
        ]

        if self.embed_var.get():
            self._embed_figures(figs)
        else:
            self._clear_embedded()
            for fig in figs:
                fig.show()
            plt.show(block=False)

        stats = (
            f"File: {os.path.basename(path)}\n"
            f"Side analysed: {side}\n"
            f"Contact area: {sum(area_cm2):.2f} cm²\n"
            f"Max pressure: {np.nanmax(data):.2f} kPa\n"
            f"Average pressure (contact): {np.nanmean(data[data > 0]):.2f} kPa\n"
            f"Total force (CSV): {total_force_csv:.1f} N  (~{total_force_csv / 9.81:.2f} kg)\n"
            f"Sum of toe forces: {sum(force_n):.1f} N"
        )
        self.stats_lbl.configure(text=stats)
        self.prompt_lbl.configure(text="Drop another CSV or Open CSV")

# ────────────────────────────────────────────────
#  Entrypoint
# ────────────────────────────────────────────────
if __name__ == "__main__":
    FootScanApp().mainloop()
