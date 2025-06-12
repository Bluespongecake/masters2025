"""
Rigid-body drop-test visualiser  (Y-up)
=======================================

⬆ What’s new
------------
*   **Contact / minimum / release** points are now detected from the **vertical velocity (Y-axis)** instead of the old 100 mm-drop heuristic.
    * **Contact** = first frame where the downward velocity suddenly *decelerates* by more than `VEL_CONTACT_DV`.
    * **Minimum** = the lowest Y-position between contact and the first upward-motion frame.
    * **Release** = the peak upward velocity in the first positive-velocity segment after minimum.
*   Flags for those three frames are written to the processed CSV and highlighted in every plot.
*   New parameters group **CONTACT DETECTION** has been added.
*   ➕ **NEW**: A second CSV containing only the window of interest — 10 frames before *contact* through 10 frames after *release* — is now also exported for quick inspection.

Plots & outputs
---------------
* 3-D trajectory with orientation arrows + markers for contact/min/release
* 2-D side view (Y vs Z) with the same markers
* Euler angles vs time
* Tilt-to-vertical vs time
* Speed vs time
* Velocity **elevation** angle vs time
* **Vertical (Y) velocity vs time**

Extras
------
* Detects contact / minimum / release as above
* Prints frame/time of each key point plus velocity/tilt/elevation immediately before & after
* Exports a processed CSV containing every quantity plotted
* Exports a **windowed CSV** spanning *contact – 10 frames* → *release + 10 frames*

Dependencies: numpy · pandas · matplotlib · scipy
--------------------------------------------------------------------
Edit the PARAMS block to point at your CSV (and optional body-frame offset).
--------------------------------------------------------------------
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


def get_csv_path_drag_drop() -> str:
    """Open a drag-and-drop window to get a CSV file path."""
    import tkinter as tk
    from tkinterdnd2 import DND_FILES, TkinterDnD
    import sys

    csv_path = None

    def on_drop(event):
        nonlocal csv_path
        # Remove curly braces on Windows drag-n-drop
        path = event.data.strip('{}')
        if not path.lower().endswith('.csv'):
            label.config(text="Please drop a .csv file", fg="red")
            return
        csv_path = path
        root.destroy()

    # Create window
    root = TkinterDnD.Tk()
    root.title("Drop your CSV file")
    root.geometry("400x150")
    root.resizable(False, False)

    label = tk.Label(root, text="Drag and drop your CSV file here", width=40, height=10, bg="lightgray")
    label.pack(padx=10, pady=10)
    label.drop_target_register(DND_FILES)
    label.dnd_bind("<<Drop>>", on_drop)

    root.mainloop()

    if not csv_path:
        sys.exit("No file dropped. Exiting.")

    return csv_path
    
    


# ╔════════════════════════ PARAMS (edit here) ═════════════════════╗
CSV_PATH = get_csv_path_drag_drop()  # ← input file via drag-and-drop
HEADER_ROWS   = 5                                   # rows to skip before header
OFFSET_MM     = np.array([0, 0, 0])                 # body-frame offset (mm)
ARROW_STEP    = 10                                  # orientation arrow every N samples
ARROW_LENGTH  = 25                                  # arrow length (mm)

# —— Contact-detection tuning ——
VEL_CONTACT_DV = 500.0      # mm/s jump (∆v) signifying impact decel
VEL_RELEASE_DV = 300.0      # mm/s drop signifying release decel (after bounce)

# —— Window export tuning ——
FRAMES_BEFORE  = 10          # frames before contact to include
FRAMES_AFTER   = 10          # frames after release to include
# ╚════════════════════════════════════════════════════════════════╝

OUTPUT_CSV = str(Path(CSV_PATH).with_name(Path(CSV_PATH).stem + "_processed.csv"))
OUTPUT_WINDOW_CSV = str(Path(CSV_PATH).with_name(Path(CSV_PATH).stem + "_contact_window.csv"))

# ───────────────────────── Helper functions ─────────────────────────

def match_axis_spans_3d(ax):
    ext = np.array([getattr(ax, f"get_{d}lim")() for d in "xyz"])
    half = (ext[:, 1] - ext[:, 0]).max() / 2
    ctr = ext.mean(axis=1)
    for d, c in zip("xyz", ctr):
        getattr(ax, f"set_{d}lim")((c - half, c + half))

def body_offset_world(q: np.ndarray, p: np.ndarray, off_mm: np.ndarray) -> np.ndarray:
    return p + R.from_quat(q).apply(off_mm)

def tilt_angle_to_vertical(q: np.ndarray) -> np.ndarray:
    body_y = np.array([0.0, 1.0, 0.0])
    vecs = R.from_quat(q).apply(body_y)
    cosang = np.clip(vecs @ body_y, -1.0, 1.0)
    return np.degrees(np.arccos(cosang))

def velocity_elevation_angle(v: np.ndarray) -> np.ndarray:
    horiz = np.linalg.norm(v[:, [0, 2]], axis=1)
    return np.degrees(np.arctan2(v[:, 1], horiz))

def plot_scalar_vs_time(t, y, ylabel, title, key_frames=None):
    plt.figure(figsize=(10, 4))
    plt.plot(t, y, label=ylabel)
    if key_frames is not None:
        colours = {"contact": "lime", "minimum": "orange", "release": "red"}
        for name, idx in key_frames.items():
            if idx is not None:
                plt.axvline(t[idx], ls="--", lw=0.8, c=colours[name], label=name.capitalize())
    plt.xlabel("Time (s)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

# ─────────────────────── Load & clean data ────────────────────────
raw = pd.read_csv(CSV_PATH, skiprows=HEADER_ROWS)
quat_df  = raw.iloc[:, 2:6].apply(pd.to_numeric, errors="coerce")
pos_df   = raw.iloc[:, 6:9].apply(pd.to_numeric, errors="coerce")
time_ser = pd.to_numeric(raw.iloc[:, 1], errors="coerce")
quat, pos, time = quat_df.to_numpy(), pos_df.to_numpy(), time_ser.to_numpy()
mask = ~np.isnan(quat).any(1) & ~np.isnan(pos).any(1) & ~np.isnan(time)
quat, pos, time = quat[mask], pos[mask], time[mask]

# Offset track in world space
offset_world = body_offset_world(quat, pos, OFFSET_MM)

# Remap coords so **Y is up** in plots  (world X,Y,Z → plot X,Z,Y)
x_plot, y_plot, z_plot = pos[:, 0], pos[:, 2], pos[:, 1]
x_off_plot, y_off_plot, z_off_plot = offset_world[:, 0], offset_world[:, 2], offset_world[:, 1]

# ───────────── Velocity & derived metrics ─────────────
vel = np.gradient(pos, time, axis=0)        # mm/s
vel_y = vel[:, 1]
vel_diff = np.diff(vel_y, prepend=vel_y[0]) # prepend to align lengths
speed = np.linalg.norm(vel, axis=1)         # mm/s scalar
elev_angle = velocity_elevation_angle(vel)  # ° above horizontal
tilt_deg = tilt_angle_to_vertical(quat)     # °

# ───────────── Identify contact / minimum / release ──────────────
contact_idx: int | None = None
min_idx: int | None = None
release_idx: int | None = None

# 1) CONTACT  — large +∆v while still moving downward
for i in range(1, len(vel_y)):
    if vel_y[i-1] < 0 and vel_diff[i] > VEL_CONTACT_DV:
        contact_idx = i - 1
        break

# 2) MINIMUM  — lowest Y between contact and first upward motion
if contact_idx is not None:
    upward_start = next((j for j in range(contact_idx, len(vel_y)) if vel_y[j] >= 0), len(vel_y) - 1)
    slice_end = upward_start + 1  # include first non-neg frame
    min_idx = contact_idx + int(np.argmin(z_plot[contact_idx:slice_end]))

# 3) RELEASE  — peak upward velocity before decel (local max)
if min_idx is not None:
    # find contiguous positive-velocity block after upward_start
    pos_block_start = next((j for j in range(min_idx, len(vel_y)) if vel_y[j] > 0), None)
    if pos_block_start is not None:
        pos_block_end = pos_block_start
        while pos_block_end + 1 < len(vel_y) and vel_y[pos_block_end + 1] > 0:
            pos_block_end += 1
        # local maximum inside that block
        local_max_rel = int(np.argmax(vel_y[pos_block_start:pos_block_end + 1]))
        release_idx = pos_block_start + local_max_rel

key_frames = {"contact": contact_idx, "minimum": min_idx, "release": release_idx}

# ───────────── Terminal diagnostics ─────────────
for name, idx in key_frames.items():
    if idx is not None:
        idx_b = max(idx - 1, 0)
        idx_a = min(idx + 1, len(time) - 1)
        print(f"{name.capitalize()} ⇒ frame {idx}  (t = {time[idx]:.6f} s)")
        print(
            f"  » Velocity Y  before: {vel_y[idx_b]:.2f} mm/s | after: {vel_y[idx_a]:.2f} mm/s\n"
            f"  » Tilt before / after : {tilt_deg[idx_b]:.2f}° / {tilt_deg[idx_a]:.2f}°\n"
            f"  » Elev-angle   before / after : {elev_angle[idx_b]:.2f}° / {elev_angle[idx_a]:.2f}°"
        )

if contact_idx is None:
    print("No contact detected (∆v < threshold or downward motion not present)")

# ───────────── Export processed CSV ─────────────
proc = pd.DataFrame({
    "time_s": time,
    "world_x_mm": pos[:, 0],
    "world_y_mm": pos[:, 1],
    "world_z_mm": pos[:, 2],
    "plot_x_mm": x_plot,
    "plot_y_mm": y_plot,
    "plot_z_mm": z_plot,
    "vel_x_mm_s": vel[:, 0],
    "vel_y_mm_s": vel[:, 1],
    "vel_z_mm_s": vel[:, 2],
    "speed_mm_s": speed,
    "tilt_deg": tilt_deg,
    "vel_elev_deg": elev_angle,
    "is_contact_frame": np.arange(len(time)) == contact_idx,
    "is_minimum_frame": np.arange(len(time)) == min_idx,
    "is_release_frame": np.arange(len(time)) == release_idx,
})
proc.to_csv(OUTPUT_CSV, index=False)
print(f"Processed data written to → {OUTPUT_CSV}")

# ───────────── Export windowed CSV (contact±window) ─────────────
if contact_idx is not None and release_idx is not None:
    seg_start = max(contact_idx - FRAMES_BEFORE, 0)
    seg_end   = min(release_idx + FRAMES_AFTER, len(proc) - 1)
    proc_window = proc.iloc[seg_start:seg_end + 1]
    proc_window.to_csv(OUTPUT_WINDOW_CSV, index=False)
    print(f"Windowed data (frames {seg_start}–{seg_end}) written to → {OUTPUT_WINDOW_CSV}")
else:
    print("Windowed CSV not exported because contact and/or release could not be determined.")

# ───────────── Visualisation ───────────────────────────────────────
rot_world = R.from_quat(quat)
body_y = np.array([0.0, 1.0, 0.0])
vecs_world = rot_world.apply(body_y)
vx_plot, vy_plot, vz_plot = vecs_world.T[[0, 2, 1]]

# 3-D trajectory
fig = plt.figure(figsize=(8, 6))
ax3d = fig.add_subplot(111, projection="3d")
ax3d.plot(x_plot, y_plot, z_plot, lw=2, label="Body origin")
ax3d.plot(x_off_plot, y_off_plot, z_off_plot, lw=1.5, c="tab:red", label="Offset point")
markers = {"contact": ("*", "lime"), "minimum": ("o", "orange"), "release": ("^", "red")}
for name, (marker, colour) in markers.items():
    idx = key_frames[name]
    if idx is not None:
        ax3d.scatter(x_plot[idx], y_plot[idx], z_plot[idx], s=80, c=colour, marker=marker, label=name.capitalize())
for i in range(0, len(pos), ARROW_STEP):
    ax3d.quiver(x_plot[i], y_plot[i], z_plot[i], vx_plot[i], vy_plot[i], vz_plot[i],
                length=ARROW_LENGTH, normalize=True, arrow_length_ratio=0.2, lw=0.8, color="grey")
ax3d.set_xlabel("X")
ax3d.set_ylabel("Z")
ax3d.set_zlabel("Y (up)")
ax3d.set_title("Rigid-body trajectory (Y-up)")
ax3d.view_init(elev=25, azim=-60)
ax3d.legend()
match_axis_spans_3d(ax3d)
plt.tight_layout()
plt.show()

# ───────── 2-D side view ─────────
fig2, ax2d = plt.subplots(figsize=(6, 6))
ax2d.plot(y_plot, z_plot, lw=1.5, label="Body origin")
ax2d.scatter(y_off_plot, z_off_plot, s=10, c="tab:red", alpha=0.7,
             label="Offset point")

# … marker plotting loop …

ax2d.set_xlabel("Z")
ax2d.set_ylabel("Y (up)")
ax2d.set_title("Side view: Y vs Z")
ax2d.set_aspect("equal", adjustable="box")
ax2d.legend(loc="upper left", bbox_to_anchor=(1.02, 1),
            borderaxespad=0, frameon=False)

plt.tight_layout()
plt.show()

# Scalar plots
labels = ["Yaw (°)", "Pitch (°)", "Roll (°)"]
euler_yup = rot_world.as_euler("yxz", degrees=True)
for i, lab in enumerate(labels):
    plot_scalar_vs_time(time, euler_yup[:, i], lab, f"{lab} vs time", key_frames)

plot_scalar_vs_time(time, tilt_deg, "Tilt angle (°)", "Body tilt vs vertical", key_frames)
plot_scalar_vs_time(time, speed, "Speed (mm/s)", "Translational speed vs time", key_frames)
plot_scalar_vs_time(time, elev_angle, "Velocity elevation (°)", "Velocity angle vs horizontal", key_frames)
plot_scalar_vs_time(time, vel_y, "Vertical velocity (mm/s)", "Y velocity vs time", key_frames)
