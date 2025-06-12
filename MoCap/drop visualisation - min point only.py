"""
Rigid-body trajectory visualiser (Y-up)

* 3-D track + orientation arrows
* 2-D side view (Y vs Z)
* Euler angles vs time
* **Tilt-to-vertical vs time**
* **Speed vs time** (new)

Extras
------
* Marks the first contact point (≥100 mm drop, lowest before rebound)
* Prints the **frame index** and **timestamp** of contact

Dependencies: numpy, pandas, matplotlib, scipy
--------------------------------------------------------------------
Edit the PARAMS block below to point at your CSV and (optionally) set a body-frame offset.
--------------------------------------------------------------------
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# ╔════════════════════════ PARAMS (edit here) ═════════════════════╗
CSV_PATH      = "initial test_001-sidebounce.csv"    # ← path to your file
HEADER_ROWS   = 5                                    # rows to skip before header
OFFSET_MM     = np.array([0, 0, 0])                  # body-frame offset (mm)
ARROW_STEP    = 10                                   # plot an orientation arrow every N samples
ARROW_LENGTH  = 25                                   # arrow length (mm)
# ╚════════════════════════════════════════════════════════════════╝

# ───────────────────────── Utility helpers ─────────────────────────

def match_axis_spans_3d(ax):
    """Force equal numeric scale on all three axes."""
    extents = np.array([getattr(ax, f"get_{dim}lim")() for dim in "xyz"])
    sizes = extents[:, 1] - extents[:, 0]
    centres = extents.mean(axis=1)
    half = sizes.max() / 2
    for dim, c in zip("xyz", centres):
        getattr(ax, f"set_{dim}lim")((c - half, c + half))


def body_offset_world(quat_array: np.ndarray, pos_array: np.ndarray, offset_vec_mm: np.ndarray) -> np.ndarray:
    rot = R.from_quat(quat_array)
    return pos_array + rot.apply(offset_vec_mm)


def tilt_angle_to_vertical(quat_array: np.ndarray) -> np.ndarray:
    body_y = np.array([0.0, 1.0, 0.0])
    vecs_world = R.from_quat(quat_array).apply(body_y)
    cosang = np.clip((vecs_world @ body_y), -1.0, 1.0)
    return np.degrees(np.arccos(cosang))


def plot_scalar_vs_time(time, scalar, ylabel, title, contact_idx=None):
    plt.figure(figsize=(10, 4))
    plt.plot(time, scalar, label=ylabel)
    if contact_idx is not None:
        plt.axvline(time[contact_idx], ls="--", lw=0.8, label="Contact")
    plt.xlabel("Time (s)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

# ─────────────────────────── Load data ─────────────────────────────
df = pd.read_csv(CSV_PATH, skiprows=HEADER_ROWS)

quat_df  = df.iloc[:, 2:6].apply(pd.to_numeric, errors="coerce")
pos_df   = df.iloc[:, 6:9].apply(pd.to_numeric, errors="coerce")
time_ser = pd.to_numeric(df.iloc[:, 1], errors="coerce")

quat, pos, time = quat_df.to_numpy(), pos_df.to_numpy(), time_ser.to_numpy()

valid = ~np.isnan(quat).any(1) & ~np.isnan(pos).any(1) & ~np.isnan(time)
quat, pos, time = quat[valid], pos[valid], time[valid]

# Offset track in world space
offset_world = body_offset_world(quat, pos, OFFSET_MM)

# ─────────────── Remap coords so **Y is up** in plots ──────────────
x_plot, y_plot, z_plot = pos[:, 0], pos[:, 2], pos[:, 1]
x_off_plot, y_off_plot, z_off_plot = offset_world[:, 0], offset_world[:, 2], offset_world[:, 1]

# ───────────── Identify point of contact ─────────────
initial_z = z_plot[0]
fall_mask = (initial_z - z_plot) >= 100.0  # ≥100 mm drop
contact_idx: int | None = None
if np.any(fall_mask):
    first_cross = int(np.argmax(fall_mask))
    rebound_idx = len(z_plot) - 1
    for j in range(first_cross, len(z_plot) - 1):
        if z_plot[j + 1] - z_plot[j] > 0:  # upward motion ⇒ bounce start
            rebound_idx = j + 1
            break
    contact_idx = first_cross + int(np.argmin(z_plot[first_cross:rebound_idx + 1]))

# Print contact frame & timestamp
if contact_idx is not None:
    print(f"Contact detected ➜ frame {contact_idx} at t = {time[contact_idx]:.6f} s")
else:
    print("No contact detected (object never dropped ≥100 mm or never rebounded).")

# ───────────── Compute velocity (mm/s) and speed ─────────────
# Use central differences via np.gradient for robustness to uneven dt
vel = np.gradient(pos, time, axis=0)  # shape (N,3)
speed = np.linalg.norm(vel, axis=1)   # scalar mm/s

# ───────────── Visualisation ───────────────────────────────────────
rot_world = R.from_quat(quat)
body_y = np.array([0.0, 1.0, 0.0])
vecs_world = rot_world.apply(body_y)
vx_plot, vy_plot, vz_plot = vecs_world.T[[0, 2, 1]]

# ––– 3-D trajectory –––
fig = plt.figure(figsize=(8, 6))
ax3d = fig.add_subplot(111, projection="3d")
ax3d.plot(x_plot, y_plot, z_plot, lw=2, label="Body origin")
ax3d.plot(x_off_plot, y_off_plot, z_off_plot, lw=1.5, color="tab:red", label="Offset point")
if contact_idx is not None:
    ax3d.scatter(x_plot[contact_idx], y_plot[contact_idx], z_plot[contact_idx], s=80, c="lime", marker="*", label="Contact")
for i in range(0, len(pos), ARROW_STEP):
    ax3d.quiver(x_plot[i], y_plot[i], z_plot[i], vx_plot[i], vy_plot[i], vz_plot[i], length=ARROW_LENGTH, normalize=True, arrow_length_ratio=0.2, lw=0.8, color="grey")
ax3d.set_xlabel("X")
ax3d.set_ylabel("Z")
ax3d.set_zlabel("Y (up)")
ax3d.set_title("Rigid-body trajectory (Y-up)")
ax3d.view_init(elev=25, azim=-60)
ax3d.legend()
match_axis_spans_3d(ax3d)
plt.tight_layout()
plt.show()

# ––– 2-D side view –––
fig2, ax2d = plt.subplots(figsize=(6, 6))
ax2d.plot(y_plot, z_plot, lw=1.5, label="Body origin")
ax2d.scatter(y_off_plot, z_off_plot, s=10, c="tab:red", alpha=0.7, label="Offset point")
if contact_idx is not None:
    ax2d.scatter(y_plot[contact_idx], z_plot[contact_idx], s=80, c="lime", marker="*", label="Contact")
ax2d.set_xlabel("Z")
ax2d.set_ylabel("Y (up)")
ax2d.set_title("Side view: Y vs Z")
ax2d.set_aspect("equal", adjustable="box")
ax2d.legend()
plt.tight_layout()
plt.show()

# ––– Euler angles –––
euler_yup = rot_world.as_euler("yxz", degrees=True)
labels = ["Yaw (°)", "Pitch (°)", "Roll (°)"]
for idx, lab in enumerate(labels):
    plot_scalar_vs_time(time, euler_yup[:, idx], lab, f"{lab} vs time", contact_idx)

# ––– Tilt-to-vertical –––
tilt_deg = tilt_angle_to_vertical(quat)
plot_scalar_vs_time(time, tilt_deg, "Tilt angle (°)", "Body tilt vs vertical", contact_idx)

# ––– Speed –––
plot_scalar_vs_time(time, speed, "Speed (mm/s)", "Translational speed vs time", contact_idx)
