"""
Rigid-body trajectory visualiser (Y-up)

* 3-D Y-up track + orientation arrows
* 2-D side view (Y vs Z) with equal scale
* Roll / Pitch / Yaw plots
* Optional body-frame offset point (e.g. “sensor puck”, “tool-tip” …)

Dependencies: numpy, pandas, matplotlib, scipy
--------------------------------------------------------------------
Edit the PARAMS block to point at your CSV and define the offset.
--------------------------------------------------------------------
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# ╔════════════════════════ PARAMS (edit here) ═════════════════════╗
CSV_PATH      = "initial test_001-sidebounce.csv"    # ← path to your file
HEADER_ROWS   = 5                                    # rows to skip before header
OFFSET_MM     = np.array([0, 0, 0])                  # body-frame offset (mm)
ARROW_STEP    = 10                                   # plot an orientation arrow every N samples
ARROW_LENGTH  = 25                                   # length of orientation arrows (mm)
# ╚════════════════════════════════════════════════════════════════╝


def match_axis_spans_3d(ax):
    """Force a cubic bounding box so that XYZ have equal numeric scale."""
    extents = np.array([getattr(ax, f"get_{dim}lim")() for dim in "xyz"])
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz)) / 2
    new_extents = np.array([[c - maxsize, c + maxsize] for c in centers])
    for dim, lim in zip("xyz", new_extents):
        getattr(ax, f"set_{dim}lim")(lim)


def body_offset_world(quat_array: np.ndarray, pos_array: np.ndarray, offset_vec_mm: np.ndarray) -> np.ndarray:
    """Return the track of an offset point expressed in world coords.

    Args
    ----
    quat_array : (N×4) Quaternions (x, y, z, w) world←body
    pos_array  : (N×3) Body origin in world coords
    offset_vec_mm : (3,) Body-frame [x,y,z] offset in **same linear units** as pos_array (mm assumed)
    """
    rot = R.from_quat(quat_array)
    offset_world = rot.apply(offset_vec_mm)
    return pos_array + offset_world

# ─────────────────────────── Load data ─────────────────────────────
df = pd.read_csv(CSV_PATH, skiprows=HEADER_ROWS)

# Convert the numeric columns *explicitly* to floats so np.isnan works on all platforms —
# some CSVs arrive as object/str and break isnan with a TypeError.
quat_df  = df.iloc[:, 2:6].apply(pd.to_numeric, errors="coerce")  # C–F → qx, qy, qz, qw
pos_df   = df.iloc[:, 6:9].apply(pd.to_numeric, errors="coerce")  # G–I →  x,  y,  z
time_ser = pd.to_numeric(df.iloc[:, 1], errors="coerce")         #    B → time (s)

quat, pos, time = quat_df.to_numpy(), pos_df.to_numpy(), time_ser.to_numpy()

valid = ~np.isnan(quat).any(1) & ~np.isnan(pos).any(1) & ~np.isnan(time)
quat, pos, time = quat[valid], pos[valid], time[valid]

# Offset track in world space
offset_world = body_offset_world(quat, pos, OFFSET_MM)

# ─────────────── Remap coords so **Y is up** in plots ──────────────
x_plot, y_plot, z_plot = pos[:, 0], pos[:, 2], pos[:, 1]
x_off_plot, y_off_plot, z_off_plot = offset_world[:, 0], offset_world[:, 2], offset_world[:, 1]

# ───────────── Identify point of contact ─────────────
# Criteria:
#   1. Body has fallen >100 mm from its initial height (z_plot axis is “Y-up”)
#   2. The contact point is the minimum z before the first upward rebound.

initial_z = z_plot[0]
fall_mask = (initial_z - z_plot) >= 100.0  # 100 mm threshold

contact_idx: int | None = None
if np.any(fall_mask):
    first_cross = int(np.argmax(fall_mask))  # first index below threshold

    # forward-scan until vertical velocity becomes positive (bounce)
    rebound_idx = len(z_plot) - 1  # default: never rebounds
    for j in range(first_cross, len(z_plot) - 1):
        if z_plot[j + 1] - z_plot[j] > 0:
            rebound_idx = j + 1
            break

    # pick the *lowest* sample between first_cross and rebound_idx inclusive
    rel_min = np.argmin(z_plot[first_cross: rebound_idx + 1])
    contact_idx = first_cross + int(rel_min)

# ─────────────── Orientation body-Y arrows in world space ───────────────
rot_world = R.from_quat(quat)
body_y = np.array([0.0, 1.0, 0.0])  # +Y in the body frame
vecs_world = rot_world.apply(body_y)

# Map to plotting axes (world X→plot X, world Z→plot Y, world Y→plot Z)
vx_plot, vy_plot, vz_plot = vecs_world.T[[0, 2, 1]]

# ────────────────────── 3-D trajectory (Y-up) ──────────────────────
fig = plt.figure(figsize=(8, 6))
ax3d = fig.add_subplot(111, projection="3d")

ax3d.plot(x_plot, y_plot, z_plot, lw=2, label="Body origin")
ax3d.plot(x_off_plot, y_off_plot, z_off_plot, lw=1.5, color="tab:red", label="Offset point")

# Mark contact in 3-D view
if contact_idx is not None:
    ax3d.scatter(x_plot[contact_idx], y_plot[contact_idx], z_plot[contact_idx],
                 s=80, c="lime", marker="*", label="Contact")

for i in range(0, len(pos), ARROW_STEP):
    ax3d.quiver(x_plot[i], y_plot[i], z_plot[i],
                vx_plot[i], vy_plot[i], vz_plot[i],
                length=ARROW_LENGTH, normalize=True,
                arrow_length_ratio=0.2, lw=0.8, color="grey")

ax3d.set_xlabel("X")
ax3d.set_ylabel("Z")
ax3d.set_zlabel("Y (up)")
ax3d.set_title("Rigid-body trajectory (Y-up)")
ax3d.view_init(elev=25, azim=-60)
ax3d.legend()
match_axis_spans_3d(ax3d)  # equal numeric scale
plt.tight_layout()
plt.show()

# ────────────────────── 2-D Y-Z side view ─────────────────────────
fig2, ax2d = plt.subplots(figsize=(6, 6))
ax2d.plot(y_plot, z_plot, lw=1.5, label="Body origin")
ax2d.scatter(y_off_plot, z_off_plot, s=10, c="tab:red", label="Offset point", alpha=0.7)

# Mark contact in 2-D view
if contact_idx is not None:
    ax2d.scatter(y_plot[contact_idx], z_plot[contact_idx], s=80, c="lime", marker="*", label="Contact")

ax2d.set_xlabel("Z")
ax2d.set_ylabel("Y (up)")
ax2d.set_title("Side view: Y vs Z (equal scale)")
ax2d.set_aspect("equal", adjustable="box")
ax2d.legend()
plt.tight_layout()
plt.show()

# ──────────────────── Euler angles vs. time ───────────────────────
euler_yup = rot_world.as_euler("yxz", degrees=True)
labels = ["Yaw (about Y, °)", "Pitch (about X, °)", "Roll (about Z, °)"]

for idx, lab in enumerate(labels):
    plt.figure(figsize=(10, 4))
    plt.plot(time, euler_yup[:, idx])
    if contact_idx is not None:
        plt.axvline(time[contact_idx], linestyle="--", linewidth=0.8, label="Contact")
    plt.xlabel("Time (s)")
    plt.ylabel(lab)
    plt.title(f"{lab} vs time")
    plt.legend()
    plt.tight_layout()
    plt.show()