edit this to generate a csv of the 2d chart: """
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
HEADER_ROWS   = 6                     # how many non-data rows to skip
ARROW_STEP    = 20                    # draw an orientation arrow every N rows
ARROW_LENGTH  = 50                    # length of those arrows (plot units)
OFFSET_MM     = np.array([-20, -21, -8.66])  # body-frame offset point, in mm
                                         #   (x, y, z) = +50 mm along body X
# ╚═════════════════════════════════════════════════════════════════╝


# ─────────────────────── helper: equal 3-D spans ───────────────────
def match_axis_spans_3d(ax):
    """Expand/contract all three 3-D axes so they share the *largest* span."""
    x_min, x_max = ax.get_xlim3d()
    y_min, y_max = ax.get_ylim3d()
    z_min, z_max = ax.get_zlim3d()

    spans = np.array([x_max - x_min, y_max - y_min, z_max - z_min])
    target = spans.max() / 2.0

    centers = np.array([
        0.5 * (x_min + x_max),
        0.5 * (y_min + y_max),
        0.5 * (z_min + z_max)
    ])

    ax.set_xlim3d(centers[0] - target, centers[0] + target)
    ax.set_ylim3d(centers[1] - target, centers[1] + target)
    ax.set_zlim3d(centers[2] - target, centers[2] + target)

    # Matplotlib ≥ 3.4 keeps the box cubic
    try:
        ax.set_box_aspect([1, 1, 1])
    except AttributeError:
        pass


# ─────────────── helper: body-frame offset → world track ───────────
def body_offset_world(quat_array, pos_array, offset_vec_mm):
    """
    Returns an (N×3) array giving the world-space coordinates of a point that
    sits at *offset_vec_mm* in the **body frame** for every sample.

    * quat_array: (N×4) Quaternions (x, y, z, w) world←body
    * pos_array : (N×3) Body origin in world coords
    * offset_vec_mm: (3,)  body-frame [x,y,z] offset in **same linear units**
                     as pos_array (mm assumed here)

    Example: a sensor 50 mm in +X of the body frame → OFFSET_MM=[50,0,0]
    """
    rot = R.from_quat(quat_array)
    offset_world = rot.apply(offset_vec_mm)
    return pos_array + offset_world


# ─────────────────────────── Load data ─────────────────────────────
df   = pd.read_csv(CSV_PATH, skiprows=HEADER_ROWS)

quat = df.iloc[:, 2:6].to_numpy()      # C–F → qx, qy, qz, qw
pos  = df.iloc[:, 6:9].to_numpy()      # G–I →  x,  y,  z
time = df.iloc[:, 1   ].to_numpy()     #    B →  time (s)

valid = ~np.isnan(quat).any(1) & ~np.isnan(pos).any(1)
quat, pos, time = quat[valid], pos[valid], time[valid]

# offset track in world space
offset_world = body_offset_world(quat, pos, OFFSET_MM)

# ─────────────── Remap coords so **Y is up** in plots ──────────────
x_plot, y_plot, z_plot           = pos[:, 0], pos[:, 2], pos[:, 1]
x_off_plot, y_off_plot, z_off_plot = (
    offset_world[:, 0], offset_world[:, 2], offset_world[:, 1]
)

rot_world = R.from_quat(quat)         # for orientation arrows
body_y     = np.array([0., 1., 0.])        # +Y in the body frame
vecs_world = rot_world.apply(body_y)       # rotate into world space

# Remap to plotting axes (world X→plot X, world Z→plot Y, world Y→plot Z)
vx_plot, vy_plot, vz_plot = vecs_world.T[[0, 2, 1]]

# ────────────────────── 3-D trajectory (Y-up) ──────────────────────
fig = plt.figure(figsize=(10, 8))
ax3d = fig.add_subplot(111, projection="3d")

ax3d.plot(x_plot, y_plot, z_plot, lw=2, label="Body origin")
ax3d.plot(x_off_plot, y_off_plot, z_off_plot,
          lw=1.5, color="tab:red", label="Offset point")

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

match_axis_spans_3d(ax3d)          # equal numeric scale
plt.tight_layout()
plt.show()

# ────────────────────── 2-D Y-Z side view ─────────────────────────
fig2, ax2d = plt.subplots(figsize=(6, 6))
ax2d.plot(y_plot, z_plot, lw=1.5, label="Body origin")
ax2d.scatter(y_off_plot, z_off_plot, s=10, c="tab:red",
             label="Offset point", alpha=0.7)

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
    plt.xlabel("Time (s)")
    plt.ylabel(lab)
    plt.title(f"{lab} vs time")
    plt.tight_layout()
    plt.show()