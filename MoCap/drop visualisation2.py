import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3-D)
import numpy as np
import pandas as pd


###############################################################################
# 1.  Helper functions
###############################################################################
def find_header_row(csv_path: Path) -> int:
    """
    In a Motive export the real column names start on the line whose first token
    is literally 'Frame'.  Return that zero-based line index so we can skip
    everything that comes before it.
    """
    with csv_path.open("r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if line.startswith("Frame"):
                return i
    raise ValueError("Couldn’t find the header row (a line that starts 'Frame').")


def q_to_matrix(w: float, x: float, y: float, z: float) -> np.ndarray:
    """
    Convert a unit quaternion to a 3 × 3 rotation matrix.
    """
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n == 0:
        return np.eye(3)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


###############################################################################
# 2.  Load the CSV
###############################################################################
csv_file = Path("initial test.csv")          # <- change if needed
header_idx = find_header_row(csv_file)

# The first rigid body lives in the first 9 columns after “Time (Seconds)”
cols = ["frame", "time", "qx", "qy", "qz", "qw", "px", "py", "pz"]
df = pd.read_csv(
    csv_file,
    header=None,
    skiprows=header_idx + 1,            # jump straight to numeric data
    usecols=range(len(cols)),           # only the columns we care about
    names=cols,
    engine="python",
)
# Down-cast everything to floats and drop rows where pose data is missing
df = df.apply(pd.to_numeric, errors="coerce").dropna(subset=["px", "py", "pz", "qw"])

###############################################################################
# 3.  Convert quaternion → body-X axis in world coords (for arrow plot)
###############################################################################
# World-space direction of the rigid body’s local x-axis
ori = np.vstack(
    [
        q_to_matrix(w, x, y, z) @ np.array([1.0, 0.0, 0.0])
        for w, x, y, z in df[["qw", "qx", "qy", "qz"]].itertuples(index=False)
    ]
)

###############################################################################
# 4.  Make the 3-D plot
###############################################################################
fig = plt.figure()
ax = fig.add_subplot(projection="3d")

# Trajectory
ax.plot(df["px"], df["py"], df["pz"])

# Orientation arrows (sampled so the view isn’t cluttered)
step = max(len(df) // 100, 1)            # ≈100 arrows max
ax.quiver(
    df["px"][::step],
    df["py"][::step],
    df["pz"][::step],
    ori[::step, 0],
    ori[::step, 1],
    ori[::step, 2],
    length=20.0,                         # ← tweak for your scale
    linewidth=0.5,
)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Rigid-Body Position & Orientation")
plt.tight_layout()
plt.show()