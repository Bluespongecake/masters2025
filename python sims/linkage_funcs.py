import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

try:
    from IPython.display import HTML
except ImportError:
    HTML = None


class FourBarLinkage:
    """
    Represents a planar four-bar linkage with fixed ground points B and D.

    Link lengths:
        L_AB: length between joints A and B
        L_AC: length between joints A and C
        L_CD: length between joints C and D (driving crank)
        L_BD: length between joints B and D (fixed base)

    initial_angle_CD: initial absolute angle (in degrees) of crank CD relative to the x-axis.
    """

    def __init__(self, L_AB, L_AC, L_CD, L_BD, initial_angle_CD=0):
        self.L_AB = L_AB
        self.L_AC = L_AC
        self.L_CD = L_CD
        self.L_BD = L_BD
        self.initial_angle_CD = math.radians(initial_angle_CD)

        # Fixed ground points: D at origin, B on x-axis
        self.D = np.array([0.0, 0.0])
        self.B = np.array([L_BD, 0.0])

        self._validate_linkage()

    def _validate_linkage(self):
        """Validate that four links can form a closed mechanism (generalized triangle inequality)."""
        lengths = [self.L_AB, self.L_AC, self.L_CD, self.L_BD]
        lengths.sort()
        if lengths[0] + lengths[1] + lengths[2] <= lengths[3]:
            raise ValueError("Invalid linkage: links cannot form a closed mechanism")

    def solve_position(self, theta_CD, A_prev=None):
        """
        Solve for positions of joints A and C given the absolute crank angle theta_CD (radians).

        If A_prev is provided (2-element array), choose the branch (of the two possible A solutions) closest
        to A_prev to ensure continuity. Otherwise, select the branch with higher y-coordinate by default.

        Returns:
            A: np.array([x, y]) for joint A
            C: np.array([x, y]) for joint C

        Note:
            Uses dot-product and arccos to compute interior angles elsewhere; uses atan2 for orientation in animation.
        """
        # Compute C from crank angle
        C = self.D + self.L_CD * np.array([math.cos(theta_CD), math.sin(theta_CD)])

        # Distance between C and B
        dist_CB = np.linalg.norm(C - self.B)
        if dist_CB > (self.L_AB + self.L_AC) or dist_CB < abs(self.L_AB - self.L_AC):
            raise ValueError(f"No solution exists for θ_CD = {math.degrees(theta_CD):.2f}°")

        # Unit vector from B to C
        CB_unit = (C - self.B) / dist_CB
        # a = distance from B along BC_unit to midpoint P of intersection chord
        a = (self.L_AB**2 - self.L_AC**2 + dist_CB**2) / (2 * dist_CB)
        h_sq = self.L_AB**2 - a**2
        h = math.sqrt(max(h_sq, 0.0))

        P = self.B + a * CB_unit
        perp = np.array([-CB_unit[1], CB_unit[0]])

        A1 = P + h * perp
        A2 = P - h * perp

        if A_prev is not None:
            # Choose branch closest to previous A for smooth motion tracking
            dist1 = np.linalg.norm(A1 - A_prev)
            dist2 = np.linalg.norm(A2 - A_prev)
            A = A1 if dist1 < dist2 else A2
        else:
            # Default: pick the higher y-coordinate to match initial configuration
            A = A1 if A1[1] > A2[1] else A2

        return A, C

    def visualize_linkage(self, theta_CD=None):
        """
        Display a static plot of the linkage at absolute crank angle theta_CD (radians).

        Note:
            theta_CD here is treated as an absolute angle about D, not relative to initial_angle_CD.
            Inconsistency with run_simulation: run_simulation passes a relative angle + initial_angle_CD.
            Users should supply the same convention if calling visualize_linkage directly.

        Draws links AB, AC, CD, BD, joint circles, length labels, and interior joint angles (via dot-product/arccos).
        """
        if theta_CD is None:
            theta_CD = self.initial_angle_CD

        A, C = self.solve_position(theta_CD)
        B = self.B
        D = self.D

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_aspect('equal')
        ax.grid(True)

        # Draw links
        ax.plot([A[0], B[0]], [A[1], B[1]], 'b-', label='AB')
        ax.plot([A[0], C[0]], [A[1], C[1]], 'g-', label='AC')
        ax.plot([C[0], D[0]], [C[1], D[1]], 'm-', label='CD')
        ax.plot([D[0], B[0]], [D[1], B[1]], 'k-', label='BD')

        # Draw joints
        ax.plot(*A, 'bo')
        ax.plot(*B, 'ko')
        ax.plot(*C, 'ro')
        ax.plot(*D, 'ko')

        # Label joints
        ax.text(A[0], A[1], ' A', fontsize=12, verticalalignment='bottom', color='blue')
        ax.text(B[0], B[1], ' B', fontsize=12, verticalalignment='bottom', color='black')
        ax.text(C[0], C[1], ' C', fontsize=12, verticalalignment='bottom', color='red')
        ax.text(D[0], D[1], ' D', fontsize=12, verticalalignment='bottom', color='black')

        # Label lengths at link midpoints
        def label_length(p1, p2, label):
            mid = (p1 + p2) / 2
            ax.text(mid[0], mid[1], f"{label:.2f}", fontsize=10, color='gray', ha='center', va='center')

        label_length(A, B, self.L_AB)
        label_length(A, C, self.L_AC)
        label_length(C, D, self.L_CD)
        label_length(B, D, self.L_BD)

        # Compute interior joint angles using dot-product and arccos
        def interior_angle(v1, v2):
            dot = np.dot(v1, v2)
            norm_prod = np.linalg.norm(v1) * np.linalg.norm(v2)
            return math.degrees(math.acos(np.clip(dot / norm_prod, -1.0, 1.0)))

        angle_A = interior_angle(B - A, C - A)
        angle_B = interior_angle(A - B, D - B)
        angle_C = interior_angle(A - C, D - C)
        angle_D = interior_angle(C - D, B - D)

        ax.text(A[0], A[1] - 0.3, f"{angle_A:.1f}°", color='blue', ha='center')
        ax.text(B[0], B[1] - 0.3, f"{angle_B:.1f}°", color='black', ha='center')
        ax.text(C[0], C[1] + 0.3, f"{angle_C:.1f}°", color='red', ha='center')
        ax.text(D[0], D[1] - 0.3, f"{angle_D:.1f}°", color='black', ha='center')

        ax.legend()
        ax.set_title("Four-Bar Linkage Configuration")
        plt.tight_layout()
        plt.show()


def find_motion_limits(linkage, resolution=720):
    """
    Determine all relative crank angles (radians) at which the linkage can form a valid position.

    Returns a 1D numpy array of relative angles from 0 to 2π (exclusive) such that
    solve_position(initial_angle_CD + angle) is valid.
    """
    angles = np.linspace(0, 2 * np.pi, resolution)
    valid = []
    for a in angles:
        try:
            linkage.solve_position(a + linkage.initial_angle_CD)
            valid.append(a)
        except ValueError:
            continue
    return np.array(valid)


def run_simulation(linkage, num_steps=360):
    """
    Run a full simulation sweep of the four-bar linkage.

    Determines if the crank is continuous or oscillating and generates trajectories accordingly.

    Returns:
        angles: 1D array of relative crank angles (radians) used in the simulation
        A_pos: Nx2 array of joint A positions over time
        C_pos: Nx2 array of joint C positions over time
        mode: 'continuous' or 'oscillating'
    """
    valid_angles = find_motion_limits(linkage)
    if valid_angles.size == 0:
        raise ValueError("No valid crank angles: linkage cannot move in any orientation.")

    sorted_valid = np.sort(valid_angles)
    # Grashof-like test: check for large gaps in valid-angle set
    angle_gaps = np.diff(sorted_valid)
    max_gap = np.max(angle_gaps) if angle_gaps.size > 0 else 0
    # If the largest invalid region is small, treat as continuous rotation
    if max_gap < (np.pi / 6):
        mode = 'continuous'
    else:
        mode = 'oscillating'

    A_pos = []
    C_pos = []

    if mode == 'continuous':
        angles = np.linspace(0, 2 * np.pi, num_steps)
    else:
        # Identify the largest contiguous segment of valid angles
        segments = []
        seg = [sorted_valid[0]]
        for i in range(1, sorted_valid.size):
            if sorted_valid[i] - sorted_valid[i - 1] < (np.pi / 6):
                seg.append(sorted_valid[i])
            else:
                segments.append(seg)
                seg = [sorted_valid[i]]
        segments.append(seg)
        largest_segment = max(segments, key=len)
        min_a, max_a = min(largest_segment), max(largest_segment)
        fwd = np.linspace(min_a, max_a, num_steps // 2)
        bwd = np.linspace(max_a, min_a, num_steps // 2)
        angles = np.concatenate([fwd, bwd])

    A_prev = None
    C_prev = None
    for a in angles:
        theta = a + linkage.initial_angle_CD
        try:
            A, C = linkage.solve_position(theta, A_prev=A_prev)
        except ValueError:
            # If invalid, hold previous position (should be rare if mode classification is correct)
            if A_prev is not None:
                A, C = A_prev, C_prev
            else:
                raise RuntimeError("Failed to compute first valid position in run_simulation.")
        A_pos.append(A)
        C_pos.append(C)
        A_prev, C_prev = A, C

    return angles, np.array(A_pos), np.array(C_pos), mode


def plot_paths(A_pos, C_pos, linkage, angles=None, num_snapshots=5):
    """
    Plot the trajectories of joints A and C, overlay snapshots of the linkage, annotate geometric minima/maxima,
    and plot the four interior joint angles versus the driving crank angle.

    If 'angles' is provided, determines whether the motion is continuous or oscillating by checking for duplicate angles.

    Parameters:
        A_pos: Nx2 array of positions of joint A over time
        C_pos: Nx2 array of positions of joint C over time
        linkage: FourBarLinkage instance
        angles: 1D array of relative crank angles (radians)
        num_snapshots: number of static linkage snapshots to display along the trajectory
    """
    if angles is not None:
        # Determine continuous vs oscillating by checking angle uniqueness
        unique_angles = np.unique(angles)
        if unique_angles.size == angles.size:
            # Continuous rotation: plot full trajectories
            A_traj = A_pos
            C_traj = C_pos
            theta_rel = angles
        else:
            # Oscillating: first half of arrays are unique, second half retraces
            half = len(A_pos) // 2
            A_traj = A_pos[:half]
            C_traj = C_pos[:half]
            theta_rel = angles[:half]
        theta_abs = theta_rel + linkage.initial_angle_CD
    else:
        # No angle data: plot full arrays by default
        A_traj = A_pos
        C_traj = C_pos
        theta_rel = None
        theta_abs = None

    B = linkage.B
    D = linkage.D
    num_points = A_traj.shape[0]

    # Compute interior joint angles for each step via dot-product/arccos
    angle_A = np.zeros(num_points)
    angle_B = np.zeros(num_points)
    angle_C = np.zeros(num_points)
    angle_D = np.zeros(num_points)

    for i in range(num_points):
        A = A_traj[i]
        C = C_traj[i]
        # Vectors for interior angles
        AB_vec = B - A
        AC_vec = C - A
        BA_vec = A - B
        BD_vec = D - B
        CB_vec = B - C
        CD_vec = D - C
        DC_vec = C - D
        DB_vec = B - D

        def interior_angle(v1, v2):
            dot = np.dot(v1, v2)
            norm_prod = np.linalg.norm(v1) * np.linalg.norm(v2)
            cos_val = np.clip(dot / norm_prod, -1.0, 1.0)
            return math.acos(cos_val)

        angle_A[i] = interior_angle(AB_vec, AC_vec)
        angle_B[i] = interior_angle(BA_vec, BD_vec)
        angle_C[i] = interior_angle(CB_vec, CD_vec)
        angle_D[i] = interior_angle(DC_vec, DB_vec)

    # Convert to degrees
    angle_A_deg = np.degrees(angle_A)
    angle_B_deg = np.degrees(angle_B)
    angle_C_deg = np.degrees(angle_C)
    angle_D_deg = np.degrees(angle_D)

    fig, (ax_spatial, ax_angles) = plt.subplots(
        2, 1, figsize=(10, 10), gridspec_kw={"height_ratios": [2, 1]}
    )

    # ======== TOP: Spatial trajectories + snapshots ========
    ax_spatial.plot(A_traj[:, 0], A_traj[:, 1], "b--", alpha=0.5, label="Path of A")
    ax_spatial.plot(C_traj[:, 0], C_traj[:, 1], "r--", alpha=0.5, label="Path of C")
    ax_spatial.plot(*D, "ko", label="D (fixed)")
    ax_spatial.plot(*B, "ko", label="B (fixed)")
    ax_spatial.plot([D[0], B[0]], [D[1], B[1]], "k-", label="Link BD (fixed)")

    # Snapshot indices evenly spaced over trajectory
    indices = np.linspace(0, num_points - 1, num_snapshots, dtype=int)
    for idx in indices:
        A_snap = A_traj[idx]
        C_snap = C_traj[idx]
        ax_spatial.plot([B[0], A_snap[0]], [B[1], A_snap[1]], "b-")      # AB
        ax_spatial.plot([A_snap[0], C_snap[0]], [A_snap[1], C_snap[1]], "g-")  # AC
        ax_spatial.plot([D[0], C_snap[0]], [D[1], C_snap[1]], "m-")      # CD
        ax_spatial.plot(*A_snap, "bo", markersize=4)
        ax_spatial.plot(*C_snap, "ro", markersize=4)

    if angles is not None:
        # Annotate min/max of CD angle positions
        min_cd_angle = np.min(theta_abs)
        max_cd_angle = np.max(theta_abs)
        C_min = D + linkage.L_CD * np.array([math.cos(min_cd_angle), math.sin(min_cd_angle)])
        C_max = D + linkage.L_CD * np.array([math.cos(max_cd_angle), math.sin(max_cd_angle)])
        ax_spatial.plot(*C_min, "bo")
        ax_spatial.annotate(
            f"Min CD: {math.degrees(min_cd_angle):.1f}°",
            xy=C_min, xytext=(C_min[0] + 0.5, C_min[1]),
            arrowprops=dict(arrowstyle="->", color="blue"),
            fontsize=10, color="blue"
        )
        ax_spatial.plot(*C_max, "go")
        ax_spatial.annotate(
            f"Max CD: {math.degrees(max_cd_angle):.1f}°",
            xy=C_max, xytext=(C_max[0] + 0.5, C_max[1]),
            arrowprops=dict(arrowstyle="->", color="green"),
            fontsize=10, color="green"
        )

        # Annotate min/max of AB (using interior angle at B)
        idx_min_B = np.argmin(angle_B_deg)
        idx_max_B = np.argmax(angle_B_deg)
        A_min = A_traj[idx_min_B]
        A_max = A_traj[idx_max_B]
        ax_spatial.plot(*A_min, "bo")
        ax_spatial.annotate(
            f"Min AB: {angle_B_deg[idx_min_B]:.1f}°",
            xy=A_min, xytext=(A_min[0] + 0.5, A_min[1]),
            arrowprops=dict(arrowstyle="->", color="navy"),
            fontsize=10, color="navy"
        )
        ax_spatial.plot(*A_max, "go")
        ax_spatial.annotate(
            f"Max AB: {angle_B_deg[idx_max_B]:.1f}°",
            xy=A_max, xytext=(A_max[0] + 0.5, A_max[1]),
            arrowprops=dict(arrowstyle="->", color="darkgreen"),
            fontsize=10, color="darkgreen"
        )

    ax_spatial.set_aspect("equal")
    ax_spatial.set_title("Linkage Paths with Geometric Min/Max Labels")
    ax_spatial.grid(True)
    ax_spatial.legend(loc="upper left")

    # ======== BOTTOM: Four Joint Angles vs. Driving Crank Angle ========
    if angles is not None:
        drive_deg = np.degrees(theta_rel)
        ax_angles.plot(drive_deg, angle_A_deg, label="Joint A")
        ax_angles.plot(drive_deg, angle_B_deg, label="Joint B")
        ax_angles.plot(drive_deg, angle_C_deg, label="Joint C")
        ax_angles.plot(drive_deg, angle_D_deg, label="Joint D")

        ax_angles.set_xlabel("Driving Crank Angle (°)")
        ax_angles.set_ylabel("Interior Joint Angle (°)")
        ax_angles.set_title("Joint Angles A, B, C, D vs. Driving Crank")
        ax_angles.grid(True)
        ax_angles.legend(loc="upper right")
    else:
        ax_angles.text(0.5, 0.5, "No angle data provided", ha="center", va="center")
        ax_angles.set_xticks([])
        ax_angles.set_yticks([])

    plt.tight_layout()
    plt.show()


def animate_linkage_motion(A_pos, C_pos, linkage, interval=100, save_path=None):
    """
    Animate the four-bar linkage motion.

    Displays the fixed joints B and D, the path of A and C, and updates moving links AB, AC, CD.
    The animation title shows link orientations via atan2 (signed angles).

    Parameters:
        A_pos: Nx2 array of joint A positions
        C_pos: Nx2 array of joint C positions
        linkage: FourBarLinkage instance
        interval: delay between frames in milliseconds
        save_path: optional file path to save the animation (requires ffmpeg)

    Returns:
        HTML(anim.to_jshtml()) if IPython is available, else the FuncAnimation object.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Determine plot limits with a margin proportional to mechanism size
    all_x = np.concatenate([A_pos[:, 0], C_pos[:, 0], [linkage.D[0], linkage.B[0]]])
    all_y = np.concatenate([A_pos[:, 1], C_pos[:, 1], [linkage.D[1], linkage.B[1]]])
    span_x = np.max(all_x) - np.min(all_x)
    span_y = np.max(all_y) - np.min(all_y)
    margin_x = 0.05 * span_x
    margin_y = 0.05 * span_y
    ax.set_xlim(np.min(all_x) - margin_x, np.max(all_x) + margin_x)
    ax.set_ylim(np.min(all_y) - margin_y, np.max(all_y) + margin_y)

    # Plot fixed points and base link BD
    ax.plot(*linkage.D, 'ko', label='D (fixed)')
    ax.plot(*linkage.B, 'ko', label='B (fixed)')
    ax.plot([linkage.D[0], linkage.B[0]], [linkage.D[1], linkage.B[1]], 'k-', label='Link BD (fixed)')

    # Plot full path traces
    ax.plot(A_pos[:, 0], A_pos[:, 1], 'b--', alpha=0.3, label='Path of A')
    ax.plot(C_pos[:, 0], C_pos[:, 1], 'r--', alpha=0.3, label='Path of C')

    # Initialize line and point artists
    line_AB, = ax.plot([], [], 'b-', linewidth=2, label='Link AB')
    line_AC, = ax.plot([], [], 'g-', linewidth=2, label='Link AC')
    line_CD, = ax.plot([], [], 'm-', linewidth=2, label='Link CD')
    point_A, = ax.plot([], [], 'bo', markersize=6)
    point_C, = ax.plot([], [], 'ro', markersize=6)

    title = ax.set_title("Animated Four-Bar Linkage with Angles")
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    plt.tight_layout()

    def update(frame):
        A = A_pos[frame]
        C = C_pos[frame]
        B = linkage.B
        D = linkage.D

        line_AB.set_data([B[0], A[0]], [B[1], A[1]])
        line_AC.set_data([A[0], C[0]], [A[1], C[1]])
        line_CD.set_data([D[0], C[0]], [D[1], C[1]])
        point_A.set_data([A[0]], [A[1]])
        point_C.set_data([C[0]], [C[1]])

        # Compute signed orientations using atan2
        theta_AB = math.degrees(math.atan2(A[1] - B[1], A[0] - B[0]))
        theta_CD = math.degrees(math.atan2(C[1] - D[1], C[0] - D[0]))
        title.set_text(f"Angles — AB: {theta_AB:.1f}°, CD: {theta_CD:.1f}°")

        return line_AB, line_AC, line_CD, point_A, point_C, title

    anim = FuncAnimation(fig, update, frames=len(A_pos), interval=interval, blit=True)

    if save_path:
        try:
            anim.save(save_path, writer='ffmpeg', fps=1000 // interval)
            print(f"Animation saved to {save_path}")
        except Exception as e:
            print(f"Failed to save animation: {e}")

    plt.close(fig)
    if HTML:
        return HTML(anim.to_jshtml())
    else:
        return anim
