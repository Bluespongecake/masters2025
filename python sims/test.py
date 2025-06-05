import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math

import linkage_funcs

# Parameters
L_AB = 15
L_AC = 14.5
L_CD = 20.006
L_BD = 15.811
initial_angle = 90  # degrees
extension_length = 100

linkage = linkage_funcs.FourBarLinkage(L_AB, L_AC, L_CD, L_BD, initial_angle)
angles, A_pos, C_pos, motion_type = linkage_funcs.run_simulation(linkage)
# linkage_funcs.plot_paths(A_pos, C_pos, linkage,angles=angles, num_snapshots=1)

from matplotlib.animation import FuncAnimation
from IPython.display import HTML

def animate_linkage_motion(A_pos, C_pos, linkage, interval=100):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    all_x = np.concatenate([A_pos[:, 0], C_pos[:, 0], [linkage.D[0], linkage.B[0]]])
    all_y = np.concatenate([A_pos[:, 1], C_pos[:, 1], [linkage.D[1], linkage.B[1]]])
    margin = 0.5
    ax.set_xlim(np.min(all_x) - margin, np.max(all_x) + margin)
    ax.set_ylim(np.min(all_y) - margin, np.max(all_y) + margin)

    ax.plot(*linkage.D, 'ko', label='D (fixed)')
    ax.plot(*linkage.B, 'ko', label='B (fixed)')
    ax.plot([linkage.D[0], linkage.B[0]], [linkage.D[1], linkage.B[1]], 'k-', label='Link BD (fixed)')
    ax.plot(A_pos[:, 0], A_pos[:, 1], 'b--', alpha=0.3, label='Path of A')
    ax.plot(C_pos[:, 0], C_pos[:, 1], 'r--', alpha=0.3, label='Path of C')

    line_AB, = ax.plot([], [], 'b-', linewidth=2, label='Link AB')
    line_AC, = ax.plot([], [], 'g-', linewidth=2, label='Link AC')
    line_CD, = ax.plot([], [], 'm-', linewidth=2, label='Link CD')
    point_A, = ax.plot([], [], 'bo', markersize=6)
    point_C, = ax.plot([], [], 'ro', markersize=6)

    title = ax.set_title("Animated Four-Bar Linkage with Angles")
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))

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

        theta_AB = np.degrees(np.arctan2(A[1] - B[1], A[0] - B[0]))
        theta_CD = np.degrees(np.arctan2(C[1] - D[1], C[0] - D[0]))

        title.set_text(f"Angles — AB: {theta_AB:.1f}°, CD: {theta_CD:.1f}°")
        return line_AB, line_AC, line_CD, point_A, point_C, title

    anim = FuncAnimation(fig, update, frames=len(A_pos), interval=interval, blit=True)
    plt.show()  # This is the key line for showing the animation
    

animate_linkage_motion(A_pos, C_pos, linkage, 100)
