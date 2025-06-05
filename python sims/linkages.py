import numpy as np
import matplotlib.pyplot as plt

    
# ---------------- CONFIGURABLE PARAMETERS ----------------
L1 = 10  # Ground link (A to D)
L2 = 7.5  # Input crank (A to B)
L3 = 2.5  # Coupler (B to C)
L4 = 10  # Output link (C to D)

n_links = 10 # number of sample links

# We'll calculate the starting angle to make coupler horizontal
angular_range_deg = 90  # Total angular range to sweep
num_steps = 360         # Number of steps for theta2

# ---------------- FIXED GEOMETRY ----------------
pointA = np.array([0, 0])
pointD = np.array([L1, 0])

# ---------------- FIND INITIAL ANGLE FOR HORIZONTAL COUPLER ----------------
def find_horizontal_coupler_angle():
    """Find the input angle theta2 that makes the coupler link horizontal"""
    # For horizontal coupler, points B and C have the same y-coordinate
    # We need to solve for theta2 such that By = Cy
    
    # Try different angles to find where coupler is horizontal
    test_angles = np.linspace(0, 2*np.pi, 1000)
    
    for theta2 in test_angles:
        B = pointA + L2 * np.array([np.cos(theta2), np.sin(theta2)])
        result = circle_intersection(B, L3, pointD, L4)
        if result is not None:
            C = result[0]  # choose "elbow-up" configuration
            # Check if coupler is approximately horizontal
            if abs(B[1] - C[1]) < 1e-6:
                return theta2
    
    # If exact horizontal not found, find the closest
    min_diff = float('inf')
    best_angle = 0
    
    for theta2 in test_angles:
        B = pointA + L2 * np.array([np.cos(theta2), np.sin(theta2)])
        result = circle_intersection(B, L3, pointD, L4)
        if result is not None:
            C = result[0]
            y_diff = abs(B[1] - C[1])
            if y_diff < min_diff:
                min_diff = y_diff
                best_angle = theta2
    
    return best_angle

# ---------------- GEOMETRY SOLVER ----------------
def circle_intersection(p1, r1, p2, r2):
    d = np.linalg.norm(p2 - p1)
    if d > r1 + r2 or d < abs(r1 - r2) or d == 0:
        return None
    a = (r1**2 - r2**2 + d**2) / (2 * d)
    h = np.sqrt(r1**2 - a**2)
    p3 = p1 + a * (p2 - p1) / d
    offset = h * np.array([-(p2[1] - p1[1]) / d, (p2[0] - p1[0]) / d])
    return p3 + offset, p3 - offset

def solve_linkage(theta2):
    B = pointA + L2 * np.array([np.cos(theta2), np.sin(theta2)])
    result = circle_intersection(B, L3, pointD, L4)
    if result is None:
        return B, None
    C = result[0]  # choose "elbow-up" configuration
    return B, C

# Find the starting angle for horizontal coupler
theta2_start = find_horizontal_coupler_angle()
theta2_start_deg = np.degrees(theta2_start)

print(f"Starting angle for horizontal coupler: {theta2_start_deg:.1f}°")

# Define the angle range centered around the horizontal position
theta2_min_deg = theta2_start_deg - angular_range_deg/2
theta2_max_deg = theta2_start_deg + angular_range_deg/2

# Convert angle range to radians
theta2_vals = np.radians(np.linspace(theta2_min_deg, theta2_max_deg, num_steps))

# ---------------- COMPUTE LINKAGE POSITIONS ----------------
pointsB = []
pointsC = []

for theta2 in theta2_vals:
    B, C = solve_linkage(theta2)
    if C is not None:
        pointsB.append(B)
        pointsC.append(C)
    else:
        pointsB.append([np.nan, np.nan])
        pointsC.append([np.nan, np.nan])

pointsB = np.array(pointsB)
pointsC = np.array(pointsC)

# ---------------- PLOT 1: PATH & LINKAGE CONFIGURATIONS ----------------
plt.figure(figsize=(10, 6))
plt.plot(pointsB[:, 0], pointsB[:, 1], label='Path of B')
plt.plot(pointsC[:, 0], pointsC[:, 1], label='Path of C')

sample_indices = np.linspace(0, len(theta2_vals) - 1, n_links, dtype=int)
for idx in sample_indices:
    A = pointA
    D = pointD
    B = pointsB[idx]
    C = pointsC[idx]
    if np.any(np.isnan(B)) or np.any(np.isnan(C)):
        continue
    plt.plot([A[0], B[0]], [A[1], B[1]], 'k-')
    plt.plot([B[0], C[0]], [B[1], C[1]], 'k-')
    plt.plot([C[0], D[0]], [C[1], D[1]], 'k-')
    plt.plot([D[0], A[0]], [D[1], A[1]], 'k--', alpha=0.3)

# Highlight the initial horizontal coupler position
initial_B, initial_C = solve_linkage(theta2_start)
if initial_C is not None:
    plt.plot([initial_B[0], initial_C[0]], [initial_B[1], initial_C[1]], 'r-', linewidth=3, alpha=0.7, label='Initial Horizontal Coupler')
    plt.plot(initial_B[0], initial_B[1], 'ro', markersize=8)
    plt.plot(initial_C[0], initial_C[1], 'ro', markersize=8)

plt.plot(pointA[0], pointA[1], 'ko', label='Point A')
plt.plot(pointD[0], pointD[1], 'ko', label='Point D')
plt.title("Four-Bar Linkage Path and Configurations (Starting with Horizontal Coupler)")
plt.axis('equal')
plt.grid(True)
plt.legend()
plt.xlabel("X Position")
plt.ylabel("Y Position")

# ---------------- PLOT 2: X POSITIONS vs THETA2 + Difference ----------------
plt.figure(figsize=(10, 4))
plt.plot(np.degrees(theta2_vals), pointsB[:, 0], label='Bx')
plt.plot(np.degrees(theta2_vals), pointsC[:, 0], label='Cx')
plt.plot(np.degrees(theta2_vals), pointsC[:, 0] - pointsB[:, 0], label='Cx - Bx', linestyle='--')
plt.axvline(theta2_start_deg, color='red', linestyle=':', alpha=0.7, label='Horizontal Coupler')
plt.title("X Position of B and C vs. Input Angle θ₂")
plt.xlabel("Input Angle θ₂ (degrees)")
plt.ylabel("X Position / Difference")
plt.grid(True)
plt.legend()

# ---------------- PLOT 3: Y POSITIONS vs THETA2 ----------------
plt.figure(figsize=(10, 4))
plt.plot(np.degrees(theta2_vals), pointsB[:, 1], label='By')
plt.plot(np.degrees(theta2_vals), pointsC[:, 1], label='Cy')
plt.axvline(theta2_start_deg, color='red', linestyle=':', alpha=0.7, label='Horizontal Coupler')
plt.title("Y Position of B and C vs. Input Angle θ₂")
plt.xlabel("Input Angle θ₂ (degrees)")
plt.ylabel("Y Position")
plt.grid(True)
plt.legend()

# ---------------- PLOT 4: Y POSITIONS vs THETA2 WITH NORMALIZATION ----------------

# Convert to degrees
theta2_deg = np.degrees(theta2_vals)

# Y positions
By = pointsB[:, 1]
Cy = pointsC[:, 1]
delta_y = By - Cy

# Mask valid data
valid_mask = ~np.isnan(delta_y)
valid_indices = np.where(valid_mask)[0]
valid_theta_indices = valid_indices[theta2_deg[valid_indices] <= theta2_max_deg]

# Only proceed if there is at least one valid value ≤ theta2_max_deg
if len(valid_theta_indices) > 0:
    reference_idx = valid_theta_indices[-1]
    delta_y_ref = delta_y[reference_idx]
    normalized_delta_y = delta_y - delta_y_ref

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(theta2_deg, By, label='By', color='blue')
    plt.plot(theta2_deg, Cy, label='Cy', color='green')
    plt.plot(theta2_deg, normalized_delta_y, label='(By - Cy) - ref', color='red', linestyle='--')
    plt.axvline(theta2_start_deg, color='red', linestyle=':', alpha=0.7, label='Horizontal Coupler')

    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.title("Y Positions and Normalized Vertical Separation of B and C")
    plt.xlabel("Input Angle θ₂ (degrees)")
    plt.ylabel("Y Position / Relative Difference")
    plt.grid(True)
    plt.legend()
else:
    print("Warning: No valid reference point for vertical normalization.")
    
plt.tight_layout()
plt.show()