import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math

class FourBarLinkage:
    def __init__(self, L_AB, L_BC, L_CD, L_DA, initial_angle_CD=0):
        """
        Initialize four-bar linkage with given link lengths.
        
        Parameters:
        L_AB: Length of link AB (output link)
        L_BC: Length of link BC (coupler link) 
        L_CD: Length of link CD (input crank)
        L_DA: Length of link DA (fixed link/ground)
        initial_angle_CD: Initial angle of CD from horizontal (degrees)
        
        Configuration: D is at origin, A is at (L_DA, 0)
        """
        self.L_AB = L_AB
        self.L_BC = L_BC  # This is AC in the problem description
        self.L_CD = L_CD
        self.L_DA = L_DA  # This is BD in the problem description (fixed link)
        self.initial_angle_CD = math.radians(initial_angle_CD)
        
        # Check if the linkage configuration is valid (Grashof condition)
        self._validate_linkage()
        
        # Fixed points
        self.D = np.array([0, 0])
        self.B = np.array([L_DA, 0])  # B is fixed at distance L_DA from D
        
    def _validate_linkage(self):
        """Check if the linkage configuration is geometrically possible"""
        lengths = [self.L_AB, self.L_BC, self.L_CD, self.L_DA]
        lengths.sort()
        
        # For a four-bar linkage to exist, the sum of the shortest three links
        # must be greater than the longest link
        if lengths[0] + lengths[1] + lengths[2] <= lengths[3]:
            raise ValueError("Invalid linkage: Links cannot form a closed mechanism")
        
        # Additional check: each triangle in the linkage must be valid
        # Triangle BCD: check if C can be reached from both B and D
        min_BC_dist = abs(self.L_CD - self.L_DA)
        max_BC_dist = self.L_CD + self.L_DA
        
        if self.L_BC < min_BC_dist or self.L_BC > max_BC_dist:
            raise ValueError(f"Invalid linkage: Link BC length {self.L_BC} is incompatible with CD={self.L_CD} and DA={self.L_DA}")
        
        # Triangle ABC: Similar check
        min_AC_dist = abs(self.L_AB - self.L_DA)
        max_AC_dist = self.L_AB + self.L_DA
        
        if self.L_BC < min_AC_dist or self.L_BC > max_AC_dist:
            raise ValueError(f"Invalid linkage: Link AC length {self.L_BC} is incompatible with AB={self.L_AB} and DA={self.L_DA}")
    
    def solve_position(self, theta_CD):
        """
        Solve for positions of points A and C given angle of CD
        
        Parameters:
        theta_CD: Angle of CD from horizontal (radians)
        
        Returns:
        A, C: Position vectors of points A and C
        """
        # Position of C (rotating about D)
        C = self.D + self.L_CD * np.array([np.cos(theta_CD), np.sin(theta_CD)])
        
        # Solve for position of A using intersection of two circles:
        # Circle 1: centered at B with radius L_AB
        # Circle 2: centered at C with radius L_BC
        
        # Distance between B and C
        BC_dist = np.linalg.norm(C - self.B)
        
        # Check if solution exists
        if BC_dist > (self.L_AB + self.L_BC) or BC_dist < abs(self.L_AB - self.L_BC):
            raise ValueError(f"No solution exists for theta_CD = {math.degrees(theta_CD):.2f}°")
        
        # Find intersection points
        # Vector from B to C
        BC_vec = C - self.B
        BC_unit = BC_vec / BC_dist
        
        # Distance from B to the line connecting the two intersection points
        a = (self.L_AB**2 - self.L_BC**2 + BC_dist**2) / (2 * BC_dist)
        
        # Half-distance between intersection points
        h = np.sqrt(self.L_AB**2 - a**2)
        
        # Point on line BC closest to intersection points
        P = self.B + a * BC_unit
        
        # Perpendicular vector
        perp = np.array([-BC_unit[1], BC_unit[0]])
        
        # Two possible positions for A
        A1 = P + h * perp
        A2 = P - h * perp
        
        # Choose the solution based on initial configuration
        # For this problem, we'll choose the solution that makes AC more vertical initially
        if theta_CD == self.initial_angle_CD:
            # For initial position, choose A such that AC is closer to vertical
            AC1_angle = math.atan2((C[1] - A1[1]), (C[0] - A1[0]))
            AC2_angle = math.atan2((C[1] - A2[1]), (C[0] - A2[0]))
            
            # Choose the one closer to vertical (π/2 or -π/2)
            if abs(abs(AC1_angle) - math.pi/2) < abs(abs(AC2_angle) - math.pi/2):
                self.preferred_branch = 1
                A = A1
            else:
                self.preferred_branch = 2
                A = A2
        else:
            # Use the same branch as established initially
            if hasattr(self, 'preferred_branch'):
                A = A1 if self.preferred_branch == 1 else A2
            else:
                A = A1  # Default choice
        
        return A, C
    
    def find_motion_limits(self, resolution=720):
        """
        Find the angular limits of the input crank by testing all angles
        
        Parameters:
        resolution: Number of angle steps to test (higher = more accurate)
        
        Returns:
        valid_angles: List of angles where mechanism can reach
        """
        test_angles = np.linspace(0, 2*np.pi, resolution)
        valid_angles = []
        
        for angle in test_angles:
            try:
                self.solve_position(angle + self.initial_angle_CD)
                valid_angles.append(angle)
            except ValueError:
                continue
        
        return np.array(valid_angles)
    
    def simulate(self, num_steps=360):
        """
        Simulate the linkage motion. If full rotation isn't possible,
        oscillate between the reachable limits.
        
        Parameters:
        num_steps: Number of steps in the simulation
        
        Returns:
        angles: Array of input angles used
        A_positions: Array of A positions
        C_positions: Array of C positions
        motion_type: 'continuous' or 'oscillating'
        """
        print("Finding motion limits...")
        valid_angles = self.find_motion_limits()
        
        if len(valid_angles) == 0:
            raise ValueError("No valid positions found for this linkage configuration")
        
        # Check if we have continuous motion (full rotation possible)
        angle_gaps = np.diff(np.sort(valid_angles))
        max_gap = np.max(angle_gaps) if len(angle_gaps) > 0 else 0
        
        # If there's a large gap, we don't have continuous rotation
        has_continuous_motion = max_gap < np.pi/6  # 30 degree threshold
        
        A_positions = []
        C_positions = []
        
        if has_continuous_motion:
            print("Mechanism can rotate continuously")
            # Use full rotation
            angles = np.linspace(0, 2*np.pi, num_steps)
            motion_type = 'continuous'
            
            for angle in angles:
                try:
                    A, C = self.solve_position(angle + self.initial_angle_CD)
                    A_positions.append(A)
                    C_positions.append(C)
                except ValueError:
                    # This shouldn't happen for continuous motion, but handle gracefully
                    if len(A_positions) > 0:
                        A_positions.append(A_positions[-1])
                        C_positions.append(C_positions[-1])
                    else:
                        A_positions.append(np.array([0, 0]))
                        C_positions.append(np.array([0, 0]))
        else:
            print("Mechanism has limited motion - will oscillate")
            # Find the continuous range of motion
            valid_angles_sorted = np.sort(valid_angles)
            
            # Find the largest continuous segment
            segments = []
            current_segment = [valid_angles_sorted[0]]
            
            for i in range(1, len(valid_angles_sorted)):
                if valid_angles_sorted[i] - valid_angles_sorted[i-1] < np.pi/6:  # Small gap
                    current_segment.append(valid_angles_sorted[i])
                else:
                    segments.append(current_segment)
                    current_segment = [valid_angles_sorted[i]]
            segments.append(current_segment)
            
            # Use the largest segment
            largest_segment = max(segments, key=len)
            min_angle = min(largest_segment)
            max_angle = max(largest_segment)
            
            print(f"Motion range: {math.degrees(min_angle):.1f}° to {math.degrees(max_angle):.1f}°")
            
            # Create oscillating motion
            half_steps = num_steps // 2
            forward_angles = np.linspace(min_angle, max_angle, half_steps)
            backward_angles = np.linspace(max_angle, min_angle, half_steps)
            angles = np.concatenate([forward_angles, backward_angles])
            motion_type = 'oscillating'
            
            for angle in angles:
                A, C = self.solve_position(angle + self.initial_angle_CD)
                A_positions.append(A)
                C_positions.append(C)
        
        return angles, np.array(A_positions), np.array(C_positions), motion_type
    
    def plot_mechanism(self, A_positions, C_positions, motion_type='continuous', save_animation=False):
        """
        Create an animated plot of the four-bar linkage
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Set up the plot
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        title = 'Four-Bar Linkage Animation\nAB=Output, AC=Coupler, CD=Input, BD=Fixed'
        if motion_type == 'oscillating':
            title += ' (Oscillating Motion)'
        else:
            title += ' (Continuous Rotation)'
        ax.set_title(title)
        
        # Plot fixed points
        ax.plot(*self.D, 'ko', markersize=8, label='D (fixed)')
        ax.plot(*self.B, 'ko', markersize=8, label='B (fixed)')
        
        # Plot fixed link BD
        ax.plot([self.D[0], self.B[0]], [self.D[1], self.B[1]], 'k-', linewidth=3, label='BD (fixed)')
        
        # Plot trajectory paths
        ax.plot(A_positions[:, 0], A_positions[:, 1], 'b--', alpha=0.5, label='Path of A')
        ax.plot(C_positions[:, 0], C_positions[:, 1], 'r--', alpha=0.5, label='Path of C')
        
        # Plot initial position of AC
        initial_A = A_positions[0]
        initial_C = C_positions[0]
        ax.plot([initial_A[0], initial_C[0]], [initial_A[1], initial_C[1]], 'g-', 
                linewidth=2, alpha=0.7, label='Initial AC position')
        
        # For oscillating motion, highlight the extreme positions
        if motion_type == 'oscillating':
            # Find the extreme positions (first and middle of the array)
            mid_idx = len(A_positions) // 2
            extreme_A1, extreme_C1 = A_positions[0], C_positions[0]
            extreme_A2, extreme_C2 = A_positions[mid_idx], C_positions[mid_idx]
            
            ax.plot([extreme_A1[0], extreme_C1[0]], [extreme_A1[1], extreme_C1[1]], 'orange', 
                    linewidth=2, alpha=0.8, label='Limit position 1')
            ax.plot([extreme_A2[0], extreme_C2[0]], [extreme_A2[1], extreme_C2[1]], 'purple', 
                    linewidth=2, alpha=0.8, label='Limit position 2')
        
        # Initialize moving elements
        line_AB, = ax.plot([], [], 'b-', linewidth=3, label='AB (output)')
        line_AC, = ax.plot([], [], 'r-', linewidth=3, label='AC (coupler)')
        line_CD, = ax.plot([], [], 'm-', linewidth=3, label='CD (input)')
        point_A, = ax.plot([], [], 'bo', markersize=8)
        point_C, = ax.plot([], [], 'ro', markersize=8)
        
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        def animate(frame):
            if frame < len(A_positions):
                A = A_positions[frame]
                C = C_positions[frame]
                
                # Update lines
                line_AB.set_data([self.B[0], A[0]], [self.B[1], A[1]])
                line_AC.set_data([A[0], C[0]], [A[1], C[1]])
                line_CD.set_data([self.D[0], C[0]], [self.D[1], C[1]])
                
                # Update points
                point_A.set_data([A[0]], [A[1]])
                point_C.set_data([C[0]], [C[1]])
            
            return line_AB, line_AC, line_CD, point_A, point_C
        
        # Set axis limits based on the range of motion
        all_x = np.concatenate([A_positions[:, 0], C_positions[:, 0], [self.D[0], self.B[0]]])
        all_y = np.concatenate([A_positions[:, 1], C_positions[:, 1], [self.D[1], self.B[1]]])
        
        margin = 0.5
        ax.set_xlim(np.min(all_x) - margin, np.max(all_x) + margin)
        ax.set_ylim(np.min(all_y) - margin, np.max(all_y) + margin)
        
        # Adjust animation speed based on motion type
        interval = 100 if motion_type == 'oscillating' else 50
        
        # Create animation
        anim = FuncAnimation(fig, animate, frames=len(A_positions), 
                           interval=interval, blit=True, repeat=True)
        
        plt.tight_layout()
        
        if save_animation:
            filename = f'fourbar_linkage_{motion_type}.gif'
            anim.save(filename, writer='pillow', fps=10 if motion_type == 'oscillating' else 20)
            print(f"Animation saved as '{filename}'")
        
        plt.show()
        
        return anim

def main():
    """Main function to run the four-bar linkage simulation"""
    print("Four-Bar Linkage Simulator")
    print("=" * 40)
    
    try:
        # Get user input
        print("Enter the link lengths:")
        L_AB = float(input("Length of AB (output link): "))
        L_AC = float(input("Length of AC (coupler link, starts vertical): "))
        L_CD = float(input("Length of CD (input crank): "))
        L_BD = float(input("Length of BD (fixed link): "))
        
        print("\nEnter initial angle:")
        initial_angle = float(input("Initial angle of CD from horizontal (degrees): "))
        
        # Create linkage (note: we use AC as BC in the class)
        linkage = FourBarLinkage(L_AB, L_AC, L_CD, L_BD, initial_angle)
        
        # Simulate the mechanism
        print("\nSimulating mechanism...")
        angles, A_pos, C_pos = linkage.simulate(num_steps=180)
        
        # Display some statistics
        print(f"\nSimulation complete!")
        print(f"Range of A: X[{np.min(A_pos[:, 0]):.2f}, {np.max(A_pos[:, 0]):.2f}], Y[{np.min(A_pos[:, 1]):.2f}, {np.max(A_pos[:, 1]):.2f}]")
        print(f"Range of C: X[{np.min(C_pos[:, 0]):.2f}, {np.max(C_pos[:, 0]):.2f}], Y[{np.min(C_pos[:, 1]):.2f}, {np.max(C_pos[:, 1]):.2f}]")
        
        # Create animation
        save_gif = input("\nSave animation as GIF? (y/n): ").lower().startswith('y')
        anim = linkage.plot_mechanism(A_pos, C_pos, save_animation=save_gif)
        
    except ValueError as e:
        print(f"\nError: {e}")
        print("Please check your input values and try again.")
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")
    except Exception as e:
        print(f"\nUnexpected error: {e}")

if __name__ == "__main__":
    main()