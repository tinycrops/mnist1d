import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors

# --- Configuration ---

# Use ARC colors for visualization
ARC_CMAP_LIST = [
    "#000000", "#0074D9", "#FF4136", "#2ECC40", "#FFDC00",
    "#AAAAAA", "#F012BE", "#FF851B", "#7FDBFF", "#870C25"
]
ARC_CMAP = mcolors.ListedColormap(ARC_CMAP_LIST)
NUM_COLORS = 10 # 0-9 for ARC

# Simulation Parameters
ALPHA = 0.25  # Learning rate / step size for the update (0 < alpha <= 1)
              # Smaller alpha = smaller steps, longer "memory"
              # Larger alpha = faster jumps towards current input

# --- Helper Functions ---

def l2norm(vec):
    """Normalizes a vector to unit length."""
    norm = np.linalg.norm(vec)
    if norm < 1e-10: # Avoid division by zero
        return vec
    return vec / norm

def get_target_direction(index, num_indices=NUM_COLORS):
    """Maps an integer index (e.g., color) to a point on the unit sphere."""
    # Simple mapping: distribute points based on index
    # Convert index to angles (phi, theta) in spherical coordinates
    # This is just one possible mapping, many others exist!
    
    # Phi: Angle from the positive z-axis (latitude)
    # Let's spread points between near north pole and near south pole
    phi = np.pi * (index + 1) / (num_indices + 1) 
    
    # Theta: Angle from the positive x-axis in the xy-plane (longitude)
    # Use golden angle increment for better distribution than simple linear spacing
    golden_angle = np.pi * (3. - np.sqrt(5.))
    theta = (index * golden_angle) % (2 * np.pi)

    # Convert spherical to Cartesian coordinates
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    
    return l2norm(np.array([x, y, z]))

def update_state_normalized_lerp(current_state, target_direction, alpha):
    """
    Updates the state using normalized Linear Interpolation (LERP).
    This approximates moving along the geodesic (shortest path on the sphere).
    It's conceptually similar to nGPT's update: h <- Norm(h + alpha * (delta - h))
    where delta is the target direction.
    """
    # Linear interpolation step
    unnormalized_next_state = current_state + alpha * (target_direction - current_state)
    
    # Normalize to project back onto the sphere surface
    next_state = l2norm(unnormalized_next_state)
    return next_state

# --- Simulation ---

# Example Input Sequence (e.g., flattened ARC grid colors, or just a sequence)
# Let's use the colors from the previous ARC grid example
arc_grid = np.array([
    [0, 1, 1, 0],
    [2, 5, 5, 3],
    [2, 5, 5, 3],
    [0, 4, 4, 0],
    [0, 0, 0, 0]
], dtype=int)
input_sequence = arc_grid.flatten() # Serialized sequence of colors

# Initial state (start at an arbitrary point on the sphere, e.g., x-axis)
initial_state = np.array([1.0, 0.0, 0.0])
current_state = initial_state.copy()

# Store the history of states
state_history = [current_state]
input_colors_history = [-1] # Store the input that *caused* the transition TO the state

print("Simulating spherical propagation...")
for i, input_index in enumerate(input_sequence):
    target_dir = get_target_direction(input_index, NUM_COLORS)
    current_state = update_state_normalized_lerp(current_state, target_dir, ALPHA)
    state_history.append(current_state)
    input_colors_history.append(input_index)
    # Optional: Print state norms to verify they stay close to 1
    # print(f"Step {i+1}, Input {input_index}: State Norm = {np.linalg.norm(current_state):.4f}")

state_history = np.array(state_history)

# --- Visualization ---
print("Plotting the path on the sphere...")
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Draw the sphere wireframe (optional, can be slow/cluttered)
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 50)
x_sph = np.outer(np.cos(u), np.sin(v))
y_sph = np.outer(np.sin(u), np.sin(v))
z_sph = np.outer(np.ones(np.size(u)), np.cos(v))
# ax.plot_surface(x_sph, y_sph, z_sph, color='grey', alpha=0.05, rstride=4, cstride=4, linewidth=0.1, edgecolors='k')
ax.plot_wireframe(x_sph, y_sph, z_sph, color='grey', alpha=0.1, rstride=10, cstride=10, linewidth=0.5)


# Plot the path of the state vector
# Color segments based on the *input* that caused the transition
for i in range(len(state_history) - 1):
    start_point = state_history[i]
    end_point = state_history[i+1]
    input_color_idx = input_colors_history[i+1] # Color causing move to end_point
    segment_color = ARC_CMAP_LIST[input_color_idx]
    ax.plot([start_point[0], end_point[0]], 
            [start_point[1], end_point[1]], 
            [start_point[2], end_point[2]], 
            color=segment_color, marker='.', markersize=3, linewidth=1.5, alpha=0.8)

# Mark start and end points
ax.scatter(state_history[0, 0], state_history[0, 1], state_history[0, 2], 
           color='lime', s=100, marker='o', label='Start State', depthshade=False)
ax.scatter(state_history[-1, 0], state_history[-1, 1], state_history[-1, 2], 
           color='red', s=100, marker='X', label='End State', depthshade=False)

# Plot target directions for reference (optional)
# for c in range(NUM_COLORS):
#     target = get_target_direction(c, NUM_COLORS)
#     ax.scatter(target[0], target[1], target[2], 
#                color=ARC_CMAP_LIST[c], s=50, marker='^', alpha=0.5, label=f'Target {c}')

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title(f"Path of State Vector on Sphere (alpha={ALPHA})")
ax.set_xlim([-1.1, 1.1])
ax.set_ylim([-1.1, 1.1])
ax.set_zlim([-1.1, 1.1])
# ax.legend() # Can get cluttered quickly
ax.set_aspect('equal') # Crucial for sphere visualization
plt.tight_layout()
plt.show()

print(f"\nFinal State Vector: {state_history[-1]}")
print(f"Norm of Final State: {np.linalg.norm(state_history[-1]):.4f}")
print("\nNote how the path stays on the sphere surface.")
print("The final position encodes a history of the sequential inputs.")