import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D # For potential 3D visualization

# Define ARC colors (consistent with common ARC visualizations)
ARC_CMAP_LIST = [
    "#000000",  # 0: black
    "#0074D9",  # 1: blue
    "#FF4136",  # 2: red
    "#2ECC40",  # 3: green
    "#FFDC00",  # 4: yellow
    "#AAAAAA",  # 5: grey
    "#F012BE",  # 6: magenta
    "#FF851B",  # 7: orange
    "#7FDBFF",  # 8: light blue
    "#870C25",  # 9: dark red
]
ARC_CMAP = mcolors.ListedColormap(ARC_CMAP_LIST)
ARC_NORM = mcolors.Normalize(vmin=0, vmax=9)

def plot_arc_grid(ax, grid, title="ARC Grid"):
    """Helper function to plot an ARC grid."""
    ax.imshow(grid, cmap=ARC_CMAP, norm=ARC_NORM, interpolation='nearest')
    ax.set_xticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid.shape[0], 1), minor=True)
    ax.grid(which='minor', color='grey', linestyle='-', linewidth=0.5)
    ax.tick_params(which='minor', size=0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)

def arc_to_flattened_cylinder(grid):
    """
    Transforms a 2D ARC grid into the 10xN "flattened cylinder" representation.

    Args:
        grid (np.ndarray): A 2D numpy array representing the ARC task grid (H x W).
                           Values should be integers from 0 to 9.

    Returns:
        np.ndarray: A 2D numpy array of shape (10, N) where N = H * W.
                    flattened_cylinder[c, i] = 1 if the color at serialized
                    position i is c, and 0 otherwise.
    """
    if grid.ndim != 2:
        raise ValueError("Input grid must be 2-dimensional.")

    h, w = grid.shape
    n = h * w

    # Serialize the grid (row-major/C order is standard)
    serialized_grid = grid.flatten()

    # Ensure colors are within the expected range
    if not np.all((serialized_grid >= 0) & (serialized_grid <= 9)):
        print("Warning: Grid contains colors outside the 0-9 range.")
        # Option: Clip, raise error, or handle as needed
        # serialized_grid = np.clip(serialized_grid, 0, 9)

    # Create the 10xN output matrix (initialized to zeros)
    flattened_cylinder = np.zeros((10, n), dtype=int)

    # Populate the matrix using one-hot encoding logic
    # np.arange(n) creates indices [0, 1, ..., n-1]
    # serialized_grid provides the color index (row) for each position index (column)
    flattened_cylinder[serialized_grid, np.arange(n)] = 1

    return flattened_cylinder

def plot_flattened_cylinder(ax, flattened_grid, title="Flattened Cylinder"):
    """Helper function to plot the 10xN representation."""
    ax.imshow(flattened_grid, cmap='binary', aspect='auto', interpolation='nearest')
    ax.set_xlabel("Serialized Position (i)")
    ax.set_ylabel("Color (c)")
    ax.set_yticks(np.arange(10))
    ax.set_title(title)

def visualize_transformation(arc_grid):
    """Visualizes the original grid and its flattened cylinder representation."""
    flattened = arc_to_flattened_cylinder(arc_grid)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    plot_arc_grid(axes[0], arc_grid, title=f"Original ARC Grid ({arc_grid.shape[0]}x{arc_grid.shape[1]})")
    plot_flattened_cylinder(axes[1], flattened, title=f"Flattened Cylinder (10x{flattened.shape[1]})")

    plt.tight_layout()
    plt.show()

    return flattened

def plot_3d_cylinder_path(ax, grid, radius=1.0, title="Conceptual Cylinder Path"):
    """Plots the conceptual path on the cylinder surface."""
    if grid.ndim != 2:
        raise ValueError("Input grid must be 2-dimensional.")

    h, w = grid.shape
    n = h * w
    serialized_grid = grid.flatten()

    z = np.arange(n)
    colors = serialized_grid
    angles = colors * (2 * np.pi / 10) # Angle in radians

    x = radius * np.cos(angles)
    y = radius * np.sin(angles)

    # Plot the cylinder surface (optional, can be slow for large N)
    # theta_cyl = np.linspace(0, 2*np.pi, 100)
    # z_cyl = np.linspace(0, n-1, 50)
    # theta_cyl, z_cyl = np.meshgrid(theta_cyl, z_cyl)
    # x_cyl = radius * np.cos(theta_cyl)
    # y_cyl = radius * np.sin(theta_cyl)
    # ax.plot_surface(x_cyl, y_cyl, z_cyl, alpha=0.1, color='grey')


    # Plot the path, colored by the ARC color
    points = np.array([x, y, z]).T.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    for i in range(len(segments)):
       segment_color_index = serialized_grid[i] # Color of the starting point of the segment
       ax.plot(segments[i, :, 0], segments[i, :, 1], segments[i, :, 2],
               color=ARC_CMAP_LIST[segment_color_index], linewidth=1.5)

    # Plot points explicitly
    scatter_colors = [ARC_CMAP_LIST[c] for c in serialized_grid]
    ax.scatter(x, y, z, c=scatter_colors, marker='o', depthshade=True, s=20)


    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Serialized Position (i)")
    ax.set_title(title)
    # Make axes equal if desired, though Z will dominate
    # ax.set_aspect('equal') # Might stretch Z too much
    # Set limits to make it look more cylindrical
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    # Keep Z scale reasonable
    ax.set_zlim(z.min(), z.max())


# --- Example Usage ---

# Create a sample ARC-like grid (e.g., 5x4)
sample_grid = np.array([
    [0, 1, 1, 0],
    [2, 5, 5, 3],
    [2, 5, 5, 3],
    [0, 4, 4, 0],
    [0, 0, 0, 0]
], dtype=int)

print("Sample ARC Grid:")
print(sample_grid)
print("-" * 20)

# Perform the transformation and visualize 2D
flattened_representation = visualize_transformation(sample_grid)

print("\nFlattened Cylinder Representation (10xN):")
print(flattened_representation)
print("Shape:", flattened_representation.shape)
print("-" * 20)

# --- Optional: 3D Visualization ---
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
plot_3d_cylinder_path(ax, sample_grid, title="Conceptual Cylinder Path (Color=ARC Color)")
plt.show()

# --- Connection to MNIST-1D ---
print("\nConnection to MNIST-1D:")
# The flattened_representation is shape (10, N)
# We can view this as a sequence of N vectors, each of dimension 10.
# Example: The feature vector for the *first* serialized position (i=0) is:
print(f"Feature vector for position i=0: {flattened_representation[:, 0]}")
# Example: The feature vector for the *fifth* serialized position (i=4, start of row 2) is:
print(f"Feature vector for position i=4: {flattened_representation[:, 4]}")

print("\nThis sequence of 10-dim vectors can be processed by models (1D CNN, RNN, Transformer)")
print("similar to how MNIST-1D's 40-dim sequence vectors are processed.")