

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import math


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

def create_arc_helix_features(grid, periods=[2, 3, 5]):
    """
    Creates an enriched feature matrix for an ARC grid, incorporating
    linear and periodic spatial features (helix-inspired).

    Args:
        grid (np.ndarray): Input ARC grid (H x W).
        periods (list): List of periods P to use for periodic features.

    Returns:
        np.ndarray: Feature matrix of shape (N, feature_dim), where N = H * W.
    """
    if grid.ndim != 2:
        raise ValueError("Input grid must be 2-dimensional.")

    h, w = grid.shape
    n = h * w
    serialized_grid = grid.flatten()

    # Calculate feature dimensions
    color_dim = 10
    linear_spatial_dim = 2
    periodic_spatial_dim = 2 * (2 + len(periods) * 2) # sin/cos for row/col relative to H/W and each period P
    # Add other dims if needed (e.g., distance to edge)
    feature_dim = color_dim + linear_spatial_dim + periodic_spatial_dim

    feature_matrix = np.zeros((n, feature_dim))

    for i in range(n):
        row = i // w
        col = i % w
        color = serialized_grid[i]

        current_feature_idx = 0

        # 1. Color Features (One-Hot)
        if 0 <= color <= 9:
            feature_matrix[i, current_feature_idx + color] = 1
        current_feature_idx += color_dim

        # 2. Linear Spatial Features (Normalized)
        feature_matrix[i, current_feature_idx] = row / (h - 1) if h > 1 else 0.5
        feature_matrix[i, current_feature_idx + 1] = col / (w - 1) if w > 1 else 0.5
        current_feature_idx += linear_spatial_dim

        # 3. Periodic Spatial Features (sin/cos)
        # Relative to grid dimensions
        angle_row_h = 2 * math.pi * row / h if h > 0 else 0
        feature_matrix[i, current_feature_idx] = math.sin(angle_row_h)
        feature_matrix[i, current_feature_idx + 1] = math.cos(angle_row_h)
        angle_col_w = 2 * math.pi * col / w if w > 0 else 0
        feature_matrix[i, current_feature_idx + 2] = math.sin(angle_col_w)
        feature_matrix[i, current_feature_idx + 3] = math.cos(angle_col_w)
        current_feature_idx += 4

        # Relative to fixed periods P
        for p in periods:
            angle_row_p = 2 * math.pi * row / p
            feature_matrix[i, current_feature_idx] = math.sin(angle_row_p)
            feature_matrix[i, current_feature_idx + 1] = math.cos(angle_row_p)
            angle_col_p = 2 * math.pi * col / p
            feature_matrix[i, current_feature_idx + 2] = math.sin(angle_col_p)
            feature_matrix[i, current_feature_idx + 3] = math.cos(angle_col_p)
            current_feature_idx += 4

        # Add other features here if desired

    return feature_matrix

def plot_feature_matrix(ax, feature_matrix, title="Enriched Feature Matrix"):
    """Plots the N x feature_dim matrix."""
    im = ax.imshow(feature_matrix.T, aspect='auto', interpolation='nearest', cmap='viridis') # Transpose for features as rows
    ax.set_xlabel("Serialized Position (i)")
    ax.set_ylabel("Feature Dimension")
    ax.set_title(title)
    # Add a colorbar
    plt.colorbar(im, ax=ax)


# --- Example Usage ---
sample_grid_helix = np.array([
    [0, 1, 1, 0],
    [2, 5, 5, 3],
    [2, 5, 5, 3],
    [0, 4, 4, 0],
    [0, 0, 0, 0]
], dtype=int)

# Create the enriched features
arc_helix_features = create_arc_helix_features(sample_grid_helix)

print("\nEnriched ARC 'Helix' Feature Matrix (N x feature_dim):")
print("Shape:", arc_helix_features.shape)
# print(arc_helix_features) # Can be large
print("-" * 20)
print("Example feature vector for first pixel (i=0, row=0, col=0, color=0):")
print(arc_helix_features[0, :])
print("\nExample feature vector for pixel i=5 (row=1, col=1, color=5):")
print(arc_helix_features[5, :])
print("-" * 20)

# Visualize the original grid and the new feature matrix
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
plot_arc_grid(axes[0], sample_grid_helix, f"Original Grid ({sample_grid_helix.shape[0]}x{sample_grid_helix.shape[1]})")
plot_feature_matrix(axes[1], arc_helix_features, f"Enriched Features ({arc_helix_features.shape[1]} dims)")
plt.tight_layout()
plt.show()

# --- Conceptual 3D Plot (Illustrative - using only a few features) ---
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

n_plot = arc_helix_features.shape[0]
z_plot = np.arange(n_plot)

# Select features for plotting: e.g., cos(row/H), sin(col/W), maybe color index directly?
# Dimensions: color=10, linear=2, periodic=16. Total=28
# Let's plot cos(row/H) (index 13), sin(col/W) (index 14), color (index 0-9, use max as proxy)
feat_idx_x = 10 + 2 + 1 # cos(row/H)
feat_idx_y = 10 + 2 + 2 # sin(col/W)

x_plot = arc_helix_features[:, feat_idx_x]
y_plot = arc_helix_features[:, feat_idx_y]
# Use original colors for plotting the path points
plot_colors = [ARC_CMAP_LIST[c] for c in sample_grid_helix.flatten()]

# Plot path segments (colored by start point color)
points_3d = np.array([x_plot, y_plot, z_plot]).T.reshape(-1, 1, 3)
segments_3d = np.concatenate([points_3d[:-1], points_3d[1:]], axis=1)
colors_flat = sample_grid_helix.flatten()
for i in range(len(segments_3d)):
   segment_color_index = colors_flat[i]
   ax.plot(segments_3d[i, :, 0], segments_3d[i, :, 1], segments_3d[i, :, 2],
           color=ARC_CMAP_LIST[segment_color_index], linewidth=1)

ax.scatter(x_plot, y_plot, z_plot, c=plot_colors, marker='o', s=15, depthshade=True)

ax.set_xlabel(f"Feature {feat_idx_x} (e.g., cos(row/H))")
ax.set_ylabel(f"Feature {feat_idx_y} (e.g., sin(col/W))")
ax.set_zlabel("Serialized Position (i)")
ax.set_title("Conceptual 3D Projection of 'ARC Helix' Path")
plt.show()