import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
import math

# --- ARC Color Definitions ---
ARC_CMAP_LIST = [
    "#000000", "#0074D9", "#FF4136", "#2ECC40", "#FFDC00",
    "#AAAAAA", "#F012BE", "#FF851B", "#7FDBFF", "#870C25",
]
ARC_CMAP = mcolors.ListedColormap(ARC_CMAP_LIST)
ARC_NORM = mcolors.Normalize(vmin=0, vmax=9)

# --- Configuration ---
# Define whether to include one-hot color encoding globally
USE_ONE_HOT_COLOR = True
SPATIAL_PERIODS = [2, 3, 5] # Periods for spatial features
MUNSELL_COLOR_DIM = 4 # sinH, cosH, V, C
LINEAR_SPATIAL_DIM = 2
PERIODIC_SPATIAL_DIM = 2 * (2 + len(SPATIAL_PERIODS) * 2)
ONE_HOT_COLOR_DIM = 10 if USE_ONE_HOT_COLOR else 0


def plot_arc_grid(ax, grid, title="ARC Grid"):
    ax.imshow(grid, cmap=ARC_CMAP, norm=ARC_NORM, interpolation='nearest')
    ax.set_xticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid.shape[0], 1), minor=True)
    ax.grid(which='minor', color='grey', linestyle='-', linewidth=0.5)
    ax.tick_params(which='minor', size=0)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title)

# --- Step 1: Approximate Munsell Mapping for ARC Colors ---
ARC_TO_MUNSELL_APPROX = {
    0: (None, 0, 0), 1: (240,  6, 10), 2: (0,    5, 12), 3: (120,  6, 10), 4: (60,   8, 12),
    5: (None, 5, 0), 6: (300,  5, 12), 7: (30,   6, 12), 8: (210,  8, 6),  9: (0,    3, 8),
}
MAX_CHROMA_APPROX = 14

def map_arc_to_munsell_features(arc_color):
    if arc_color not in ARC_TO_MUNSELL_APPROX:
        h, v, c = ARC_TO_MUNSELL_APPROX[0]
    else:
        h, v, c = ARC_TO_MUNSELL_APPROX[arc_color]
    norm_v = v / 10.0
    norm_c = c / MAX_CHROMA_APPROX
    if h is not None:
        h_rad = math.radians(h)
        sin_h = math.sin(h_rad)
        cos_h = math.cos(h_rad)
    else:
        sin_h = 0.0; cos_h = 0.0
    return [sin_h, cos_h, norm_v, norm_c]

# --- Step 2: Create Enriched Spatial-Color Helix Features ---

def create_arc_spatial_color_helix_features(grid):
    """ Uses the global SPATIAL_PERIODS and USE_ONE_HOT_COLOR """
    if grid.ndim != 2:
        raise ValueError("Input grid must be 2-dimensional.")
    h, w = grid.shape
    n = h * w
    serialized_grid = grid.flatten()

    # Use global constants for dimensions
    feature_dim = (ONE_HOT_COLOR_DIM +
                   LINEAR_SPATIAL_DIM +
                   PERIODIC_SPATIAL_DIM +
                   MUNSELL_COLOR_DIM)

    feature_matrix = np.zeros((n, feature_dim))

    for i in range(n):
        row = i // w
        col = i % w
        arc_color = serialized_grid[i]
        current_feature_idx = 0

        # 1. (Optional) One-Hot Color Features
        if USE_ONE_HOT_COLOR:
            if 0 <= arc_color <= 9:
                feature_matrix[i, current_feature_idx + arc_color] = 1
            current_feature_idx += ONE_HOT_COLOR_DIM

        # 2. Linear Spatial Features
        feature_matrix[i, current_feature_idx] = row / (h - 1) if h > 1 else 0.5
        feature_matrix[i, current_feature_idx + 1] = col / (w - 1) if w > 1 else 0.5
        current_feature_idx += LINEAR_SPATIAL_DIM

        # 3. Periodic Spatial Features
        angle_row_h = 2 * math.pi * row / h if h > 0 else 0
        feature_matrix[i, current_feature_idx:current_feature_idx+2] = [math.sin(angle_row_h), math.cos(angle_row_h)]
        angle_col_w = 2 * math.pi * col / w if w > 0 else 0
        feature_matrix[i, current_feature_idx+2:current_feature_idx+4] = [math.sin(angle_col_w), math.cos(angle_col_w)]
        current_feature_idx += 4
        for p in SPATIAL_PERIODS:
            angle_row_p = 2 * math.pi * row / p
            feature_matrix[i, current_feature_idx:current_feature_idx+2] = [math.sin(angle_row_p), math.cos(angle_row_p)]
            angle_col_p = 2 * math.pi * col / p
            feature_matrix[i, current_feature_idx+2:current_feature_idx+4] = [math.sin(angle_col_p), math.cos(angle_col_p)]
            current_feature_idx += 4

        # 4. Munsell-Inspired Color Features
        munsell_features = map_arc_to_munsell_features(arc_color)
        feature_matrix[i, current_feature_idx : current_feature_idx + MUNSELL_COLOR_DIM] = munsell_features
        current_feature_idx += MUNSELL_COLOR_DIM

    return feature_matrix

# --- Step 3: Visualization ---

def plot_feature_matrix(ax, feature_matrix, title="Enriched Feature Matrix"):
    im = ax.imshow(feature_matrix.T, aspect='auto', interpolation='nearest', cmap='viridis')
    ax.set_xlabel("Serialized Position (i)")
    ax.set_ylabel("Feature Dimension")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, shrink=0.6)

def plot_3d_helix_projection(ax, grid, feature_matrix, title="Conceptual 3D Projection"):
    """ Uses the global USE_ONE_HOT_COLOR and SPATIAL_PERIODS """
    n_plot = feature_matrix.shape[0]
    z_plot = np.arange(n_plot)

    # --- Calculate indices based on global constants ---
    idx_base_lin_sp = ONE_HOT_COLOR_DIM
    idx_base_per_sp = idx_base_lin_sp + LINEAR_SPATIAL_DIM
    idx_base_munsell = idx_base_per_sp + PERIODIC_SPATIAL_DIM

    # Example features to plot
    feat_idx_x = idx_base_per_sp + 3 # cos(col/W)
    feat_idx_y = idx_base_munsell + 0 # sin(Hue)
    feat_label_x = "cos(col/W)"
    feat_label_y = "sin(Hue)"

    x_plot = feature_matrix[:, feat_idx_x]
    y_plot = feature_matrix[:, feat_idx_y]
    plot_colors = [ARC_CMAP_LIST[c] for c in grid.flatten()]

    # Plot path segments
    points_3d = np.array([x_plot, y_plot, z_plot]).T.reshape(-1, 1, 3)
    segments_3d = np.concatenate([points_3d[:-1], points_3d[1:]], axis=1)
    colors_flat = grid.flatten()
    for i in range(len(segments_3d)):
       segment_color_index = colors_flat[i]
       ax.plot(segments_3d[i, :, 0], segments_3d[i, :, 1], segments_3d[i, :, 2],
               color=ARC_CMAP_LIST[segment_color_index], linewidth=1, alpha=0.8)

    # Plot points
    non_bg_indices = np.where(colors_flat != 0)[0]
    if len(non_bg_indices) > 0 :
         ax.scatter(x_plot[non_bg_indices], y_plot[non_bg_indices], z_plot[non_bg_indices],
                    c=np.array(plot_colors)[non_bg_indices], marker='o', s=20, depthshade=True)
    else:
         ax.scatter(x_plot, y_plot, z_plot, c=plot_colors, marker='o', s=20, depthshade=True)

    ax.set_xlabel(f"Feature {feat_idx_x} ({feat_label_x})")
    ax.set_ylabel(f"Feature {feat_idx_y} ({feat_label_y})")
    ax.set_zlabel("Serialized Position (i)")
    ax.set_title(title)

# --- Example Usage ---
sample_grid_munsell = np.array([
    [0, 1, 1, 0],
    [2, 5, 5, 3],
    [2, 5, 5, 3],
    [0, 4, 4, 0],
    [6, 0, 0, 7]
], dtype=int)

# Create the spatial-color helix features
spatial_color_helix_features = create_arc_spatial_color_helix_features(sample_grid_munsell)

print("\nEnriched Spatial-Color 'Helix' Feature Matrix (N x feature_dim):")
print("Shape:", spatial_color_helix_features.shape)
print("-" * 20)
print("Example feature vector for pixel i=5 (row=1, col=1, color=5 - Grey):")
print(spatial_color_helix_features[5, :])
print("\nExample feature vector for pixel i=16 (row=4, col=0, color=6 - Magenta):")
print(spatial_color_helix_features[16, :])
print("-" * 20)

# --- Visualize ---
fig1, axes1 = plt.subplots(1, 2, figsize=(14, 6))
plot_arc_grid(axes1[0], sample_grid_munsell, f"Original Grid ({sample_grid_munsell.shape[0]}x{sample_grid_munsell.shape[1]})")
plot_feature_matrix(axes1[1], spatial_color_helix_features, f"Spatial-Color Features ({spatial_color_helix_features.shape[1]} dims)")
plt.tight_layout()

fig2 = plt.figure(figsize=(9, 9))
ax_3d = fig2.add_subplot(111, projection='3d')
plot_3d_helix_projection(ax_3d, sample_grid_munsell, spatial_color_helix_features,
                         title="Spatial-Color 'Helix' Projection")
plt.tight_layout()
plt.show()