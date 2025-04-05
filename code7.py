import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.ndimage import gaussian_filter
import imageio # Optional: for creating a GIF
from tqdm import tqdm # Changed from tqdm.notebook to regular tqdm

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
    ax.grid(which='minor', color='white', linestyle='-', linewidth=0.5, alpha=0.3)
    ax.tick_params(which='minor', size=0)
    # Hide major ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, color='white')
    # Make spines invisible
    for spine in ax.spines.values():
        spine.set_visible(False)


def simulate_pond_interference(arc_grid, steps_per_drop=5, decay=0.98, diffusion_sigma=1.0, impulse_strength=5.0, output_gif=False, gif_filename='pond_simulation.gif'):
    """
    Simulates waves interfering on a 2D surface based on serialized ARC grid input.

    Args:
        arc_grid (np.ndarray): The input ARC grid.
        steps_per_drop (int): Number of simulation steps between dropping stones.
        decay (float): Wave amplitude decay factor per step.
        diffusion_sigma (float): Sigma for Gaussian filter simulating wave spread.
        impulse_strength (float): Initial amplitude of a dropped stone's wave.
        output_gif (bool): Whether to save the simulation frames as a GIF.
        gif_filename (str): Filename for the output GIF.
    """
    h, w = arc_grid.shape
    pond_size_factor = 3 # Make the pond larger than the grid to see propagation
    pond_h, pond_w = h * pond_size_factor, w * pond_size_factor
    pond_state = np.zeros((pond_h, pond_w), dtype=float)

    # Serialize grid and map grid coordinates to pond coordinates
    serialized_colors = arc_grid.flatten()
    grid_coords = [(r, c) for r in range(h) for c in range(w)]
    # Center the grid drop points within the larger pond
    offset_r = (pond_h - h) // 2
    offset_c = (pond_w - w) // 2
    pond_coords = [(r + offset_r, c + offset_c) for r, c in grid_coords]

    num_steps = len(serialized_colors) * steps_per_drop
    frames = []

    print(f"Simulating {num_steps} steps...")
    # Use tqdm for progress
    step_iterator = tqdm(range(num_steps))


    fig, (ax_grid, ax_pond) = plt.subplots(1, 2, figsize=(12, 6), facecolor='black')


    for step in step_iterator:
        # --- Drop a stone periodically ---
        if step % steps_per_drop == 0:
            drop_index = step // steps_per_drop
            if drop_index < len(serialized_colors):
                r_pond, c_pond = pond_coords[drop_index]
                color = serialized_colors[drop_index] # Get the color
                # Add impulse - could scale strength by color if desired
                pond_state[r_pond, c_pond] += impulse_strength
                # Optional: print(f"Step {step}: Dropped stone for color {color} at ({r_pond}, {c_pond})")


        # --- Simulate wave propagation (diffusion) and decay ---
        # Apply Gaussian filter to spread the waves
        if diffusion_sigma > 0:
             # Use 'wrap' mode to handle boundaries like a continuous surface
            pond_state = gaussian_filter(pond_state, sigma=diffusion_sigma, mode='wrap')

        # Apply decay
        pond_state *= decay

        # --- Visualization (optional, update periodically) ---
        if step % steps_per_drop == 0 or step == num_steps - 1 : # Visualize every drop + final state
             # Update plots
            ax_grid.clear()
            plot_arc_grid(ax_grid, arc_grid)
            # Highlight the cell corresponding to the *last* dropped stone (if any)
            if step % steps_per_drop == 0 and drop_index < len(grid_coords):
                 r_grid, c_grid = grid_coords[drop_index]
                 rect = plt.Rectangle((c_grid-0.5, r_grid-0.5), 1, 1, fill=False, edgecolor='yellow', linewidth=2)
                 ax_grid.add_patch(rect)


            ax_pond.clear()
            # Use a colormap suitable for waves (e.g., diverging like 'coolwarm' or 'bwr')
            # Center the colormap around zero
            max_abs_val = np.max(np.abs(pond_state)) if np.max(np.abs(pond_state)) > 1e-6 else 1.0
            norm = mcolors.Normalize(vmin=-max_abs_val, vmax=max_abs_val)
            im = ax_pond.imshow(pond_state, cmap='coolwarm', interpolation='bilinear', norm=norm)
            ax_pond.set_title(f"Pond State (Step {step+1}/{num_steps})", color='white')
            ax_pond.set_xticks([])
            ax_pond.set_yticks([])
            for spine in ax_pond.spines.values():
                spine.set_edgecolor('white') # Make border visible

            plt.tight_layout()
            # If saving GIF, capture frame
            if output_gif:
                fig.canvas.draw() # Draw the canvas, cache the renderer
                frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
                frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                frames.append(frame)
            else:
                plt.pause(0.01) # Pause briefly to allow plot update


    if not output_gif:
        plt.show() # Keep final plot open

    # Save GIF if requested
    if output_gif and frames:
        print(f"Saving GIF ({len(frames)} frames) to {gif_filename}...")
        # Calculate duration based on number of frames, aim for ~5-10 seconds total
        duration_ms = max(50, (10 * 1000) / len(frames)) # ms per frame
        imageio.mimsave(gif_filename, frames, duration=duration_ms)
        print("GIF saved.")
        plt.close(fig) # Close figure after saving GIF


# --- Example Usage ---

# Example ARC Grid (you can replace this with an actual task grid)
# Task: cea4e149.json (example input 1)
arc_grid_example = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 5, 5, 0, 0, 0, 0, 0],
    [0, 5, 5, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 8, 8, 0, 0],
    [0, 0, 0, 0, 8, 8, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0]
], dtype=int)

# Run the simulation
# Note: Setting output_gif=True can take time and requires imageio
simulate_pond_interference(
    arc_grid_example,
    steps_per_drop=2,      # Fewer steps between drops for faster simulation
    decay=0.97,            # Slightly less decay
    diffusion_sigma=0.8,   # Less diffusion for sharper initial waves
    impulse_strength=10.0, # Stronger initial impulse
    output_gif=False,       # Set to True to create a GIF (requires imageio)
    gif_filename='arc_pond_sim.gif'
)

print("\nSimulation finished. The final plot shows the complex interference pattern.")
print("This pattern encodes the history of all the 'stone drops' (grid cell inputs).")