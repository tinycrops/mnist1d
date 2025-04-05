import numpy as np
from scipy import ndimage
from skimage.measure import label, regionprops
from collections import Counter
import copy

# --- Helper Functions ---

def get_neighbors(grid, r, c, connectivity=1):
    """Gets neighbors of a pixel (4 or 8 connectivity)."""
    neighbors = []
    rows, cols = grid.shape
    if connectivity == 1: # 4-connectivity (von Neumann)
        coords = [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]
    else: # 8-connectivity (Moore)
        coords = [(r-1, c-1), (r-1, c), (r-1, c+1),
                  (r,   c-1),           (r,   c+1),
                  (r+1, c-1), (r+1, c), (r+1, c+1)]

    for nr, nc in coords:
        if 0 <= nr < rows and 0 <= nc < cols:
            neighbors.append(((nr, nc), grid[nr, nc]))
    return neighbors

def apply_laplacian(grid):
    """Simple edge/contrast detection (approximates DoG)."""
    # Kernel for Laplacian
    kernel = np.array([[0, 1, 0],
                       [1,-4, 1],
                       [0, 1, 0]])
    # Convolve, non-background pixels are treated as 1, background as 0
    # This is a simplification, a better approach would handle colors.
    bw_grid = (grid != 0).astype(int)
    laplacian = ndimage.convolve(bw_grid, kernel, mode='constant', cval=0.0)
    return laplacian # High absolute values indicate edges/contrast

def get_objects(grid, background_color=0):
    """Identify connected components (Objects/CSS)."""
    objects = []
    # Create a binary mask where non-background pixels are True
    mask = grid != background_color
    labeled_grid, num_labels = label(mask, connectivity=2, return_num=True) # 8-connectivity

    props = regionprops(labeled_grid, intensity_image=grid)

    for i in range(num_labels):
        region = props[i]
        # Get the color (most frequent non-background color in the object)
        coords = region.coords
        colors = grid[coords[:, 0], coords[:, 1]]
        non_bg_colors = colors[colors != background_color]
        if len(non_bg_colors) > 0:
            color = Counter(non_bg_colors).most_common(1)[0][0]
        else:
             # Should not happen if mask is correct, but handle anyway
             color = background_color

        obj_data = {
            'id': i + 1,
            'label': region.label, # The label value in labeled_grid
            'coords': coords,
            'bbox': region.bbox, # (min_row, min_col, max_row, max_col)
            'centroid': region.centroid, # (row, col)
            'area': region.area,
            'color': int(color),
            'grid_repr': region.image_intensity # Mini-grid of the object
        }
        objects.append(obj_data)
    return objects, labeled_grid

def get_local_context_stats(grid, r, c, neighborhood_size=3, background_color=0):
    """Calculate stats in a neighborhood (for DN inspiration)."""
    rows, cols = grid.shape
    half = neighborhood_size // 2
    min_r, max_r = max(0, r - half), min(rows, r + half + 1)
    min_c, max_c = max(0, c - half), min(cols, c + half + 1)

    neighborhood = grid[min_r:max_r, min_c:max_c]
    unique_colors, counts = np.unique(neighborhood[neighborhood != background_color], return_counts=True)

    stats = {
        'num_unique_colors': len(unique_colors),
        'total_pixels': neighborhood.size,
        'non_bg_pixels': np.sum(neighborhood != background_color),
        'density': np.sum(neighborhood != background_color) / neighborhood.size if neighborhood.size > 0 else 0,
        'dominant_color': unique_colors[np.argmax(counts)] if len(unique_colors) > 0 else background_color,
        # Could add entropy, etc.
    }
    return stats

def check_symmetry(grid):
    """Check for horizontal and vertical symmetry (Global Structure)."""
    rows, cols = grid.shape
    is_horizontally_symmetric = np.array_equal(grid, np.flipud(grid))
    is_vertically_symmetric = np.array_equal(grid, np.fliplr(grid))
    # More complex symmetries (diagonal, rotational) are harder
    return {'horizontal': is_horizontally_symmetric, 'vertical': is_vertically_symmetric}

# --- Main Solver Sketch ---

def analyze_pair(pair):
    """Analyze a single training pair using neuro-inspired features."""
    input_grid = np.array(pair['input'])
    output_grid = np.array(pair['output'])

    analysis = {}

    # 1. Locality / Basic Features
    analysis['input_shape'] = input_grid.shape
    analysis['output_shape'] = output_grid.shape
    analysis['input_colors'] = np.unique(input_grid).tolist()
    analysis['output_colors'] = np.unique(output_grid).tolist()

    # 2. Objects (CSS / Grouping)
    analysis['input_objects'], analysis['input_labeled_grid'] = get_objects(input_grid)
    analysis['output_objects'], analysis['output_labeled_grid'] = get_objects(output_grid)
    analysis['num_input_objects'] = len(analysis['input_objects'])
    analysis['num_output_objects'] = len(analysis['output_objects'])

    # 3. Contrast / Edges (DoG)
    analysis['input_laplacian'] = apply_laplacian(input_grid)
    analysis['output_laplacian'] = apply_laplacian(output_grid)
    # Further analysis could compare laplacian maps, count edges etc.

    # 4. Context (DN) - Example for center pixel if grid exists
    if input_grid.size > 0:
         center_r, center_c = input_grid.shape[0] // 2, input_grid.shape[1] // 2
         analysis['input_center_context'] = get_local_context_stats(input_grid, center_r, center_c)
    if output_grid.size > 0:
         center_r, center_c = output_grid.shape[0] // 2, output_grid.shape[1] // 2
         analysis['output_center_context'] = get_local_context_stats(output_grid, center_r, center_c)

    # 5. Global Structure
    analysis['input_symmetry'] = check_symmetry(input_grid)
    analysis['output_symmetry'] = check_symmetry(output_grid)

    # --- Rule Inference (Highly Simplified) ---
    # This part is the hardest and where real AI/search would be needed.
    # We just look for simple patterns here.
    rules = []
    if analysis['input_shape'] == analysis['output_shape']:
        diff = input_grid != output_grid
        if np.sum(diff) > 0:
            rules.append("Color/Pixel Change")
        else:
            rules.append("No Change (?)") # Or maybe a filter?

    if analysis['num_input_objects'] == analysis['num_output_objects'] and analysis['num_input_objects'] > 0:
         # Could compare object properties (color, position, area)
         rules.append("Object Transformation/Recoloring")

    if analysis['num_output_objects'] > analysis['num_input_objects']:
         rules.append("Object Addition/Creation/Filling")

    if analysis['num_output_objects'] < analysis['num_input_objects']:
         rules.append("Object Deletion/Removal")

    if analysis['output_symmetry']['horizontal'] and not analysis['input_symmetry']['horizontal']:
         rules.append("Horizontal Reflection")
    # ... many more rule possibilities ...

    analysis['potential_rules'] = rules
    return analysis


def solve_arc_task(task):
    """Attempts to solve an ARC task using neuro-inspired analysis."""
    print(f"--- Analyzing Task ---")
    train_analyses = [analyze_pair(pair) for pair in task['train']]
    test_inputs = [np.array(pair['input']) for pair in task['test']]
    num_test = len(test_inputs)
    predictions = []

    # --- Very Simple Rule Application Strategy ---
    # Find the most common simple rule across training pairs
    all_rules = [rule for analysis in train_analyses for rule in analysis.get('potential_rules', [])]
    if not all_rules:
        print("Could not infer any simple rules.")
        # Default: return input unchanged (or maybe just the grid size)
        for i in range(num_test):
             predictions.append(test_inputs[i].tolist())
        return predictions

    rule_counts = Counter(all_rules)
    most_common_rule, _ = rule_counts.most_common(1)[0]
    print(f"Most common inferred rule (simplistic): {most_common_rule}")

    # Apply the most common rule (very basic implementation)
    for test_grid in test_inputs:
        predicted_grid = copy.deepcopy(test_grid) # Start with input

        if most_common_rule == "Horizontal Reflection":
            predicted_grid = np.flipud(test_grid)
        elif most_common_rule == "Vertical Reflection": # Need to add check above
            predicted_grid = np.fliplr(test_grid)
        # --- Add more rule implementations here! ---
        # e.g., Color Change: find which color changed to what in training
        # e.g., Object Movement: track centroids
        # e.g., Filling: identify objects and fill bounding boxes/contours
        # This requires much more sophisticated logic matching training examples.
        else:
            # Default if rule not implemented or complex
            print(f"Rule '{most_common_rule}' not implemented for application. Returning input.")
            predicted_grid = test_grid # Fallback

        predictions.append(predicted_grid.tolist()) # Convert back to list

    print(f"--- Finished Analysis ---")
    return predictions

# --- Example Usage (Requires ARC dataset files) ---
# import json
#
# def load_task(filename):
#     with open(filename, 'r') as f:
#         task = json.load(f)
#     return task
#
# # Replace with actual path to an ARC task JSON file
# task_file = 'path/to/arc/data/training/xxxxxxxx.json'
# try:
#     task_data = load_task(task_file)
#     predicted_outputs = solve_arc_task(task_data)
#
#     print("\nTest Inputs:")
#     for i, pair in enumerate(task_data['test']):
#         print(f"Input {i}:\n{np.array(pair['input'])}")
#
#     print("\nPredicted Outputs:")
#     for i, output in enumerate(predicted_outputs):
#         print(f"Prediction {i}:\n{np.array(output)}")
#
#     print("\nActual Outputs (for comparison):")
#      for i, pair in enumerate(task_data['test']):
#         if 'output' in pair: # Check if actual output is available
#             print(f"Actual {i}:\n{np.array(pair['output'])}")
#         else:
#             print(f"Actual {i}: Not available in this test set part.")
#
# except FileNotFoundError:
#      print(f"Error: Task file not found at {task_file}")
# except Exception as e:
#      print(f"An error occurred: {e}")