import json
import numpy as np
import os
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple

def load_arc_task(filepath: str) -> Dict:
    """
    Load an ARC task from a JSON file.
    
    Args:
        filepath: Path to the JSON file
        
    Returns:
        Dictionary containing the ARC task
    """
    with open(filepath, 'r') as f:
        task = json.load(f)
    return task

def convert_2d_to_1d(grid: List[List[int]], method: str = 'flatten') -> List[int]:
    """
    Convert a 2D ARC grid to a 1D sequence.
    
    Args:
        grid: 2D grid from an ARC task
        method: Method to use for conversion
            - 'flatten': Flatten the grid row by row
            - 'histogram': Create a color histogram (counts of each color)
            - 'row_concat': Concatenate rows with separators
            - 'col_concat': Concatenate columns with separators
            
    Returns:
        1D sequence representing the grid
    """
    grid_np = np.array(grid)
    
    if method == 'flatten':
        # Simply flatten the grid row by row
        return grid_np.flatten().tolist()
    
    elif method == 'histogram':
        # Create a histogram of color values
        result = []
        for color in range(10):  # ARC uses colors 0-9
            count = np.sum(grid_np == color)
            # Represent the count by repeating the color that many times
            result.extend([color] * int(count))
        return result
    
    elif method == 'row_concat':
        # Concatenate rows with separator (-1)
        result = []
        for row in grid_np:
            result.extend(row.tolist())
            result.append(-1)  # Separator
        return result[:-1]  # Remove last separator
    
    elif method == 'col_concat':
        # Concatenate columns with separator (-1)
        result = []
        for col in grid_np.T:
            result.extend(col.tolist())
            result.append(-1)  # Separator
        return result[:-1]  # Remove last separator
    
    else:
        raise ValueError(f"Unknown conversion method: {method}")

def pad_sequence(sequence: List[int], target_length: int = 40) -> List[int]:
    """
    Pad or truncate a sequence to the target length.
    
    Args:
        sequence: Input sequence
        target_length: Desired length
        
    Returns:
        Padded or truncated sequence
    """
    if len(sequence) > target_length:
        # Truncate
        return sequence[:target_length]
    elif len(sequence) < target_length:
        # Pad with zeros
        return sequence + [0] * (target_length - len(sequence))
    else:
        return sequence

def convert_arc_task_to_1d(task: Dict, method: str = 'flatten', target_length: int = 40) -> Dict:
    """
    Convert an entire ARC task to 1D sequences.
    
    Args:
        task: ARC task dictionary
        method: Conversion method
        target_length: Target sequence length
        
    Returns:
        Dictionary with 1D sequences
    """
    result = {
        'train': [],
        'test': []
    }
    
    # Convert training examples
    for example in task['train']:
        input_1d = pad_sequence(convert_2d_to_1d(example['input'], method), target_length)
        output_1d = pad_sequence(convert_2d_to_1d(example['output'], method), target_length)
        
        result['train'].append({
            'input': input_1d,
            'output': output_1d
        })
    
    # Convert test examples
    for example in task['test']:
        input_1d = pad_sequence(convert_2d_to_1d(example['input'], method), target_length)
        output_1d = pad_sequence(convert_2d_to_1d(example['output'], method), target_length)
        
        result['test'].append({
            'input': input_1d,
            'output': output_1d
        })
    
    return result

def visualize_1d_task(task_1d: Dict):
    """
    Visualize a 1D ARC task.
    
    Args:
        task_1d: 1D ARC task dictionary
    """
    n_train = len(task_1d['train'])
    n_test = len(task_1d['test'])
    
    fig, axes = plt.subplots(n_train + n_test, 2, figsize=(10, 2 * (n_train + n_test)))
    
    # Plot training examples
    for i, example in enumerate(task_1d['train']):
        ax1, ax2 = axes[i]
        
        # Plot input
        ax1.imshow(np.array([example['input']]), aspect='auto', cmap='tab10')
        ax1.set_title(f'Train {i+1} Input')
        ax1.set_yticks([])
        
        # Plot output
        ax2.imshow(np.array([example['output']]), aspect='auto', cmap='tab10')
        ax2.set_title(f'Train {i+1} Output')
        ax2.set_yticks([])
    
    # Plot test examples
    for i, example in enumerate(task_1d['test']):
        ax1, ax2 = axes[n_train + i]
        
        # Plot input
        ax1.imshow(np.array([example['input']]), aspect='auto', cmap='tab10')
        ax1.set_title(f'Test {i+1} Input')
        ax1.set_yticks([])
        
        # Plot output
        ax2.imshow(np.array([example['output']]), aspect='auto', cmap='tab10')
        ax2.set_title(f'Test {i+1} Output')
        ax2.set_yticks([])
    
    plt.tight_layout()
    plt.show()

def infer_rule_from_examples(task_1d: Dict) -> str:
    """
    Attempt to infer the rule from the 1D examples.
    This is a simple heuristic approach that works for some basic rules.
    
    Args:
        task_1d: 1D ARC task dictionary
        
    Returns:
        String describing the inferred rule
    """
    # Check if it's a simple shift
    def is_shift(input_seq, output_seq, direction):
        for shift in range(1, 5):  # Try shifts 1-4
            shifted = np.zeros_like(input_seq)
            if direction == 'right':
                for i in range(len(input_seq)):
                    if input_seq[i] > 0:
                        new_i = min(i + shift, len(input_seq) - 1)
                        shifted[new_i] = input_seq[i]
            else:  # left
                for i in range(len(input_seq)):
                    if input_seq[i] > 0:
                        new_i = max(i - shift, 0)
                        shifted[new_i] = input_seq[i]
            
            if np.array_equal(shifted, output_seq):
                return f"{direction} shift by {shift}"
        return None
    
    # Check if it's a color change (recoloring)
    def is_recolor(input_seq, output_seq):
        color_map = {}
        for i in range(len(input_seq)):
            if input_seq[i] > 0:
                if input_seq[i] not in color_map:
                    color_map[input_seq[i]] = output_seq[i]
                elif color_map[input_seq[i]] != output_seq[i]:
                    return None
        
        # Check if the mapping works for all elements
        for i in range(len(input_seq)):
            if input_seq[i] > 0:
                if output_seq[i] != color_map[input_seq[i]]:
                    return None
        
        return f"recolor: {color_map}"
    
    # Check if it's a flip
    def is_flip(input_seq, output_seq):
        non_zero_input = [x for x in input_seq if x > 0]
        non_zero_output = [x for x in output_seq if x > 0]
        
        if non_zero_input == non_zero_output[::-1]:
            return "flip (reverse)"
        return None
    
    # Check each training example
    rules = []
    for example in task_1d['train']:
        input_seq = np.array(example['input'])
        output_seq = np.array(example['output'])
        
        # Try each rule type
        rule = None
        rule = is_shift(input_seq, output_seq, 'right')
        if rule is None:
            rule = is_shift(input_seq, output_seq, 'left')
        if rule is None:
            rule = is_recolor(input_seq, output_seq)
        if rule is None:
            rule = is_flip(input_seq, output_seq)
        
        if rule:
            rules.append(rule)
        else:
            rules.append("unknown")
    
    # Check if all examples follow the same rule
    if len(set(rules)) == 1 and rules[0] != "unknown":
        return rules[0]
    else:
        return "complex or inconsistent rule"

def convert_arc_directory(input_dir: str, output_dir: str, method: str = 'flatten', target_length: int = 40):
    """
    Convert all ARC tasks in a directory to 1D format.
    
    Args:
        input_dir: Directory containing ARC JSON files
        output_dir: Directory to save converted tasks
        method: Conversion method
        target_length: Target sequence length
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Process all JSON files
    for filename in os.listdir(input_dir):
        if filename.endswith('.json'):
            filepath = os.path.join(input_dir, filename)
            
            try:
                # Load and convert the task
                task = load_arc_task(filepath)
                task_1d = convert_arc_task_to_1d(task, method, target_length)
                
                # Save the converted task
                output_path = os.path.join(output_dir, f"1d_{filename}")
                with open(output_path, 'w') as f:
                    json.dump(task_1d, f)
                
                print(f"Converted {filename} to 1D format")
                
            except Exception as e:
                print(f"Error converting {filename}: {e}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert ARC tasks to 1D format')
    parser.add_argument('--input', '-i', type=str, required=True, help='Input ARC JSON file or directory')
    parser.add_argument('--output', '-o', type=str, help='Output file or directory')
    parser.add_argument('--method', '-m', type=str, default='flatten', 
                        choices=['flatten', 'histogram', 'row_concat', 'col_concat'],
                        help='Method to convert 2D grids to 1D sequences')
    parser.add_argument('--length', '-l', type=int, default=40, help='Target sequence length')
    parser.add_argument('--visualize', '-v', action='store_true', help='Visualize the converted task')
    parser.add_argument('--infer', action='store_true', help='Try to infer the rule')
    
    args = parser.parse_args()
    
    if os.path.isfile(args.input):
        # Process a single file
        task = load_arc_task(args.input)
        task_1d = convert_arc_task_to_1d(task, args.method, args.length)
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(task_1d, f)
            print(f"Saved 1D task to {args.output}")
        
        if args.visualize:
            visualize_1d_task(task_1d)
        
        if args.infer:
            rule = infer_rule_from_examples(task_1d)
            print(f"Inferred rule: {rule}")
            
    elif os.path.isdir(args.input):
        # Process a directory
        output_dir = args.output or f"{args.input}_1d"
        convert_arc_directory(args.input, output_dir, args.method, args.length)
    
    else:
        print(f"Input {args.input} is not a valid file or directory")

if __name__ == "__main__":
    main() 