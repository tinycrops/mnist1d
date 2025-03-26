# ARC-1D: A Minimal 1D Adaptation of the ARC AGI Challenge

## Inspiration from MNIST-1D

This project is inspired by the MNIST-1D approach described in the paper "Scaling Down Deep Learning with MNIST-1D" by Sam Greydanus and Dmitry Kobak. Just as MNIST-1D reduced the dimensionality of the MNIST dataset to focus on fundamental deep learning concepts, ARC-1D reduces the dimensionality of the Abstraction and Reasoning Corpus (ARC) to create a minimal, computationally efficient testbed for exploring rule learning and discovery.

Key similarities to MNIST-1D:
- Low dimensionality (40 data points per sample)
- Procedurally generated with controlled parameters
- Fast training time (minutes instead of hours)
- Designed to differentiate between model architectures based on their inductive biases
- Enables rapid experimentation for research on a low budget

## Components

This implementation has three main components:

1. **ARC-1D Generator** (`arc_1d_minimal.py`): A procedural generator for 1D ARC-like tasks with predefined rule templates.
2. **ARC to 1D Converter** (`arc_to_1d.py`): A tool to convert existing 2D ARC tasks to 1D sequences for training and testing.
3. **Model Implementations**: Simple CNN, RNN, and MLP models for comparing different architectural biases on ARC-1D tasks.

## ARC-1D Rules

The ARC-1D generator includes several rule templates inspired by common patterns in ARC tasks:

- `rule_move_right/left`: Move colored blocks in a specified direction
- `rule_recolor`: Change colors based on a mapping
- `rule_fill_gap`: Fill gaps between blocks of the same color
- `rule_mirror`: Mirror the pattern around a central point
- `rule_count_color`: Count occurrences of colors and create a histogram
- `rule_flip`: Flip/reverse the entire sequence

## Usage

### Generating a dataset

```python
from arc_1d_minimal import ARC1D

# Create ARC1D generator
arc1d = ARC1D(grid_size=40, num_samples=4000, random_seed=42)

# Generate a dataset
dataset = arc1d.generate_dataset()

# Save to a file
arc1d.save_dataset('arc1d_dataset.json')

# Visualize samples
arc1d.visualize_sample(dataset['train'][0])
```

### Converting real ARC tasks to 1D

```
python arc_to_1d.py --input arc-dataset-collection/dataset/ARC-AGI-2/data/training --output arc1d_converted --method flatten
```

### Training models

```python
from arc_1d_minimal import ARC1D, ARC1DDataset, SimpleConvNet, train_model
from torch.utils.data import DataLoader

# Create dataset
arc1d = ARC1D(grid_size=40, num_samples=4000)
dataset = arc1d.generate_dataset()

# Create PyTorch datasets and dataloaders
train_dataset = ARC1DDataset(dataset['train'])
test_dataset = ARC1DDataset(dataset['test'])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Train a CNN model
cnn_model = SimpleConvNet()
train_losses, test_accuracies = train_model(cnn_model, train_loader, test_loader, num_epochs=10)
```

## Why ARC-1D?

The ARC-AGI Challenge involves complex 2D tasks with multiple, often unforeseeable rules. By reducing the problem to 1D, we:

1. **Simplify computation**: Train models in minutes instead of hours
2. **Focus on rule learning**: Isolate the core challenge of discovering rules
3. **Compare architectures**: Test which model types are best at learning specific types of rules
4. **Rapid prototyping**: Quickly test ideas before scaling to full ARC

Just as MNIST-1D allows for studying fundamental deep learning concepts like lottery tickets, double descent, and meta-learning at low computational cost, ARC-1D enables rapid exploration of rule learning approaches for abstraction and reasoning.

## Key Insights

Experiments with ARC-1D reveal several key insights:

1. **Architectural biases matter**: CNNs outperform MLPs on spatial transformation rules due to their inductive biases
2. **Rule discovery varies by model**: Different architectures are better at discovering different types of rules
3. **Compositional rules are hardest**: Rules that combine multiple transformations are the most challenging

These insights can guide the development of more sophisticated approaches for the full 2D ARC challenge.

## Extensions

Possible extensions to this project:

1. **Curriculum learning**: Gradually increase the complexity of rules
2. **Meta-learning**: Train models to discover rules from few examples
3. **Compositional rules**: Create rules that combine multiple transformations
4. **Neuro-symbolic approaches**: Combine neural networks with symbolic reasoning
5. **Scaling up**: Apply insights from ARC-1D to the full 2D ARC challenge

## Requirements

- Python 3.7+
- PyTorch 1.7+
- NumPy
- Matplotlib
