import numpy as np
import json
import os
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Any
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random

class ARC1D:
    """
    ARC1D: A minimal 1D adaptation of the ARC AGI challenge
    inspired by MNIST1D methodology to enable rapid experimentation
    and rule discovery in ARC problems.
    """
    def __init__(self, grid_size=40, num_samples=1000, random_seed=42):
        """
        Initialize the ARC1D generator.
        
        Args:
            grid_size: Length of the 1D sequence (similar to MNIST1D's 40 points)
            num_samples: Number of samples to generate
            random_seed: Random seed for reproducibility
        """
        self.grid_size = grid_size
        self.num_samples = num_samples
        self.random_seed = random_seed
        
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        
        # Similar to MNIST1D, we'll create "rule templates" that we apply 
        # with variations to generate data
        self.rule_templates = [
            self.rule_move_right,
            self.rule_move_left,
            self.rule_recolor,
            self.rule_fill_gap,
            self.rule_mirror,
            self.rule_count_color,
            self.rule_flip
        ]
    
    def rule_move_right(self, sequence, params=None):
        """Move all colored blocks right by specified steps"""
        if params is None:
            params = {'steps': np.random.randint(1, 4)}
        
        steps = params.get('steps', 1)
        result = np.zeros_like(sequence)
        
        # Find all non-zero blocks and move them right
        for i in range(len(sequence)):
            if sequence[i] > 0:
                new_pos = min(i + steps, len(sequence) - 1)
                result[new_pos] = sequence[i]
                
        return result
    
    def rule_move_left(self, sequence, params=None):
        """Move all colored blocks left by specified steps"""
        if params is None:
            params = {'steps': np.random.randint(1, 4)}
        
        steps = params.get('steps', 1)
        result = np.zeros_like(sequence)
        
        # Find all non-zero blocks and move them left
        for i in range(len(sequence)):
            if sequence[i] > 0:
                new_pos = max(i - steps, 0)
                result[new_pos] = sequence[i]
                
        return result
    
    def rule_recolor(self, sequence, params=None):
        """Change colors of blocks based on a mapping"""
        if params is None:
            # Create a random mapping for colors 1-9
            color_map = {}
            for c in range(1, 10):
                color_map[c] = np.random.randint(1, 10)
            params = {'color_map': color_map}
        
        color_map = params.get('color_map', {})
        result = np.zeros_like(sequence)
        
        for i in range(len(sequence)):
            if sequence[i] > 0:
                result[i] = color_map.get(sequence[i], sequence[i])
                
        return result
    
    def rule_fill_gap(self, sequence, params=None):
        """Fill gaps between blocks of the same color"""
        result = sequence.copy()
        
        # Find continuous blocks and fill gaps
        for color in range(1, 10):
            # Find indices of this color
            color_indices = np.where(sequence == color)[0]
            
            if len(color_indices) >= 2:
                # Fill between min and max
                result[color_indices.min():color_indices.max()+1] = color
                
        return result
    
    def rule_mirror(self, sequence, params=None):
        """Mirror the pattern around a central point"""
        # Find the central point or use midpoint
        center = len(sequence) // 2
        
        # Compute mirrored sequence
        result = np.zeros_like(sequence)
        
        for i in range(len(sequence)):
            mirrored_i = 2 * center - i
            if 0 <= mirrored_i < len(sequence):
                result[i] = sequence[mirrored_i]
                
        return result
    
    def rule_count_color(self, sequence, params=None):
        """Count occurrences of colors and create a histogram"""
        result = np.zeros_like(sequence)
        
        # Count occurrences of each color (1-9)
        counts = {}
        for c in range(1, 10):
            counts[c] = np.sum(sequence == c)
        
        # Create a histogram representation at the start of the sequence
        idx = 0
        for c in range(1, 10):
            if counts[c] > 0:
                result[idx:idx+counts[c]] = c
                idx += counts[c]
                
        return result
    
    def rule_flip(self, sequence, params=None):
        """Flip/reverse the entire sequence"""
        return sequence[::-1]
    
    def generate_input_sequence(self):
        """Generate a random input sequence with colored blocks"""
        sequence = np.zeros(self.grid_size, dtype=int)
        
        # Number of colored blocks to add
        num_blocks = np.random.randint(1, max(2, self.grid_size // 4))
        
        # Add colored blocks
        for _ in range(num_blocks):
            # Block length
            block_len = np.random.randint(1, max(2, self.grid_size // 8))
            
            # Block position
            pos = np.random.randint(0, self.grid_size - block_len + 1)
            
            # Block color (1-9)
            color = np.random.randint(1, 10)
            
            # Place the block
            sequence[pos:pos+block_len] = color
            
        return sequence
    
    def generate_sample(self):
        """Generate a single (input, output) sample pair"""
        input_seq = self.generate_input_sequence()
        
        # Choose a random rule to apply
        rule_fn = random.choice(self.rule_templates)
        
        # Apply the rule to get output
        output_seq = rule_fn(input_seq)
        
        return {
            'input': input_seq.tolist(),
            'output': output_seq.tolist(),
            'rule': rule_fn.__name__
        }
    
    def generate_dataset(self):
        """Generate a full dataset with train/test split"""
        samples = [self.generate_sample() for _ in range(self.num_samples)]
        
        # Split into train/test (80/20)
        train_size = int(0.8 * self.num_samples)
        
        train_samples = samples[:train_size]
        test_samples = samples[train_size:]
        
        return {
            'train': train_samples,
            'test': test_samples
        }
    
    def visualize_sample(self, sample):
        """Visualize a sample as a colored 1D sequence"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 4))
        
        # Plot input
        ax1.imshow(np.array([sample['input']]), aspect='auto', cmap='tab10')
        ax1.set_title('Input')
        ax1.set_yticks([])
        
        # Plot output
        ax2.imshow(np.array([sample['output']]), aspect='auto', cmap='tab10')
        ax2.set_title('Output')
        ax2.set_yticks([])
        
        plt.tight_layout()
        plt.show()
    
    def save_dataset(self, filename):
        """Save the generated dataset to a JSON file"""
        dataset = self.generate_dataset()
        
        with open(filename, 'w') as f:
            json.dump(dataset, f)
        
        return dataset


class ARC1DDataset(Dataset):
    """PyTorch Dataset for ARC1D"""
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Convert to tensors
        input_seq = torch.tensor(sample['input'], dtype=torch.float)
        output_seq = torch.tensor(sample['output'], dtype=torch.float)
        
        # One-hot encode the input and output sequences
        input_one_hot = torch.zeros(10, len(input_seq))
        output_one_hot = torch.zeros(10, len(output_seq))
        
        for i, val in enumerate(input_seq):
            input_one_hot[int(val), i] = 1.0
            
        for i, val in enumerate(output_seq):
            output_one_hot[int(val), i] = 1.0
        
        return input_one_hot, output_one_hot


class SimpleConvNet(nn.Module):
    """Simple 1D CNN for ARC rule learning"""
    def __init__(self, grid_size=40, num_colors=10):
        super(SimpleConvNet, self).__init__()
        
        # Input shape: [batch_size, num_colors, grid_size]
        self.conv1 = nn.Conv1d(num_colors, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, num_colors, kernel_size=3, padding=1)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # x shape: [batch_size, num_colors, grid_size]
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x


class SimpleRNN(nn.Module):
    """Simple RNN for ARC rule learning"""
    def __init__(self, grid_size=40, num_colors=10):
        super(SimpleRNN, self).__init__()
        
        self.rnn = nn.GRU(num_colors, 64, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(128, num_colors)  # 128 = 64*2 for bidirectional
    
    def forward(self, x):
        # x shape: [batch_size, num_colors, grid_size]
        x = x.permute(0, 2, 1)  # [batch_size, grid_size, num_colors]
        
        outputs, _ = self.rnn(x)
        outputs = self.fc(outputs)
        
        outputs = outputs.permute(0, 2, 1)  # [batch_size, num_colors, grid_size]
        return outputs


class SimpleMLP(nn.Module):
    """Simple MLP for ARC rule learning"""
    def __init__(self, grid_size=40, num_colors=10):
        super(SimpleMLP, self).__init__()
        
        self.grid_size = grid_size
        self.num_colors = num_colors
        
        self.fc1 = nn.Linear(grid_size * num_colors, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, grid_size * num_colors)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # x shape: [batch_size, num_colors, grid_size]
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)  # Flatten
        
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        
        # Reshape back
        x = x.view(batch_size, self.num_colors, self.grid_size)
        return x


def train_model(model, train_loader, test_loader, num_epochs=10, device='cpu'):
    """Train a model on the ARC1D dataset"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.to(device)
    
    train_losses = []
    test_accuracies = []
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Reshape for cross entropy
            outputs = outputs.permute(0, 2, 1)  # [batch, grid_size, num_colors]
            targets = targets.permute(0, 2, 1).argmax(dim=2)  # [batch, grid_size]
            
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Evaluate on test set
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = model(inputs)
                
                # Convert to class predictions
                outputs = outputs.permute(0, 2, 1).argmax(dim=2)  # [batch, grid_size]
                targets = targets.permute(0, 2, 1).argmax(dim=2)  # [batch, grid_size]
                
                # Count correct predictions (entire sequences must match)
                correct += (outputs == targets).all(dim=1).sum().item()
                total += outputs.size(0)
        
        test_accuracy = 100.0 * correct / total
        test_accuracies.append(test_accuracy)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_train_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
    
    return train_losses, test_accuracies


def rule_discovery_analysis(model, test_dataset, device='cpu'):
    """Analyze which rules the model has learned"""
    model.eval()
    
    # Group test samples by rule
    rule_samples = {}
    for i, sample in enumerate(test_dataset.data):
        rule = sample['rule']
        if rule not in rule_samples:
            rule_samples[rule] = []
        rule_samples[rule].append(i)
    
    # Evaluate accuracy for each rule
    rule_accuracies = {}
    
    for rule, indices in rule_samples.items():
        correct = 0
        total = len(indices)
        
        for idx in indices:
            inputs, targets = test_dataset[idx]
            inputs = inputs.unsqueeze(0).to(device)  # Add batch dimension
            
            with torch.no_grad():
                outputs = model(inputs)
                
                # Convert to class predictions
                outputs = outputs.permute(0, 2, 1).argmax(dim=2)  # [batch, grid_size]
                targets = targets.permute(0, 2, 1).argmax(dim=2).unsqueeze(0)  # [batch, grid_size]
                
                if (outputs == targets).all():
                    correct += 1
        
        accuracy = 100.0 * correct / total
        rule_accuracies[rule] = accuracy
        print(f"Rule '{rule}': {accuracy:.2f}% accuracy ({correct}/{total})")
    
    return rule_accuracies


def main():
    # Create ARC1D dataset
    arc1d = ARC1D(grid_size=40, num_samples=4000, random_seed=42)
    dataset = arc1d.generate_dataset()
    
    # Save dataset to file
    arc1d.save_dataset('arc1d_dataset.json')
    
    # Create PyTorch datasets
    train_dataset = ARC1DDataset(dataset['train'])
    test_dataset = ARC1DDataset(dataset['test'])
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Visualize a few samples
    print("Visualizing samples:")
    for i in range(3):
        arc1d.visualize_sample(dataset['train'][i])
    
    # Train different model architectures
    print("\nTraining CNN model:")
    cnn_model = SimpleConvNet()
    cnn_losses, cnn_accuracies = train_model(cnn_model, train_loader, test_loader, num_epochs=10)
    
    print("\nTraining RNN model:")
    rnn_model = SimpleRNN()
    rnn_losses, rnn_accuracies = train_model(rnn_model, train_loader, test_loader, num_epochs=10)
    
    print("\nTraining MLP model:")
    mlp_model = SimpleMLP()
    mlp_losses, mlp_accuracies = train_model(mlp_model, train_loader, test_loader, num_epochs=10)
    
    # Compare model performances
    plt.figure(figsize=(10, 6))
    plt.plot(cnn_accuracies, label='CNN')
    plt.plot(rnn_accuracies, label='RNN')
    plt.plot(mlp_accuracies, label='MLP')
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Model Performance Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig('model_comparison.png')
    plt.close()
    
    # Analyze which rules each model learned best
    print("\nRule discovery analysis for CNN:")
    cnn_rule_acc = rule_discovery_analysis(cnn_model, test_dataset)
    
    print("\nRule discovery analysis for RNN:")
    rnn_rule_acc = rule_discovery_analysis(rnn_model, test_dataset)
    
    print("\nRule discovery analysis for MLP:")
    mlp_rule_acc = rule_discovery_analysis(mlp_model, test_dataset)
    
    # Plot rule discovery accuracies
    rules = list(cnn_rule_acc.keys())
    x = np.arange(len(rules))
    width = 0.25
    
    plt.figure(figsize=(12, 6))
    plt.bar(x - width, [cnn_rule_acc[rule] for rule in rules], width, label='CNN')
    plt.bar(x, [rnn_rule_acc[rule] for rule in rules], width, label='RNN')
    plt.bar(x + width, [mlp_rule_acc[rule] for rule in rules], width, label='MLP')
    
    plt.xlabel('Rule Type')
    plt.ylabel('Accuracy (%)')
    plt.title('Rule Discovery Performance by Model Architecture')
    plt.xticks(x, [r.replace('rule_', '') for r in rules], rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig('rule_discovery.png')
    plt.close()


if __name__ == "__main__":
    main() 