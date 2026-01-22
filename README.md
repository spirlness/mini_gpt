# Mini GPT

A simplified implementation of GPT (Generative Pre-trained Transformer) for educational purposes.

## Features

- **Data Loading Module**: Character-level data loading and batching
  - `CharTokenizer`: Character-level tokenization with encode/decode
  - `TextDataset`: Dataset class for text data management
  - `DataLoader`: Batch generation for training

## Installation

```bash
pip install -e .
```

## Data Loading Module Usage

```python
from mini_gpt import TextDataset, DataLoader

# Create dataset
dataset = TextDataset('data/tiny.txt', seq_length=128)

# Create data loader
loader = DataLoader(dataset, batch_size=32, device='cpu')

# Get batches for training
x, y = loader.get_batch()  # Returns (input, target) tensors
```

## Training

```bash
python -m mini_gpt.train --data_path data/tiny.txt --batch_size 32 --lr 1e-3
```

## Testing

Run the data loader tests:

```bash
python test_data_loader.py
```

## Project Structure

```
mini_gpt/
├── src/mini_gpt/
│   ├── __init__.py
│   ├── data_loader.py  # Data loading module
│   └── train.py        # Training script
├── data/
│   └── tiny.txt        # Sample training data
├── test_data_loader.py # Data loader tests
└── pyproject.toml      # Project configuration
```

## License

Educational use only.
