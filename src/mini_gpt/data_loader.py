"""
Data loading module for mini GPT.

This module provides:
- CharTokenizer: Character-level tokenization
- TextDataset: Dataset class for text data
- DataLoader: Batch generation for training
"""

import torch
import random


class CharTokenizer:
    """Character-level tokenizer for text data."""

    def __init__(self, text):
        """
        Initialize the tokenizer with a text corpus.

        Args:
            text: String containing the full text corpus
        """
        # Get unique characters and create mappings
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(chars)}

    def encode(self, text):
        """
        Encode text to indices.

        Args:
            text: String to encode

        Returns:
            List of integer indices

        Raises:
            ValueError: If text contains characters not in vocabulary
        """
        try:
            return [self.char_to_idx[ch] for ch in text]
        except KeyError as e:
            raise ValueError(
                f"Character '{e.args[0]}' not found in vocabulary. "
                f"Available characters: {sorted(self.char_to_idx.keys())}"
            )

    def decode(self, indices):
        """
        Decode indices back to text.

        Args:
            indices: List of integer indices

        Returns:
            Decoded string
        """
        return "".join([self.idx_to_char[i] for i in indices])


class TextDataset:
    """Dataset class for handling text data."""

    def __init__(self, data_path, seq_length=128):
        """
        Initialize the dataset.

        Args:
            data_path: Path to the text file
            seq_length: Length of each sequence (context window)
        """
        with open(data_path, "r", encoding="utf-8") as f:
            self.text = f.read()

        if len(self.text) <= seq_length:
            raise ValueError(
                f"Dataset is too small ({len(self.text)} chars). "
                f"Must be larger than seq_length ({seq_length})."
            )

        self.seq_length = seq_length
        self.tokenizer = CharTokenizer(self.text)
        self.data = self.tokenizer.encode(self.text)

        print(f"Loaded text with {len(self.text)} characters")
        print(f"Vocabulary size: {self.tokenizer.vocab_size}")
        print(f"Total tokens: {len(self.data)}")

    def get_batch(self, batch_size, device="cpu"):
        """
        Generate a random batch of data.

        Args:
            batch_size: Number of sequences in the batch
            device: Device to place tensors on ('cpu' or 'cuda')

        Returns:
            Tuple of (input_batch, target_batch) as torch tensors
        """
        # Randomly sample starting indices
        max_start_idx = len(self.data) - self.seq_length - 1
        start_indices = [random.randint(0, max_start_idx) for _ in range(batch_size)]

        # Build input and target sequences
        input_seqs = []
        target_seqs = []

        for start_idx in start_indices:
            input_seq = self.data[start_idx : start_idx + self.seq_length]
            target_seq = self.data[start_idx + 1 : start_idx + self.seq_length + 1]
            input_seqs.append(input_seq)
            target_seqs.append(target_seq)

        # Convert to tensors
        x = torch.tensor(input_seqs, dtype=torch.long, device=device)
        y = torch.tensor(target_seqs, dtype=torch.long, device=device)

        return x, y


class DataLoader:
    """
    Data loader wrapper that provides batches for training.
    """

    def __init__(self, dataset, batch_size, device="cpu"):
        """
        Initialize the data loader.

        Args:
            dataset: TextDataset instance
            batch_size: Batch size for training
            device: Device to place tensors on
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device

    def get_batch(self):
        """
        Get a batch of data.

        Returns:
            Tuple of (input_batch, target_batch)
        """
        return self.dataset.get_batch(self.batch_size, self.device)

    @property
    def vocab_size(self):
        """Get the vocabulary size."""
        return self.dataset.tokenizer.vocab_size
