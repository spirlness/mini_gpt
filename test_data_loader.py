"""
Test script for the data loading module.
"""

import sys
import os

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from mini_gpt.data_loader import CharTokenizer, TextDataset, DataLoader


def test_char_tokenizer():
    """Test the CharTokenizer class."""
    print("=" * 50)
    print("Testing CharTokenizer")
    print("=" * 50)

    text = "Hello, world! This is a test."
    tokenizer = CharTokenizer(text)

    print(f"Original text: {text}")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Unique characters: {sorted(tokenizer.char_to_idx.keys())}")

    # Test encoding
    encoded = tokenizer.encode(text)
    print(f"Encoded: {encoded[:10]}...")  # Show first 10 tokens

    # Test decoding
    decoded = tokenizer.decode(encoded)
    print(f"Decoded: {decoded}")

    assert text == decoded, "Encoding/decoding should be reversible"
    print("✓ CharTokenizer test passed!\n")


def test_text_dataset():
    """Test the TextDataset class."""
    print("=" * 50)
    print("Testing TextDataset")
    print("=" * 50)

    data_path = "data/tiny.txt"
    if not os.path.exists(data_path):
        print(f"Warning: {data_path} not found. Skipping dataset test.")
        return

    dataset = TextDataset(data_path, seq_length=32)

    # Test batch generation
    batch_size = 4
    x, y = dataset.get_batch(batch_size, device="cpu")

    print(f"Input batch shape: {x.shape}")
    print(f"Target batch shape: {y.shape}")

    assert x.shape == (batch_size, 32), "Input shape should match batch_size x seq_length"
    assert y.shape == (batch_size, 32), "Target shape should match batch_size x seq_length"

    # Verify that target is shifted by 1
    print("\nSample sequence:")
    input_seq = dataset.tokenizer.decode(x[0].tolist())
    target_seq = dataset.tokenizer.decode(y[0].tolist())
    print(f"Input:  '{input_seq}'")
    print(f"Target: '{target_seq}'")

    print("✓ TextDataset test passed!\n")


def test_data_loader():
    """Test the DataLoader class."""
    print("=" * 50)
    print("Testing DataLoader")
    print("=" * 50)

    data_path = "data/tiny.txt"
    if not os.path.exists(data_path):
        print(f"Warning: {data_path} not found. Skipping data loader test.")
        return

    dataset = TextDataset(data_path, seq_length=64)
    loader = DataLoader(dataset, batch_size=8, device="cpu")

    # Test batch generation
    x, y = loader.get_batch()

    print(f"Batch shape: {x.shape}")
    print(f"Vocabulary size: {loader.vocab_size}")

    assert x.shape == (8, 64), "Batch should have correct shape"
    assert loader.vocab_size == dataset.tokenizer.vocab_size, "Vocab size should match"

    print("✓ DataLoader test passed!\n")


def main():
    """Run all tests."""
    print("\n" + "=" * 50)
    print("Running Data Loading Module Tests")
    print("=" * 50 + "\n")

    try:
        test_char_tokenizer()
        test_text_dataset()
        test_data_loader()

        print("=" * 50)
        print("All tests passed! ✓")
        print("=" * 50)

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
