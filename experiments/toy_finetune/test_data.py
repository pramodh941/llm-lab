import pytest
from data import load_and_preprocess


def test_data_loading():
    """Test that data loading works and returns correct shapes."""
    try:
        # This will download SST-2 if not cached, so it may take time
        train_loader, val_loader = load_and_preprocess(
            model_name="distilbert-base-uncased",
            task="sentiment",
            max_seq_length=128,
            batch_size=8
        )
        
        # Check that we got dataloaders
        assert train_loader is not None
        assert val_loader is not None
        
        # Check first batch
        batch = next(iter(train_loader))
        assert "input_ids" in batch
        assert "attention_mask" in batch
        assert "labels" in batch
        assert batch["input_ids"].shape[1] == 128  # seq_length
        
        print("✓ Data loading test passed")
    except Exception as e:
        print(f"✗ Data loading test failed: {e}")
        raise


if __name__ == "__main__":
    test_data_loading()
