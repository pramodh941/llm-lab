import os
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


def load_and_preprocess(model_name, task, max_seq_length, batch_size):
    """Load SST-2 sentiment dataset, tokenize, and return dataloaders."""
    
    # Load dataset
    print(f"Loading {task} dataset...")
    dataset = load_dataset("glue", "sst2")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Tokenize function
    def tokenize_fn(examples):
        return tokenizer(
            examples["sentence"],
            truncation=True,
            max_length=max_seq_length,
            padding="max_length",
        )
    
    # Apply tokenization
    print("Tokenizing...")
    dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["sentence", "idx"])
    dataset = dataset.rename_column("label", "labels")
    dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    
    # Create dataloaders
    train_loader = DataLoader(dataset["train"], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset["validation"], batch_size=batch_size)
    
    print(f"Train samples: {len(dataset['train'])}, Val samples: {len(dataset['validation'])}")
    return train_loader, val_loader


if __name__ == "__main__":
    # Quick test
    train_loader, val_loader = load_and_preprocess(
        model_name="distilbert-base-uncased",
        task="sentiment",
        max_seq_length=128,
        batch_size=16
    )
    batch = next(iter(train_loader))
    print(f"Batch keys: {batch.keys()}")
    print(f"Input shape: {batch['input_ids'].shape}")
