import argparse
import torch
import yaml
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset


def evaluate_checkpoint(checkpoint_path, model_name, batch_size, max_seq_length):
    """Load checkpoint and evaluate on validation set."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model and checkpoint
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Load tokenizer and dataset
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = load_dataset("glue", "sst2")["validation"]
    
    # Tokenize
    def tokenize_fn(examples):
        return tokenizer(
            examples["sentence"],
            truncation=True,
            max_length=max_seq_length,
            padding="max_length",
        )
    
    dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["sentence", "idx"])
    dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    
    # Evaluate
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=batch_size)
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("label")
            outputs = model(**batch)
            preds = outputs.logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += len(labels)
    
    accuracy = correct / total
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Validation Accuracy: {accuracy:.4f}")
    return accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    evaluate_checkpoint(args.checkpoint, args.model_name, config["batch_size"], config["max_seq_length"])
