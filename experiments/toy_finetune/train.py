import argparse
import os
import torch
import yaml
from torch.optim import Adam
from transformers import AutoModelForSequenceClassification
from tqdm import tqdm
from data import load_and_preprocess


def validate_config(config):
    """Validate and convert config types."""
    required_keys = [
        "model_name", "task", "max_seq_length", "batch_size", 
        "learning_rate", "num_epochs", "checkpoint_dir"
    ]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")
    
    # Convert types
    config["max_seq_length"] = int(config["max_seq_length"])
    config["batch_size"] = int(config["batch_size"])
    config["learning_rate"] = float(config["learning_rate"])
    config["num_epochs"] = int(config["num_epochs"])
    config["warmup_steps"] = int(config.get("warmup_steps", 0))
    config["log_every"] = int(config.get("log_every", 50))
    config["eval_every"] = int(config.get("eval_every", 200))
    config["seed"] = int(config.get("seed", 42))
    
    return config


def train(config):
    """Main training loop."""
    
    # Setup
    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config["seed"])
    
    print(f"Using device: {device}")
    
    # Load data
    train_loader, val_loader = load_and_preprocess(
        model_name=config["model_name"],
        task=config["task"],
        max_seq_length=config["max_seq_length"],
        batch_size=config["batch_size"],
    )
    
    # Load model
    print(f"Loading model: {config['model_name']}")
    model = AutoModelForSequenceClassification.from_pretrained(
        config["model_name"], num_labels=2
    )
    model.to(device)
    
    # Optimizer and loss
    optimizer = Adam(model.parameters(), lr=config["learning_rate"])
    criterion = torch.nn.CrossEntropyLoss()
    
    # Training loop
    global_step = 0
    for epoch in range(config["num_epochs"]):
        model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config['num_epochs']}")
        for batch in pbar:
            # Move to device
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")
            
            # Forward
            outputs = model(**batch)
            loss = criterion(outputs.logits, labels)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            global_step += 1
            total_loss += loss.item()
            
            # Log
            if global_step % config["log_every"] == 0:
                avg_loss = total_loss / config["log_every"]
                print(f"[Step {global_step}] Loss: {avg_loss:.4f}")
                total_loss = 0
            
            # Eval
            if global_step % config["eval_every"] == 0:
                val_acc = evaluate(model, val_loader, device)
                print(f"[Step {global_step}] Val Accuracy: {val_acc:.4f}")
        
        # Save checkpoint
        ckpt_path = os.path.join(config["checkpoint_dir"], f"epoch_{epoch}.pt")
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")


def evaluate(model, loader, device):
    """Evaluate model accuracy on loader."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")
            outputs = model(**batch)
            preds = outputs.logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += len(labels)
    return correct / total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file path")
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    config = validate_config(config)
    train(config)
