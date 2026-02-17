# Toy Finetuning Project

**Goal:** End-to-end finetuning pipeline on a tiny dataset (sentiment classification). Learn data loading, training loops, logging, and eval metrics on a single GPU.

## Skills Developed
- Config-driven training (YAML)
- Data loading pipelines (HuggingFace Datasets)
- Training loop with checkpointing
- Evaluation metrics (accuracy, F1)
- Reproducibility (seeds, logging)

## Quick Start (Local)

1. Create a venv and install deps:
```bash
cd experiments/toy_finetune
python -m venv .venv
.venv\Scripts\Activate.bat    # Windows
pip install -r requirements.txt
```

2. Review the config:
```bash
cat config.yaml
```

3. Run training:
```bash
python train.py --config config.yaml
```

4. Evaluate:
```bash
python eval.py --checkpoint runs/epoch_0.pt
```

## Quick Start (Google Colab) ⚡ RECOMMENDED

**Use this for fast GPU training without local setup!**

1. Download or copy the Colab notebook: [colab_training.ipynb](colab_training.ipynb)
2. Open in Google Colab: https://colab.research.google.com
3. Upload the notebook
4. Follow the cells to:
   - Clone your repo
   - Install dependencies
   - Train with GPU (16GB T4 or better)
   - Download checkpoints
5. Expected time: ~10 min training vs 1-2 hours on CPU

## Project Structure
- `config.yaml` — hyperparameters and paths
- `config_colab.yaml` — auto-generated for Colab (GPU-optimized)
- `data.py` — load, tokenize, and create dataloaders
- `train.py` — training loop with checkpointing
- `eval.py` — evaluation and metrics
- `colab_training.ipynb` — Google Colab notebook for GPU training
- `data/` — toy dataset (CSV)
- `runs/` — checkpoints and logs (auto-created)

## Expected Results
- **Baseline (no finetune):** ~60% accuracy
- **After 3 epochs:** ~85% accuracy
- **Training time (CPU):** 1–2 hrs
- **Training time (GPU):** 10–15 min

## Next Steps
- Log metrics to TensorBoard (add `torch.utils.tensorboard.SummaryWriter`)
- Add early stopping with validation loss
- Experiment with different model sizes (distilbert vs bert-base)
- Push Colab results back to GitHub with `!git push`

