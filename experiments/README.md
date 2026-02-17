Experiments folder

Progressive learning projects to develop production ML skills.

Active Projects

### 1. toy_finetune (Complete)
**Goal:** End-to-end sentiment classification finetuning on SST-2 (small dataset, single GPU/CPU).

**Skills:** Config-driven training, data pipelines, checkpointing, eval metrics, reproducibility.

**Quickstart:**
```bash
cd toy_finetune
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
python train.py --config config.yaml
```

**Expected results:** 60% → 85% accuracy over 3 epochs.

See [toy_finetune/README.md](toy_finetune/README.md) for details.

### 2. production-lite (TODO)
Goal: Production pipeline with logging, checkpointing, val early-stop, and FastAPI server.

### 3. research (TODO)
Goal: Ablation studies, quantization, distillation, and write-up results.

Each experiment includes:
- `README.md` — goals, skills, and quickstart
- `config.yaml` — hyperparameters
- `data.py` — data loading and preprocessing
- `src/` or `train.py`, `eval.py` — core scripts
- `tests/` — unit tests
