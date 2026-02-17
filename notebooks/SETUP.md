Setup (Windows)

1. Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1   # or use Activate.bat for cmd
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Run the sample script:

```powershell
python src/train_sample.py
```

Notes
- If you prefer conda, create an environment with `conda create -n llm-lab python=3.10` and activate it.
- GPU usage requires an appropriate `torch` wheel for CUDA; consult PyTorch install guide.
