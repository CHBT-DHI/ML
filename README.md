## ML — Small experiments and training scripts

This repository contains small machine learning experiments, data preprocessing utilities, and training scripts for XGBoost and Transformer-based models.

## Contents

- `dataset.py` — data loading and preprocessing helpers.
- `model.py` — model definitions and utilities.
- `train_transformer.py` — training script for a transformer-based model.
- `train_xgboost2.py` — training script for an XGBoost model.
- `utils.py` — miscellaneous helper functions used across scripts.
- `requirements.txt` — Python dependencies used by the project.
- `data/` — dataset files and preprocessing outputs:
  - `raw/` — original raw data (e.g. `raw.csv`, `raw.txt`).
  - `raw_normalized/` — normalized data and normalization stats (`normalized.csv`, `normalization_stats.csv`).
- `notebooks/` — Jupyter notebooks for exploration and preprocessing (e.g. `preprocess_data.ipynb`, `XGBOOST2.ipynb`).

## Quick start

1. Create a virtual environment and activate it (recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Prepare data

- Place your raw dataset files in `data/raw/`. If you already have normalized files, they should be in `data/raw_normalized/`.
- Optionally run the preprocessing notebook `notebooks/preprocess_data.ipynb` to explore and normalize the data.

4. Run training scripts

- Train XGBoost model (defaults will be used by the script):

```bash
python train_xgboost2.py
```

- Train transformer model (defaults will be used by the script):

```bash
python train_transformer.py
```

Notes
- The training scripts assume the data is available in `data/raw/` or `data/raw_normalized/` depending on the preprocessing pipeline used. Check the scripts for configurable paths or command-line arguments.
- Use the notebooks in `notebooks/` for interactive exploration or to reproduce preprocessing steps.

## Project conventions / contract

- Python: tested with Python 3.8+ (use a virtual environment).
- Inputs: CSV / text files in `data/raw/`.
- Outputs: trained model artifacts and logs (check the scripts for output paths).
- Error modes: scripts will raise exceptions for missing data files or invalid formats. Inspect the stack trace and verify the input file paths.

## Helpful tips and edge cases

- If data files are large, ensure enough disk space and memory are available.
- For reproducibility, fix random seeds inside training scripts (check `train_*.py`).
- If you get dependency issues, verify `pip` is using the virtual environment's interpreter.

## Contributing

1. Fork the repository and create a feature branch.
2. Add tests or a small notebook demonstrating new behavior where appropriate.
3. Open a PR describing the change.

## License

This repository does not include a license file. Add a `LICENSE` if you plan to publish it under a specific license.

## Contact

If you have questions, open an issue in the repository.

---

Small, focused repo — keep changes minimal and document new training flags or model outputs in `README.md` as you go.
