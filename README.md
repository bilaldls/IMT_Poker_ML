# Poker ML Toolkit

This repository contains a structured Python package dedicated to the study of
poker win-rate prediction. The legacy scripts have been replaced by a clean
`src/` layout with clear responsibilities for data preparation, feature
engineering and model evaluation.

## Project structure

```
├── data/
│   ├── raw/                # Datasets generated via Monte Carlo simulations
│   └── processed/          # Encoded or cleaned datasets ready for modelling
├── models/                 # Recommended location for saving trained models
├── src/
│   └── poker_ml/
│       ├── data/           # Dataset generation, cleaning, encoding, visualisation
│       ├── models/         # Classification & regression evaluation utilities
│       └── utils/          # Shared helpers (paths, directory creation, ...)
├── pyproject.toml          # Project metadata and runtime dependencies
└── README.md
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use `.venv\\Scripts\\activate`
pip install -e .
```

Optional deep-learning components (TensorFlow based models) can be installed
with:

```bash
pip install -e .[deep-learning]
```

## Usage

### Dataset generation

```python
from pathlib import Path
from poker_ml.data.generation import SimulationConfig, generate_dataset

config = SimulationConfig(
    simulations=1_000,
    dataset_size=5_000,
    output_path=Path("data/raw/poker_winrates.csv"),
)
path = generate_dataset(config)
print(f"Dataset saved to {path}")
```

### Cleaning raw win-rate files

```python
from pathlib import Path
from poker_ml.data.cleaning import CleaningConfig, clean_win_rates, load_dataset, save_dataset

config = CleaningConfig(input_path=Path("data/raw/monte_carlo/poker_winrates_mc_0100.csv"))
df = load_dataset(config.resolved_input_path(), header=None)
clean_df = clean_win_rates(df, config)
save_dataset(clean_df, config.resolved_output_path(), header=False)
```

### Encoding cards

```python
from pathlib import Path
from poker_ml.data.encoding import EncodingConfig, encode_dataset

config = EncodingConfig(
    input_path=Path("data/raw/monte_carlo/poker_winrates_mc_0100.csv"),
    output_path=Path("data/processed/poker_encoded.csv"),
    encoding="full_one_hot",
)
encoded_path = encode_dataset(config)
```

### Model evaluation

```python
import pandas as pd
from poker_ml.models.classification import (
    discretize_probabilities,
    evaluate_multiclass_logistic_regression,
)

encoded = pd.read_csv("data/processed/poker_encoded.csv", header=None)
X = encoded.iloc[:, :119]
y = discretize_probabilities(encoded.iloc[:, 120], boundaries=(0.4, 0.7))
summary = evaluate_multiclass_logistic_regression(X, y, use_smote=True)
print("Accuracy:", summary.mean_accuracy)
```

Refer to the modules under `src/poker_ml/` for additional utilities, such as
random forest classifiers or regression estimators.
