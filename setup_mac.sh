#!/usr/bin/env bash
# One-command setup for macOS (Apple Silicon). Safe to re-run.
#
#   bash setup_mac.sh
#
# Creates .venv, installs everything, registers a Jupyter kernel for VS Code,
# and verifies the whole stack imports.

set -euo pipefail
cd "$(dirname "$0")"
EXP="$(pwd)"
PKG="$EXP/../36120-26SP-group38-25740655-package"

say() { printf "\n\033[1m==> %s\033[0m\n" "$1"; }

say "1/6  Checking Python"
PY=""
for c in python3.12 python3.11 python3.13 python3; do
  if command -v "$c" >/dev/null 2>&1; then PY="$c"; break; fi
done
[ -n "$PY" ] || { echo "No python3 found. Install with: brew install python@3.12"; exit 1; }
echo "     using $PY ($($PY --version 2>&1))"

say "2/6  Checking OpenMP (XGBoost and LightGBM need it on macOS)"
if brew list libomp >/dev/null 2>&1; then
  echo "     libomp already installed"
else
  echo "     installing libomp..."
  brew install libomp
fi

say "3/6  Creating virtual environment at .venv"
if [ -d .venv ]; then
  echo "     .venv already exists, reusing it"
else
  "$PY" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install -q --upgrade pip

say "4/6  Installing dependencies (a few minutes on first run)"
# pandas 2.2.3 segfaults on pd.to_datetime with numpy >= 2.5, so it is excluded.
python -m pip install -q \
  "pandas>=2.0,!=2.2.3" "numpy>=1.23" "scikit-learn>=1.3" "scipy>=1.10" \
  "xgboost>=2.0" "lightgbm>=4.0" "matplotlib>=3.6" "seaborn>=0.12" "joblib>=1.2" \
  jupyter ipykernel nbconvert

say "5/6  Installing the individual package (editable) + registering the kernel"
if [ -f "$PKG/pyproject.toml" ]; then
  python -m pip install -q -e "$PKG[xgboost,lightgbm,plots,dev]"
  echo "     installed nba_longevity_player from the sibling package repository"
else
  python -m pip install -q --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple \
    "nba-longevity-player-25740655[xgboost,lightgbm,plots]"
  echo "     installed nba_longevity_player from TestPyPI"
fi
python -m ipykernel install --user --name at1-nba --display-name "Python (AT1 NBA)" >/dev/null
echo "     kernel registered as: Python (AT1 NBA)"

say "6/6  Verifying"
python - <<'PYCHECK'
import importlib, sys
mods = ["pandas", "numpy", "sklearn", "scipy", "xgboost", "lightgbm", "matplotlib", "joblib"]
for m in mods:
    mod = importlib.import_module(m)
    print(f"     {m:12s} {getattr(mod, '__version__', '?')}")
import nba_longevity_player as nlp
print(f"     {'package':12s} nba_longevity_player {nlp.__version__}")
for name in ("SeedBaggedXGB", "SeedBaggedLGBM", "blend_rank_scores",
             "build_player_table", "within_season_zscore", "build_submission"):
    assert hasattr(nlp, name), f"missing {name}"
print("     package API OK")
PYCHECK

DATA="$EXP/data/raw/train.csv"
if [ -f "$DATA" ]; then
  echo "     competition data found in data/raw/"
else
  printf "\n\033[33m     WARNING: %s not found.\033[0m\n" "$DATA"
  echo "     Put train.csv, test.csv, sample_submission.csv, metadata.csv there."
fi

cat <<'DONE'

------------------------------------------------------------------
Setup complete.

In VS Code:
  1. File > Open Folder... and choose this repository folder.
  2. Install the "Python" and "Jupyter" extensions (Extensions sidebar).
  3. Open a notebook in 36120-26SP-group38-experiments/notebooks/
  4. Top-right "Select Kernel" > Python Environments > Python (AT1 NBA)
  5. Run All.

From the terminal instead:
  source .venv/bin/activate
  make test        # 128 unit tests
  make notebooks   # execute all three experiments
------------------------------------------------------------------
DONE
