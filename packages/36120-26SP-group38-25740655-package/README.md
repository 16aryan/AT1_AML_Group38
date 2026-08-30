# nba-longevity-player-25740655

Player-level feature construction, seed-bagged XGBoost and AUPRC evaluation for
predicting sustained NBA careers from college basketball statistics.

Built for UTS 36120 Advanced Machine Learning Application, AT1, group 38.

## Why this package exists

The competition scores **one prediction per player**, but the raw data holds
**one row per player-season**. Which design wins depends on the model. Measured
under identical folds, features and hyperparameters:

| Representation | XGBoost | Logistic Regression |
|---|---|---|
| Player rows (`build_player_table`) | **0.5024** | 0.4430 |
| Season rows, predictions averaged | 0.4801 | **0.4717** |
| Season rows, predictions collapsed by `max` | 0.4654 | 0.4430 |

Aggregation is a **trade, not a free improvement**: it exchanges ~41,000
training rows for ~17,900 while tripling the feature count. A tree ensemble
exploits the richer description and gains; a linear model cannot and loses.

Independently, `max` is a poor collapsing rule: a four-season player gets four
draws at the top against a one-season player's one, while the underlying
positive rate rises only from 3.1% to 3.7% across that range. Use the mean.

The package encodes both findings in tested functions rather than leaving them
to be retyped per notebook.

## Installation

```bash
pip install --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple \
    "nba-longevity-player-25740655[xgboost,plots]"
```

`--extra-index-url` is required because TestPyPI does not mirror pandas, numpy,
scikit-learn or xgboost.

Extras: `xgboost` (the model), `plots` (matplotlib curves), `dev` (pytest, build,
twine). The core install deliberately avoids xgboost so that feature building,
evaluation and submission assembly stay usable and testable without it.

On macOS, xgboost needs OpenMP: `brew install libomp`.

## Usage

```python
import nba_longevity_player as nlp

data = nlp.load_competition_data()          # resolves NBA_DATA_DIR or common paths
train_season = nlp.prepare_season_rows(data.train)
test_season = nlp.prepare_season_rows(data.test)
numeric = nlp.numeric_feature_columns(train_season, test_season)

# Standardise within each season cohort before aggregating (era robustness).
train_season = nlp.within_season_zscore(train_season, numeric)
test_season = nlp.within_season_zscore(test_season, numeric)

player_train = nlp.build_player_table(train_season, numeric, include_target=True)
player_test = nlp.build_player_table(test_season, numeric)

X = player_train.drop(columns=["pid", "label"])
y = player_train["label"].to_numpy()
X_test = player_test.drop(columns=["pid"])
X, X_test = nlp.drop_nuisance_features(X, X_test)
X, X_test = nlp.align_categoricals(X, X_test)

model = nlp.SeedBaggedXGB().fit(X, y)
scores = model.predict_rank_score(X_test)

submission = nlp.build_submission(player_test["pid"], scores)
nlp.save_submission(submission, "submission.csv", sample_submission=data.sample_submission)
```

## Modules

| Module | Contents |
|---|---|
| `config` | Target, identifiers, tuned hyperparameters, aggregation set, seeds, nuisance features |
| `data` | `load_competition_data`, `resolve_data_dir`, `audit_player_structure` |
| `features` | `prepare_season_rows`, `build_player_table`, `within_season_zscore`, `drop_nuisance_features`, `align_categoricals`, leakage check |
| `modeling` | `SeedBaggedXGB`, `build_xgb_classifier`, importance, persistence |
| `evaluation` | AUPRC metrics, out-of-fold scoring, `era_shifted_scores`, `adversarial_validation`, threshold tools |
| `submission` | `build_submission`, `validate_submission`, `save_submission` |

## Design decisions worth knowing

**Text columns are parsed, not dropped.** `ht` (`'6-7'`), `dob` and `num` arrive
as text, so a numeric-only column selection silently discards height, age and
jersey number along with every categorical. `prepare_season_rows` recovers them.

**No scaling anywhere.** Tree splits depend only on rank ordering, so any
monotonic rescaling leaves the fitted model unchanged. A scaler would add a
preprocessing step and an opportunity for train/serve drift with no benefit.

**Missing values are kept.** XGBoost learns a default split direction per node.
Here a statistic is often missing precisely because a player played too few
minutes for it to be computed, so the absence is itself evidence.

**`team` is excluded.** Its 353 levels reduced cross-validated AUPRC from 0.4814
to 0.4545. Excluded on evidence, not principle.

**Absolute `year` never becomes a feature.** Training data covers 2008–2016 and
the test set 2020–2023, with no overlap. Only `career_span_years` survives.

**Ranks, not probabilities, by default.** AUPRC reads only ordering.
`ranks_to_unit_interval` maps scores into `(0, 1)` preserving order exactly.
Use `predict_proba_mean` when you need a probability.

## Era robustness (v0.2.0)

Training data covers **2008-2016**; the Kaggle test set covers **2020-2023**.
`adversarial_validation` separates the two at **0.94 AUC** - close to trivially.
A model fitted on raw features can therefore partly learn *which period* a row
belongs to rather than the outcome.

Random folds drawn inside the training years contain no such shift, so they
cannot measure resistance to one. They ranked a tuned configuration ~0.045 above
a histogram-boosting baseline; the real leaderboard gap was 0.0017.

**Select with `era_shifted_scores`**, which fits on earlier cohorts and scores
later ones across several cutoffs. Cutoffs are a paired comparison against the
same rival, not independent samples, so the per-cutoff win record matters more
than the mean: a variant that never loses beats one with a better average that
sometimes does.

Two tools address the shift directly:

- `within_season_zscore` standardises each statistic inside its own season
  cohort. It scores **worse** on random folds (0.4564 vs 0.4814) and **better**
  under era-shifted validation (0.4362 vs 0.4294, winning at every cutoff).
  That disagreement is the point - random folds are the wrong instrument.
- `drop_nuisance_features` removes columns that mark the era without predicting
  the outcome. `jersey_number` ranked as the strongest era marker in the whole
  feature set and has no causal link to NBA longevity.

A hyperparameter search of 18 candidates on the z-scored representation, scored
under era-shifted validation, produced **no** configuration beating the incumbent
at every cutoff, so the tuned settings were retained unchanged.

## Development

```bash
pip install -e ".[xgboost,plots,dev]"
pytest
```

112 tests, 94% statement coverage. The suite runs without xgboost installed —
`test_modeling.py` skips itself via `importorskip`.

```bash
python -m build
twine upload --repository testpypi dist/*
```

## License

MIT.
