# Contributing to fraud-shield

PRs welcome. Here's how to get started.

## Setup

```bash
git clone https://github.com/iamadhitya1/fraud-shield
cd fraud-shield
pip install -e .
pip install scikit-learn pandas numpy
```

Download the [Kaggle Credit Card Fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) to test with real data.

## Project structure

```
fraudshield/
  __init__.py    # exports FraudDetector
  detector.py    # main FraudDetector class
  result.py      # PredictionResult dataclass
```

## What's in scope

- Bug fixes
- Additional ML models beyond Random Forest (XGBoost, LightGBM, etc.)
- Better handling of imbalanced datasets (SMOTE, etc.)
- Batch prediction performance improvements
- New evaluation metrics
- Documentation fixes

## Guidelines

- Keep `FraudDetector` as the single public class — don't expand the surface area without good reason
- Don't break the `.train()` / `.predict()` / `.predict_batch()` API
- Keep dependencies minimal — only add a new dependency if it's truly necessary
- One feature or fix per PR

## Submitting a PR

1. Fork the repo
2. Create a branch: `git checkout -b feat/your-feature-name`
3. Make your change
4. Open a PR against `main` with a clear title and description of what changed and why

---

MIT © 2025 M Adhitya · [Rewrite Labs](https://rewritelabs.vercel.app)
