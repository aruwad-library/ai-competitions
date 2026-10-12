# Competition

- [LLM Classification Finetuning](https://www.kaggle.com/competitions/llm-classification-finetuning/overview)

- Task: 26,707/26,708 train/test samples, with 35 predictive features (`respondent_id` excluded)

- Participants: 9,223 participants / 2,443 competitors (as of 2026-10-09)

- Period: 2026-10-10 ~ 2026-10-12

# Results (as of 2026-10-09)
| Method | Log-loss | Rank |
|---|---|---|
| `google/gemma-2-2b-it`/head-only | (bad) | (bad) |
| `google/gemma-2-2b-it`/lora | 0.9722 | 41/248 (16.53%) |
| `google/gemma-2-9b-it`/lora | (wip) | (wip) |



<img src="./final_leaderboard.png" width="500">
<br><br>


# Highlights

- Identified missing survey responses as predictive information; preserving missingness delivered the largest preprocessing gain.

- Controlled experiments favored keeping all original predictors over feature pruning, with no consistent benefit from complex imputation or class weighting.

- Selected preprocessing and XGBoost–CatBoost voting weights separately for H1N1 and seasonal vaccination, rather than assuming equal model contributions.

# Technical Summary

- Input: $X_{\mathrm{train}}$ with shape $(26{,}707, 35)$
  - 23 numerically coded + 12 categorical features
  - Binary, ordinal, and categorical survey responses

- Output: $y \in \{0,1\}^{26{,}707 \times 2}$
  - `h1n1_vaccine` / `seasonal_vaccine`
  - Submission: one positive-class probability per target

- Evaluation: mean ROC-AUC across the two targets

- Preprocessing
  - Numerical — XGBoost
    - Missingness indicators
    - `SimpleImputer(strategy='median')` versus native missing-value handling
    - `StandardScaler()`
    - Opinion recoding with separate “don't know” indicators
  - Categorical — XGBoost
    - Employment-aware missing categories; observed answers preserved
    - Explicit missing-category tokens instead of mode imputation
    - `OneHotEncoder(handle_unknown='ignore')`
  - CatBoost
    - Native categorical features and numerical missing-value handling

- Validation and HPO
  - Five-fold `KFold`, with preprocessing fitted within each training fold
  - Optuna: `TPESampler` + fold-level `MedianPruner`
  - Early stopping for MLP and CatBoost
  - Out-of-fold predictions for preprocessing and ensemble selection

- EDA
  - Label dependence, missingness patterns, response encoding, and train/test shift
  - Held-out individual/grouped permutation importance and subgroup error analysis
  - Controlled comparisons of imputation, class weights, scaling, and feature subsets
  - Retained all original predictors; no outlier removal or class weighting

- Model
  - Base candidates: `XGBClassifier` / `CatBoostClassifier` / two-hidden-layer PyTorch MLP
  - Final model: target-specific weighted soft voting of XGBoost and CatBoost
  - XGBoost preprocessing variant and voting weights selected separately for each target
  - MLP excluded from the final submission