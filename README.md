# Applied ML and NLP Projects

Experiments in tabular prediction, human preference classification, and biomedical question answering.

This repository includes scored competition submissions and independent post-competition studies. Start with **Arena** for the most recent end-to-end training, evaluation, and inference work.

## Projects

| Project | Main work | Results and evidence |
|---|---|---|
| [Arena](./3_Arena/) | Gemma 2 LoRA fine-tuning, prompt-grouped validation, response-preserving token budgets, and offline inference | **Public log loss 0.90805; rank 28/250**, recorded on 2026-10-13 |
| [H1N1](./2_H1N1/) | Missingness-aware preprocessing, target-specific model selection, and weighted ensembles | **Public mean ROC-AUC 0.8637; rank 96/2,448**, recorded on 2026-10-09 |
| [House Prices](./1_housing/) | Regression, feature processing, data pruning, and ensemble comparisons | **Public log-RMSE 0.1180; rank 153/3,931**, recorded on 2026-10-09 |
| [Google QUEST](./4_Google_QUEST/) | Transformer-based prediction with a metadata bypass and staged fine-tuning | Post-competition study; **best internal-validation mean Spearman correlation 0.3813** |
| [BioASQ](./5_BioASQ/) | Biomedical encoders, document collection, LoRA, and class-weighted training | Post-competition study; exploratory internal evaluation |

## Start with Arena

- [Training and evaluation](./3_Arena/3_gemma2_9b_full_lora.ipynb)
- [Single-pass submission](./3_Arena/3_gemma2_9b_submission_no_validation.ipynb)
- [Results, implementation details, and limitations](./3_Arena/README.md)

The task predicts human preferences using a classification loss. Training used LoRA with a BF16 backbone on one RTX 5090. The two-GPU setup was used for inference, not distributed training.

## Evaluation notes

Leaderboard ranks are dated public snapshots, not final-placement or medal claims. Each project's README links to the recorded evidence.

Google QUEST uses an 80/20 split of the released training data; the validation split was also used for checkpoint selection. BioASQ's existing runs use class-dependent validation augmentation. These internal results are not converted into official leaderboard ranks.

## Running the notebooks

Review each notebook's setup, data paths, model requirements, and hardware assumptions before running it. Some datasets, checkpoints, and local run artifacts are not included in this repository.

## Related repository

[AI/ML foundations and implementation notes](https://github.com/aruwad-library/ai-summary)
