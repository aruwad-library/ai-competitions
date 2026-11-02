# Machine Learning Competition Projects

Projects covering task-specific modeling, validation, and inference with competition datasets. Start with Arena for the most recent end-to-end example.

| Project | Focus | Results and evidence |
|---|---|---|
| [Human Preference Prediction with Gemma 2](./3_Arena/README.md) | Structured prompt/response inputs, LoRA fine-tuning, grouped validation, and offline multi-GPU inference | Public leaderboard snapshot and local validation are reported separately |
| [BioASQ Biomedical Yes/No QA](./5_BioASQ/README.md) | Biomedical text classification, evidence retrieval, class imbalance, and 4-bit LoRA experiments | Internal experiment logs and notebooks |
| [Google QUEST Q&A Labeling](./4_Google_QUEST/README.md) | Multi-label prediction, metadata bypass, and staged fine-tuning | Post-competition internal validation |
| [H1N1 Vaccine Prediction](./2_H1N1/README.md) | Tabular classification | See the project README and notebooks |
| [Housing Price Prediction](./1_housing/README.md) | Tabular regression | See the project README and notebooks |

Each project documents its own evaluation setting. An internal validation score is not an official competition placement, and a public leaderboard snapshot is not a final rank.

## Running the notebooks

The projects use Python/Jupyter and project-specific datasets, model checkpoints, and hardware. Review the imports and file paths in each notebook and install the dependencies listed in the root or project requirements before running it. Datasets and trained model artifacts are not bundled with these notebooks.
