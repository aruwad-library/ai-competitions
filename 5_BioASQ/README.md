# BioASQ Biomedical Yes/No Question Answering

Independent post-competition experiments combining biomedical text classification with evidence from provided snippets and PubMed documents.

## What this project implements

- Biomedical encoder experiments with BioBERT, BioELECTRA, and PubMedBERT.
- Query-only classification baselines and evidence-augmented yes/no classification.
- Retrieval and preprocessing of document text from provided PubMed URLs, including asynchronous fetching and HTML parsing.
- Training examples constructed from question/snippet and question/document pairs.
- Class-weighted loss and augmentation for imbalanced yes/no labels.
- LoRA adapters on a base model loaded in 4-bit precision, an 8-bit AdamW optimizer, Optuna tuning, and ensemble experiments.

## Files

- [Main experiment notebook](./main_final.ipynb): data exploration, query-only baselines, document collection, fine-tuning, and ensemble experiments.
- [Helper notebook](./bioasq_libs.ipynb): shared data preparation, metrics, and result logging.
- [Recorded experiment results](./bioasq_results.txt): historical internal training/validation logs.
- [Additional requirements](./requirements.txt): dependencies used alongside the repository-level requirements.

## Evaluation status

These are exploratory post-competition experiments. No official competition placement or estimated rank is claimed.

Existing runs augment validation examples for the negative class. Their scores therefore use a different protocol from the original question-level test batches and should not be compared directly with official leaderboard scores. The historical logs are preserved; a headline test score requires a documented run, data split, and question-level evaluation protocol.

## Quantization and fine-tuning

`load_in_4bit=True` quantizes base-model weights. LoRA supplies trainable adapter parameters while the quantized base weights remain frozen. `adamw_bnb_8bit` refers to optimizer-state precision, not 8-bit model weights. See the [Transformers bitsandbytes documentation](https://huggingface.co/docs/transformers/quantization/bitsandbytes).

## Reproduction

Start with `main_final.ipynb` and review its imports and local dataset paths. The notebooks reference local datasets, retrieved document text, and checkpoints that are not included in this repository. Model downloads and PubMed retrieval require network access; quantized fine-tuning requires a compatible bitsandbytes environment. The `main_bck*.ipynb` files preserve earlier exploratory versions.
