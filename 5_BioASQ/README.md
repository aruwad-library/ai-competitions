# BioASQ Biomedical Yes/No Question Answering

Experiments combining biomedical text classification with evidence from provided snippets and PubMed documents.

## What this project implements

- Query-only classification baselines and evidence-augmented yes/no classification.
- Retrieval and preprocessing of document text from PubMed URLs, including asynchronous fetching and HTML parsing.
- Training examples constructed from question/snippet and question/document pairs.
- Class-weighted loss for the imbalanced yes/no labels.
- LoRA adapters on a base model loaded in 4-bit precision, with an 8-bit optimizer, Optuna tuning, and ensemble experiments.

The retrieval component prepares evidence for classification; the notebook does not implement free-form answer generation.

## Files

- [Main experiment notebook](./main_final.ipynb): data exploration, query-only baselines, document retrieval, fine-tuning, and ensemble experiments.
- [Helper notebook](./bioasq_libs.ipynb): shared data preparation, metrics, and result logging.
- [Recorded experiment results](./bioasq_results.txt): historical internal training/validation logs.
- [Additional requirements](./requirements.txt): dependencies used alongside the repository-level requirements.

## Evaluation and precision

The recorded results are internal experiments, not an official BioASQ leaderboard placement. Consult the corresponding notebook cells and logs for each model and evaluation setup; the experiments have not been rerun for this documentation update.

`load_in_4bit=True` quantizes base-model weights. LoRA supplies trainable adapter parameters while the quantized base weights remain frozen. `adamw_bnb_8bit` refers to optimizer-state precision, not 8-bit model weights. See the [Transformers bitsandbytes documentation](https://huggingface.co/docs/transformers/quantization/bitsandbytes).

## Running

Start with `main_final.ipynb` and review its imports and local dataset paths. Prepare the expected BioASQ data and saved intermediate datasets before running dependent cells. Model downloads and PubMed retrieval require network access; quantized fine-tuning requires a compatible bitsandbytes environment. The `main_bck*.ipynb` files preserve earlier exploratory versions.
