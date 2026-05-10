# RACE Reading Comprehension Project

This folder contains the project dataset, source code, UI, notebooks, models, evaluation outputs, reports, and optional BERT backend.

## Project Structure

- `data/raw/` - RACE train/dev/test CSVs and raw RACE folder.
- `data/processed/` - generated feature-engineered CSVs from `src/preprocessing.py`.
- `models/model_a/neural/` - BERT checkpoints.
- `models/model_a/traditional/` - classical Model A, generator, and unsupervised artifacts.
- `models/model_b/traditional/` - Model B distractor and hint artifacts.
- `models/model_b/neural/` - reserved neural checkpoint folder for Model B extensions.
- `src/preprocessing.py` - rubric entrypoint for preprocessing.
- `src/model_a_train.py` - rubric entrypoint for Model A training targets.
- `src/model_b_train.py` - rubric entrypoint for Model B training.
- `src/inference.py` - rubric entrypoint for Model A, Model A generator, and Model B inference.
- `src/evaluate.py` - rubric entrypoint for evaluation targets.
- `src/evaluate_model_b_generation.py` - BLEU/ROUGE-L/METEOR evaluation for Model B distractor text quality.
- `ui/app.py` and `ui/components/` - Streamlit interface. Root `app.py` is a compatibility wrapper.
- `notebooks/EDA.ipynb` and `notebooks/experiments.ipynb` - EDA and experiment logs.
- `tests/` - smoke tests for inference artifacts and project layout.
- `report/final_report.pdf` - final report, with editable Markdown/DOCX copies in `report/`.
- `bts/` - behind-the-scenes experiment outputs, old checkpoints, scratch results, and versions.

## GitHub Demo Artifacts

The repository tracks the important runtime artifacts needed for the classical demo:
`data/raw/dev.csv`, `data/raw/test.csv`, the classical Model A `.joblib` files, the Model A
generator artifact, the Model A unsupervised artifact, and the Model B artifact. The very
large raw `train.csv`, extracted `data/raw/RACE/` folder, and BERT/neural checkpoints stay
ignored to keep the repository pushable under normal GitHub limits. To retrain everything
from scratch, restore the full RACE training data locally under `data/raw/`; to demo the
advanced BERT backend, restore or download the checkpoint under `models/model_a/neural/`.

## Current Best Model A Result

The strongest measured Model A is now the BERT multiple-choice run trained in two stages on 60,000 unique RACE questions:

```text
Checkpoint models/model_a/neural/bert_mc_30000q_plus_new30000q
Test BLEU    0.5431
Test ROUGE-L 0.6073
Test METEOR  0.5526
Test Exact   0.5172
```

The strongest classical supervised Model A remains:

```text
0.2 Logistic Regression + 0.8 XGBoost
```

Optimized test results:

```text
BLEU    0.4106
ROUGE-L 0.4922
METEOR  0.4438
Exact   0.3739
```

## Run Classical Inference

Run from this `Project/` folder:

```bash
../venv/bin/python src/model_a_inference.py \
  --backend classical \
  --article "Tom bought three red apples." \
  --question "What did Tom buy?" \
  --options '{"A":"three red apples","B":"a car","C":"a book","D":"a pen"}'
```

The classical backend has been retrained with question-relative option features. In addition
to absolute TF-IDF/cosine/overlap features, each option is compared against the other three
options for the same question using ranks, max flags, differences from group mean/max/min,
and within-question z-scores. This improved the held-out test exact diagnostic from
`0.3640` to `0.3739`.

## Run Dual Correct/Wrong Ranker Experiment

The dual-ranker experiment trains a second model to score likely wrong answers, then
combines:

```text
correct-answer ranking
wrong-answer ranking
within-question relative resolver
```

Run:

```bash
../venv/bin/python src/train_model_a_dual_ranker.py
```

Results:

| model | split | BLEU | ROUGE-L | METEOR | exact |
| ----- | ----- | ---- | ------- | ------ | ----- |
| Relative-feature correct ranker | dev | 0.3950 | 0.4892 | 0.4400 | 0.3653 |
| Relative-feature correct ranker | test | 0.4106 | 0.4922 | 0.4438 | 0.3739 |
| Dual correct/wrong ranker | dev | 0.3958 | 0.4919 | 0.4427 | 0.3685 |
| Dual correct/wrong ranker | test | 0.4098 | 0.4905 | 0.4426 | 0.3723 |

Finding: the dual ranker improved dev performance but did not beat the current classical
model on held-out test, so it remains an experimental option rather than the default. It can
be tested from CLI with:

```bash
../venv/bin/python src/model_a_inference.py \
  --backend classical \
  --article "Tom bought three red apples." \
  --question "What did Tom buy?" \
  --options '{"A":"three red apples","B":"a car","C":"a book","D":"a pen"}' \
  --use-dual-ranker
```

## Run Model A Generation

The generation side of Model A now uses a supervised hybrid pipeline trained on the RACE
training split. It extracts candidate answer spans, predicts the question type, applies a
matching WH/cloze template, and ranks the generated candidates before returning them. The
final display score blends the learned ranker with candidate quality and a small typed
question bonus so rare `who`, `where`, `when`, and `how_many` outputs are visible in the UI.

Train or refresh the generator artifacts:

```bash
../venv/bin/python src/train_model_a_generator.py \
  --output-dir models/model_a/traditional/generator \
  --max-classifier-train-rows 60000 \
  --max-ranker-train-rows 25000 \
  --max-ranker-eval-rows 1200 \
  --negatives-per-question 4
```

The generator saves:

```text
models/model_a/traditional/generator/generator_artifacts.joblib
models/model_a/traditional/generator/question_type_classifier_results.csv
models/model_a/traditional/generator/generation_ranker_results.csv
models/model_a/traditional/generator/question_type_label_distribution.csv
models/model_a/traditional/generator/generation_examples.csv
```

Current generator training results:

| component | split | metric 1 | metric 2 |
| --------- | ----- | -------- | -------- |
| Logistic Regression question-type classifier | dev | accuracy 0.3382 | macro-F1 0.1641 |
| Logistic Regression question-type classifier | test | accuracy 0.3405 | macro-F1 0.1593 |
| Linear SVM question-type classifier | dev | accuracy 0.4571 | macro-F1 0.1838 |
| Linear SVM question-type classifier | test | accuracy 0.4710 | macro-F1 0.1833 |
| Random Forest candidate ranker | dev | top-1 accuracy 0.9792 | MRR 0.9850 |
| Random Forest candidate ranker | test | top-1 accuracy 0.9833 | MRR 0.9870 |

Generate questions:

```bash
../venv/bin/python src/model_a_generation.py \
  --article "Tom bought three red apples from the market. He shared them with his sister." \
  --top-k 5 \
  --mode both
```

Generation modes:

- `ml`: supervised typed generation using the saved classifier and ranker.
- `cloze`: fill-in-the-blank fallback generation.
- `both`: combines typed WH questions and cloze fallback candidates.

## Run Model A Unsupervised Clustering

Section 4.2.2 of the assignment asks for an unsupervised or semi-supervised Model A
experiment. This project adds K-Means clustering with dendrogram analysis:

```bash
../venv/bin/python src/train_model_a_unsupervised.py \
  --output-dir models/model_a/traditional/unsupervised \
  --max-question-train-rows 60000 \
  --max-answer-train-questions 12000 \
  --max-eval-questions 2000 \
  --svd-components 70 \
  --min-k 3 \
  --max-k 12
```

Saved artifacts:

```text
models/model_a/traditional/unsupervised/model_a_unsupervised.joblib
models/model_a/traditional/unsupervised/question_cluster_dendrogram.png
models/model_a/traditional/unsupervised/answer_cluster_dendrogram.png
models/model_a/traditional/unsupervised/question_cluster_eval.csv
models/model_a/traditional/unsupervised/answer_cluster_eval.csv
models/model_a/traditional/unsupervised/question_cluster_profiles.csv
models/model_a/traditional/unsupervised/answer_cluster_profiles.csv
```

Current clustering results:

| component | best K | split | silhouette | purity / accuracy |
| --------- | ------ | ----- | ---------- | ----------------- |
| Question-type K-Means | 4 | train | 0.1898 | question-type purity 0.6247 |
| Question-type K-Means | 4 | dev | 0.2042 | question-type purity 0.6035 |
| Question-type K-Means | 4 | test | 0.1853 | question-type purity 0.6490 |
| Answer-option K-Means | 5 | train | 0.3179 | correctness purity 0.7500 |
| Answer-option K-Means | 5 | dev | 0.3171 | cluster-prior answer accuracy 0.2700 |
| Answer-option K-Means | 5 | test | 0.3183 | cluster-prior answer accuracy 0.2675 |

Finding: K-Means is useful for EDA and cluster metadata, especially for broad question
forms such as `which` and title-style `what` questions. It is not selected as the main
answer verifier because the cluster-prior answer accuracy is close to random/majority
behavior. The classical verifier therefore keeps LR + XGBoost as the final supervised
model, while exposing `--unsupervised-weight` as an optional experimental score blend:

```bash
../venv/bin/python src/model_a_inference.py \
  --backend classical \
  --article "Tom bought three red apples." \
  --question "What did Tom buy?" \
  --options '{"A":"three red apples","B":"a car","C":"a book","D":"a pen"}' \
  --unsupervised-weight 0.05
```

## Run Model B Distractor and Hint Generator

Model B now uses the question-type labels from Model A (`who`, `what`, `where`, `when`,
`how_many`, `which`, `why`, `cloze`) to guide distractor and hint generation. For example,
`how_many` questions favor numeric distractors, `when` questions favor dates/times, and
`who` questions favor names or named entities. Runtime generation now defaults to
passage-grounded distractors, using extracted passage phrases and action/prepositional
phrase patterns so the wrong options still look connected to the article. The final
distractor ranker also uses TF-IDF text features, answer type compatibility, lexical
overlap, one-hot/binary cosine similarity to the correct answer, character similarity,
passage frequency, candidate source, and a diversity filter.

Train or refresh Model B artifacts:

```bash
../venv/bin/python src/train_model_b.py \
  --max-train-rows 30000 \
  --max-eval-rows 2500 \
  --negatives-per-question 8 \
  --rf-max-train-rows 90000 \
  --output-dir models/model_b/traditional
```

Saved artifacts:

```text
models/model_b/traditional/model_b_artifacts.joblib
models/model_b/traditional/model_b_distractor_results.csv
models/model_b/traditional/model_b_hint_results.csv
models/model_b/traditional/model_b_generation_text_metrics.csv
models/model_b/traditional/model_b_examples.csv
models/model_b/traditional/model_b_training_summary.json
```

Current Model B results:

| component | model | split | metric 1 | metric 2 | metric 3 |
| --------- | ----- | ----- | -------- | -------- | -------- |
| Distractor ranker | Logistic Regression | dev | Precision@3 0.9995 | Recall@3 1.0000 | F1@3 0.9997 |
| Distractor ranker | Logistic Regression | test | Precision@3 0.9999 | Recall@3 1.0000 | F1@3 0.9999 |
| Distractor ranker | Random Forest | dev | Precision@3 0.9995 | Recall@3 1.0000 | F1@3 0.9997 |
| Distractor ranker | Random Forest | test | Precision@3 0.9999 | Recall@3 1.0000 | F1@3 0.9999 |
| Hint ranker | Logistic Regression | dev | Top-1 0.7232 | MRR 0.8362 | 2500 questions |
| Hint ranker | Logistic Regression | test | Top-1 0.7392 | MRR 0.8436 | 2500 questions |

Model B distractor text metrics against official RACE wrong-answer references:

| split | mode | BLEU | ROUGE-L | METEOR | exact text match |
| ----- | ---- | ---- | ------- | ------ | ---------------- |
| dev | option pool | 0.8688 | 0.9672 | 0.8477 | 0.9647 |
| test | option pool | 0.8998 | 0.9770 | 0.8729 | 0.9760 |
| dev | passage grounded | 0.0142 | 0.0955 | 0.0674 | 0.0195 |
| test | passage grounded | 0.0149 | 0.0737 | 0.0537 | 0.0081 |

The distractor metrics are option-pool ranking metrics: during supervised evaluation, the
candidate pool contains the official RACE wrong options, the correct option, and sampled
passage candidates. This measures whether the model can identify plausible distractors
from a labelled candidate pool. In runtime passage-only generation, the system uses
passage candidates only and relies more heavily on heuristic type, form, overlap,
frequency, and diversity rules.

Regenerate the Model B text metrics:

```bash
../venv/bin/python src/evaluate_model_b_generation.py --max-eval-rows 500
```

Generate distractors and hints:

```bash
../venv/bin/python src/model_b.py \
  --article "Tom bought three red apples from the market. He shared them with his sister." \
  --question "What did Tom buy?" \
  --correct-answer "three red apples"
```

## Optional BERT Backend

BERT dependencies are installed in `../venv`, and local checkpoints exist at:

```text
models/model_a/neural/bert_mc_10000q
models/model_a/neural/bert_mc_30000q
models/model_a/neural/bert_mc_30000q_plus_new30000q
```

The continued 30k + fresh 30k checkpoint is the strongest fine-tuned BERT multiple-choice checkpoint and is the default `bert` backend in `model_a_inference.py`.

## GPU Setup

The local environment has been configured for the available RTX GPU:

```text
GPU detected: NVIDIA GeForce RTX 3060 Laptop GPU
Driver/CUDA support: CUDA 12.4
PyTorch installed: torch 2.6.0+cu124
PyTorch CUDA available: True
```

The BERT backend uses `--device auto` by default, which selects CUDA when PyTorch can access the GPU and falls back to CPU otherwise.

Note: if a restricted IDE or sandbox hides `/dev/nvidia*`, PyTorch may report CUDA as unavailable from that shell. Run the commands from a normal terminal where `nvidia-smi` works to use the RTX 3060.

Fine-tune BERT multiple-choice:

```bash
../venv/bin/python src/train_bert_multiple_choice_model_a.py \
  --model-name models/model_a/neural/bert \
  --output-dir models/model_a/neural/bert_mc_30000q \
  --epochs 1 \
  --batch-size 4 \
  --gradient-accumulation-steps 2 \
  --max-length 256 \
  --max-train-questions 30000 \
  --max-dev-questions 3000 \
  --learning-rate 1.5e-5 \
  --eval-steps 750 \
  --logging-steps 100 \
  --early-stopping-patience 2 \
  --label-smoothing 0.05
```

Continue from the 30k checkpoint on a fresh, non-overlapping 30k sample:

```bash
../venv/bin/python src/train_bert_multiple_choice_model_a.py \
  --model-name models/model_a/neural/bert_mc_30000q \
  --output-dir models/model_a/neural/bert_mc_30000q_plus_new30000q \
  --epochs 1 \
  --batch-size 4 \
  --gradient-accumulation-steps 2 \
  --max-length 256 \
  --max-train-questions 30000 \
  --exclude-train-sample-size 30000 \
  --exclude-train-random-state 42 \
  --random-state 43 \
  --max-dev-questions 3000 \
  --learning-rate 1.0e-5 \
  --eval-steps 750 \
  --logging-steps 100 \
  --early-stopping-patience 2 \
  --label-smoothing 0.05
```

Run BERT inference:

```bash
../venv/bin/python src/model_a_inference.py \
  --backend bert \
  --bert-checkpoint models/model_a/neural/bert_mc_30000q_plus_new30000q \
  --device auto \
  --article "Tom bought three red apples." \
  --question "What did Tom buy?" \
  --options '{"A":"three red apples","B":"a car","C":"a book","D":"a pen"}'
```

Use `--device cuda` to force GPU execution and fail loudly if CUDA is unavailable.

Evaluate BERT:

```bash
../venv/bin/python src/evaluate_bert_multiple_choice_model_a.py \
  --checkpoint models/model_a/neural/bert_mc_30000q_plus_new30000q \
  --output-prefix bts/results/bert_mc_30000q_plus_new30000q \
  --max-length 256 \
  --batch-size 8 \
  --device auto
```

## UI

```bash
../venv/bin/streamlit run app.py
```

The Streamlit UI includes:

- Section 6.2-style screens: Article Input, Question & Answer Quiz View, Hint Panel, and Developer / Analytics Dashboard.
- A passage-first workflow: load a RACE sample, clear it for a custom passage, then press Submit to run Model A question generation plus Model B distractors/hints.
- Optional viewing/loading of the exact RACE question, correct option, and official options when the loaded passage has not been cleared or edited.
- A quiz answer selector with a Check button and colour-coded result.
- Progressive hints: Hint 1 vague, Hint 2 moderate/contextual, Hint 3 near-explicit, with Reveal Answer shown only after all hints are used.
- Developer analytics with Model A last-N accuracy, precision, recall, F1, confusion matrix, latency, and CSV export.
- Model B analytics with distractor/hint evaluation tables, last-N inference latency, and CSV export.
- A sidebar selector for `BERT multiple-choice` or `Classical LR + XGBoost`.
- Automatic loading of the final BERT checkpoint at `models/model_a/neural/bert_mc_30000q_plus_new30000q`.
- Automatic loading of the classical LR + XGBoost artifacts from `models/model_a/traditional`.
- Model A generation settings for `ML typed generation`, `Cloze fallback`, and default `Both` mode.
- Generated question details: question type, classifier confidence, ranker score, answer, generated question, source sentence, and unsupervised cluster metadata.
- Model B controls for passage-grounded distractors, optional hint refresh, and loading generated or RACE options into the quiz.
- RACE test-sample browser with Previous, Next, and Random controls.
- Optional automatic model run after changing the current test sample.
- Current-answer comparison for generated/RACE quiz options.
- Manual article, question, and A/B/C/D option input.
- Option score table, confidence chart, and downloadable JSON result.
- Dark-mode compatible prediction display.
