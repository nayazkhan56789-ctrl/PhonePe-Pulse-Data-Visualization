# Medical Abstract Sentence Classifier using PubMed RCT - Final Report

*Literature-mining aid. Not medical advice.*

## 1. Task
Given a sentence from a randomized-controlled-trial abstract, predict its rhetorical role:
BACKGROUND, OBJECTIVE, METHODS, RESULTS or CONCLUSIONS. Applications: highlighting abstracts for faster
skimming, biomedical search that jumps to "Methods" or "Results", and a building block for summarisation/retrieval.

## 2. Data
PubMed 20k RCT (Dernoncourt & Lee, 2017): 15,000 / 2,500 / 2,500 abstracts (train / dev / test), 180,040 / 30,212 / 30,135 sentences.
Class balance (train): METHODS 33.0%, RESULTS 32.2%, CONCLUSIONS 15.1%, BACKGROUND 12.1%, OBJECTIVE 7.7% (`results/eda_label_and_length.png`).
Because of the imbalance, **macro-F1** is the primary metric. Median sentence length is 24 tokens (95th percentile 58).
Public abstracts contain no patient records, which is why this is a safer classroom dataset than clinical notes
(no consent, de-identification or privacy-compliance burden - although some abstracts may still be copyrighted).

## 3. Method
**Preprocessing:** lowercase, regex tokenisation, numbers -> `<num>`; vocabularies/vectorisers fitted on train only; train/dev/test kept separate.
**Baseline:** TF-IDF (1-2-grams, 207k features) + Logistic Regression (class-balanced). C in {1,3,10} chosen on dev -> C=1.
**Neural model:** Text-CNN (128-d embeddings, kernel sizes 2-5 x 128 filters, max-pooling, dropout 0.5, class-weighted loss, Adam),
5 epochs, best epoch (3) chosen on dev macro-F1. Test evaluated once. Single run, seed 42.

## 4. Results (test set, 30,135 sentences)
| Model | Accuracy | Macro-F1 | Weighted-F1 | (Dev macro-F1) |
|---|---|---|---|---|
| TF-IDF + Logistic Regression | **0.8265** | **0.7692** | **0.8279** | 0.7764 |
| Text-CNN | 0.8148 | 0.7589 | 0.8158 | 0.7636 |

Per-class F1 (test):
| Class | TF-IDF+LR | Text-CNN |
|---|---|---|
| METHODS | 0.905 | 0.896 |
| RESULTS | 0.886 | 0.870 |
| CONCLUSIONS | 0.776 | 0.753 |
| BACKGROUND | 0.661 | 0.656 |
| OBJECTIVE | 0.618 | 0.619 |

Full precision/recall tables: `results/*_classification_report.txt`; confusion matrices: `results/*_confusion_matrix.png`.

## 5. Analysis
- **Methods and Results are easier than Background and Objective.** METHODS/RESULTS reach F1 ~0.87-0.91 for both models; BACKGROUND/OBJECTIVE only ~0.62-0.66.
  Methods and Results sentences contain distinctive vocabulary (patients, randomized, mg, p-values, "was measured", numbers).
- **The main confusion is BACKGROUND <-> OBJECTIVE** (TF-IDF: 666 background sentences predicted as objective and 652 the other way).
  Both are introductory, general-language sentences; the same sentence ("We hypothesised that...") can plausibly be either.
  Other notable errors: RESULTS -> METHODS (717) and RESULTS -> CONCLUSIONS (642), and CONCLUSIONS -> BACKGROUND (367): results and conclusions both interpret findings.
- **The neural model did not beat the baseline here.** The Text-CNN was ~1 point lower in accuracy and macro-F1. Its dev score peaked at epoch 3 while training loss kept falling
  (overfitting: 0.76 -> 0.757 dev). With only 180k short sentences and strongly lexical cues, bag-of-n-grams is a very competitive baseline. This is a single run
  with one seed, so a ~1-point gap should not be over-interpreted; we did not tune the CNN extensively.
- **Demo:** `04_demo_colorcode.py` colour-codes a new abstract (`results/demo_abstract_tfidf.html`, `demo_abstract_cnn.html`).
  On a hand-written 8-sentence sample abstract both models labelled every sentence as intended - a sanity check, not an evaluation.

## 6. Limitations
- **Sentences are classified independently.** Position in the abstract is a very strong cue (mean relative position: OBJECTIVE 0.07, BACKGROUND 0.23, METHODS 0.35, RESULTS 0.66, CONCLUSIONS 0.93),
  and neighbouring sentences matter. Sequence models (e.g. a CRF or hierarchical LSTM over sentences) typically do better; we deliberately kept the task per-sentence as the brief describes.
- Single dataset (RCT abstracts, structured-abstract style); performance on other study types or unstructured abstracts is unknown.
- One run per model, no confidence intervals; PubMed 20k only (PubMed 200k and Transformer models such as BioBERT were not tried).
- Labels come from section headings in the source abstracts and contain noise (e.g. headings that merge "Background and Objective").
- The simple regex sentence splitter used in the demo can mis-split sentences with unusual abbreviations.
- This is a literature-skimming tool and must not be used to make clinical decisions.

## 7. Future work
Add sentence-position / context features or a sequence model over the whole abstract; try a pretrained biomedical Transformer; scale to PubMed 200k; evaluate on non-RCT abstracts.
