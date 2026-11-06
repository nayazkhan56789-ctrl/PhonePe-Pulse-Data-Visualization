# Medical Abstract Sentence Classifier using PubMed RCT

Classifies every sentence of a randomized-controlled-trial abstract into its rhetorical role:
**BACKGROUND, OBJECTIVE, METHODS, RESULTS, CONCLUSIONS**. A literature-skimming aid - **not medical advice**.

## Results at a glance (test set, 30,135 sentences)
| Model | Accuracy | Macro-F1 | Weighted-F1 |
|---|---|---|---|
| TF-IDF + Logistic Regression (baseline) | 0.8265 | 0.7692 | 0.8279 |
| Text-CNN | 0.8148 | 0.7589 | 0.8158 |

METHODS and RESULTS are the easiest classes (F1 ~0.87-0.91); BACKGROUND and OBJECTIVE are the hardest (F1 ~0.62-0.66) and are mostly confused with each other.
Full analysis and limitations: `REPORT.md`. Colour-coded demo examples: `results/demo_abstract_tfidf.html`, `results/demo_abstract_cnn.html`.

## Data (already included)
`data/PubMed_20k_RCT/{train,dev,test}.txt` - from https://github.com/Franck-Dernoncourt/pubmed-rct
(Dernoncourt & Lee, *PubMed 200k RCT: a Dataset for Sequential Sentence Classification in Medical Abstracts*, IJCNLP 2017).
Original dataset README and licence/copyright note: `data/DATASET_README.md` (some abstracts may be protected by copyright; public, no patient records).

| split | abstracts | sentences |
|---|---|---|
| train | 15,000 | 180,040 |
| dev | 2,500 | 30,212 |
| test | 2,500 | 30,135 |

Class balance (train): METHODS 33%, RESULTS 32%, CONCLUSIONS 15%, BACKGROUND 12%, OBJECTIVE 8% -> imbalanced, so **macro-F1** is the headline metric.
To scale up, use PubMed 200k from the same GitHub repo (train file is compressed there) and point `DATA_DIR` in `src/common.py` at it.

## Run (matches the 5-day timeline)
```bash
pip install -r requirements.txt
cd src
python 01_explore.py                     # Day 1: sizes, label balance, sentence lengths
python 02_baseline_tfidf_lr.py           # Day 2: TF-IDF + Logistic Regression  (~5 min on CPU)
python 03_neural_model.py --model cnn --epochs 5      # Day 3: Text-CNN (~20 min on 1 CPU core, much faster on GPU)
python 03_neural_model.py --model bilstm --epochs 4   # optional BiLSTM
python 04_demo_colorcode.py              # Day 4: colour-coded demo -> results/demo_abstract.html (the shipped examples are demo_abstract_tfidf.html / demo_abstract_cnn.html)
python 04_demo_colorcode.py --file my_abstract.txt --model cnn
```

## Preprocessing (documented choices)
1. Lowercase; regex tokenisation into words / numbers / punctuation.
2. Every number becomes the token `<num>` (the *presence* of numbers signals METHODS/RESULTS; the exact value does not).
3. TF-IDF: unigrams+bigrams, `min_df=3`, sublinear TF, fitted on **train only**.
4. Neural: vocabulary from train only (min freq 2, max 40k), sequences padded/truncated to 64 tokens (~97% of sentences), label encoding in fixed order.
5. Class imbalance handled with class weights; best epoch / C chosen on **dev**; **test used once** at the end.
6. Each sentence is classified **independently** (no position-in-abstract feature) - see limitations.

## Outputs
- `models/` - `tfidf_vectorizer.joblib`, `tfidf_logreg.joblib`, `neural_cnn.pt`, `neural_cnn_vocab.json`
- `results/` - per model: `*_metrics.json`, `*_classification_report.txt`, `*_confusion_matrix.png/csv`; plus EDA figures and `demo_abstract.html`
- `REPORT.md` - final report (task, method, results, limitations)
