# E-commerce Review Sentiment Analyzer

Web application that takes e-commerce product reviews and predicts whether the sentiment is Positive, Neutral, or Negative.

## Project Goal

Build a machine learning pipeline that classifies customer product reviews by sentiment, and expose it through a simple web interface (Streamlit) where a user can paste a review and get an instant prediction.

## Dataset

- **Source:** Kaggle — Datafiniti's *Consumer Reviews of Amazon Products*
- **Link:** <add https://www.kaggle.com/datasets/datafiniti/consumer-reviews-of-amazon-products>
- Real Amazon product reviews (mostly electronics: Kindles, Fire tablets, batteries, chargers), each with a 1–5 star rating and review text.

## Sentiment Labeling

Ratings are bucketed into 3 classes, since the dataset ships with star ratings rather than ready-made sentiment labels:

| Rating | Sentiment |
|--------|-----------|
| 1–2    | Negative  |
| 3      | Neutral   |
| 4–5    | Positive  |

## Tech Stack

- Python, pandas, scikit-learn
- Jupyter notebooks for exploration
- Streamlit (Week 4 — web app)
- Git / GitHub for version control

## Setup

```bash
git clone https://github.com/<your-username>/ecommerce-sentiment-analyzer.git
cd ecommerce-sentiment-analyzer
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Download the dataset from Kaggle (link above) and place the CSV in a `data/` folder (not tracked by git).

## Project Structure

```
ecommerce-sentiment-analyzer/
├── data/                    # raw dataset (gitignored)
├── notebooks/
│   └── 01_exploration.ipynb # Week 1: loading, EDA, train/test split
    └── 02_exploration.ipynb # Week 1: loading, EDA, train/test split
├── requirements.txt
├── .gitignore
└── README.md
```


## Week 1 — Environment, Dataset & EDA

**What I did:**
- Set up a Python virtual environment and recorded dependencies in `requirements.txt`.
- Loaded the dataset and inspected its structure (shape, columns, dtypes).
- Derived a 3-class `sentiment` label from `reviews.rating`.
- Checked class balance, missing values, and duplicate reviews.
- Inspected review length distribution and read sample reviews from each class.
- Created a stratified 80/20 train/test split.

**What I found:**
- Dataset shape: `<34660, 21>`
- Class balance: Positive `<93.33%>`, Neutral `<4.33%>`, Negative `<2.35%>`
- Missing values dropped: `<34>`
- Duplicate reviews dropped: `<0>`
- Review length: min `<1>`, max `<1858>`, average `<30.24>` words

<img width="1459" height="675" alt="Screenshot 2026-09-27 223913" src="https://github.com/user-attachments/assets/57e57a01-876a-4311-8fe5-7c3d25942205" />

<img width="475" height="112" alt="Screenshot 2026-09-27 223931" src="https://github.com/user-attachments/assets/a4ab8bfd-977b-4fd1-b77d-49b5844a23ee" />


## Week 2 — Text Preprocessing & Baseline Model

**What I did:**
- Cleaned review text (lowercased, stripped punctuation, removed stopwords while preserving negation words).
- Converted text to TF-IDF features (unigrams + bigrams, fit on training data only).
- Trained a Logistic Regression baseline with `class_weight='balanced'` to account for class imbalance.

**Baseline result:**
- Accuracy: `<0.841>`

<img width="920" height="93" alt="Screenshot 2026-09-27 223952" src="https://github.com/user-attachments/assets/bec775a5-38a9-4f33-945a-75afcd7a0e7b" />


## Week 3 — Evaluation & Improvement

**What I did:**
- Generated precision, recall, and F1-score per class with `classification_report`.
- Generated and interpreted a confusion matrix.
- Inspected misclassified examples to identify failure patterns.
- Trained a Linear SVM as a second approach and compared it against the Logistic Regression baseline.
- Documented limitations surfaced during error analysis.

**Results:**

| Model | Accuracy | Macro-F1 | Weighted-F1 |
|-------|----------|----------|-------------|
| Logistic Regression (baseline) | 0.84 | 0.49 | 0.87 |
| Linear SVM | 0.91 | 0.53 | 0.91 |

Per-class F1 (Negative / Neutral / Positive):
- Logistic Regression: 0.34 / 0.22 / 0.92
- Linear SVM: 0.40 / 0.23 / 0.96

<img width="655" height="439" alt="Screenshot 2026-10-05 174732" src="https://github.com/user-attachments/assets/d5c95dd9-9fa9-48d6-94b9-cd782777f903" />


<img width="456" height="357" alt="Screenshot 2026-10-05 174748" src="https://github.com/user-attachments/assets/308655c4-c728-40bb-92e4-aaa5dede5067" />

**Model comparison and decision:**

Logistic Regression scored a macro-F1 of 0.49, and Linear SVM scored 0.53 — a 4-point gap. The per-class breakdown explains why: with `class_weight='balanced'`, Logistic Regression aggressively over-predicts the minority classes (Negative recall 0.54, Neutral recall 0.42), but at a steep precision cost — only 25% of its "Negative" predictions and 15% of its "Neutral" predictions are actually correct. Linear SVM trades some of that recall for much higher precision (0.39, 0.21) and a large gain on the dominant Positive class (F1 0.96 vs 0.92), driving its overall accuracy to 91% vs 84%.

Carrying forward **Linear SVM**. Its macro- and weighted-F1 are both meaningfully higher, and more importantly, Logistic Regression's low precision on Negative/Neutral (15–25%) means most of what it labels "Negative" or "Neutral" would be wrong if shown to a user. SVM doesn't provide `predict_proba()` out of the box, so I'll wrap it in `CalibratedClassifierCV` in Week 4 to get calibrated confidence scores without reverting to the weaker minority-class precision.

**Limitations:**
- **Ambiguous reviews:** Reviews mixing praise and criticism (e.g. "cheap but broke fast") are hard to classify into a single class.
- **Sarcasm:** TF-IDF has no way to detect tone, so sarcastic reviews are frequently misclassified.
- **Neutral class:** Weakest-performing class for both models — 3-star reviews are inherently ambiguous, and there are fewer training examples.
- **Small Negative sample:** Only 162 Negative examples in the test set — too small to draw strong conclusions from.
- **Domain:** Trained on Amazon electronics reviews; sentiment vocabulary may not transfer well to other product categories (e.g. clothing, groceries).
- **Short reviews:** Very short text gives TF-IDF little signal to work with.


## Progress

- [x] Week 1 — Environment, dataset, EDA
- [x] Week 2 — Text preprocessing & baseline model (TF-IDF + Logistic Regression)
- [x] Week 3 — Evaluation & improvement
- [ ] Week 4 — Streamlit app & project completion
