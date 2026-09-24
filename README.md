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

## Progress

- [x] Week 1 — Environment, dataset, EDA
- [ ] Week 2 — Text preprocessing & baseline model (TF-IDF + Logistic Regression)
- [ ] Week 3 — Evaluation & improvement
- [ ] Week 4 — Streamlit app & project completion
