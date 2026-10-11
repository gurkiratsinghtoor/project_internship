# 📰 Fake News Generator & Predictor (NLP + TF-IDF)

Internship project: a machine-learning pipeline that classifies news text as **FAKE** or **REAL**,
with a small synthetic-news generator for demos/stress-testing and a Streamlit web app.

## How it works
1. **Preprocessing** – lowercase, remove URLs, numbers, punctuation, agency tags (e.g. "(Reuters)") to avoid data leakage.
2. **TF-IDF** – converts text to numeric vectors (unigrams + bigrams, English stop-words removed, `max_df=0.7`, `min_df=2`, sublinear TF).
3. **Models compared** – Logistic Regression, Passive-Aggressive (SGD, pa1), Multinomial Naive Bayes. Best F1 is saved.
4. **Evaluation** – accuracy, precision, recall, F1, confusion matrix, top influential words.
5. **Explainability** – each prediction shows the terms that influenced it most.
6. **Generator** – template-based, *fictional* labelled articles (invented names/places) used as demo data and to test the model.

## Setup
```bash
pip install -r requirements.txt
```
## Dataset (recommended for real results)
Download Kaggle **"Fake and Real News Dataset"** (ISOT) and place `Fake.csv` and `True.csv` in `data/`.
Without them the project trains on synthetic demo data (will show ~100%, which is *not* a real benchmark).

## Run
```bash
python src/train.py                       # trains, writes models/ and reports/
python src/predict.py "paste news text"   # command-line prediction
streamlit run app.py                      # web app (Predictor | Generator | Report)
```

## Structure
```
app.py              Streamlit UI
src/preprocess.py   text cleaning
src/generator.py    synthetic news generator
src/train.py        training + evaluation
src/predict.py      inference + explanations
data/  models/  reports/
```

## Limitations (mention in your presentation)
- TF-IDF learns *writing style / vocabulary*, not factual truth; it cannot fact-check.
- Trained on one dataset (US politics 2016-17) -> may not generalise to other topics or years.
- Future work: BERT/transformers, fact-check API integration, multilingual (Hindi/Punjabi) support.

## API-based generation (Claude)
```bash
export ANTHROPIC_API_KEY="your-key"     # Windows: set ANTHROPIC_API_KEY=your-key
python src/api_generator.py --n 10 --topic "city transport"
```
Generates fictional labelled articles, saves `data/synthetic_api.csv`, and reports how well the
predictor classifies them. The same generator powers the app's *Synthetic News Generator* tab.
Without a key it falls back to the template generator.
