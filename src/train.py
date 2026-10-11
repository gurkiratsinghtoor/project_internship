"""
Train fake news classifier: TF-IDF + (Logistic Regression | Passive Aggressive | Naive Bayes).

Data:  put Kaggle "Fake and Real News" files at data/Fake.csv and data/True.csv
       (columns: title,text,subject,date).  If missing, falls back to synthetic demo data
       from generator.py.
"""
import os
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, \
    confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split

from preprocess import clean_text
from generator import generate

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA, MODELS, REPORTS = (os.path.join(ROOT, d) for d in ("data", "models", "reports"))


def load_data() -> pd.DataFrame:
    fake_p, true_p = os.path.join(DATA, "Fake.csv"), os.path.join(DATA, "True.csv")
    if os.path.exists(fake_p) and os.path.exists(true_p):
        print("Using Kaggle dataset")
        fake, true = pd.read_csv(fake_p), pd.read_csv(true_p)
        fake["label"], true["label"] = 1, 0
        df = pd.concat([fake, true], ignore_index=True)
    else:
        print("Kaggle files not found -> using synthetic demo data (see README)")
        df = generate(600)
    df["content"] = (df["title"].fillna("") + " " + df["text"].fillna("")).map(clean_text)
    df = df[df["content"].str.len() > 20].drop_duplicates("content")
    return df[["content", "label"]]


def main():
    os.makedirs(MODELS, exist_ok=True)
    os.makedirs(REPORTS, exist_ok=True)
    df = load_data()
    print("Samples:", len(df), "| fake:", int(df.label.sum()), "| real:", int((df.label == 0).sum()))

    X_train, X_test, y_train, y_test = train_test_split(
        df["content"], df["label"], test_size=0.2, random_state=42, stratify=df["label"])

    tfidf = TfidfVectorizer(stop_words="english", max_df=0.7, min_df=2, ngram_range=(1, 2), sublinear_tf=True)
    Xtr, Xte = tfidf.fit_transform(X_train), tfidf.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Passive Aggressive": SGDClassifier(loss="hinge", penalty=None, learning_rate="pa1", eta0=1.0, max_iter=1000, random_state=42),
        "Multinomial NB": MultinomialNB(),
    }
    results, fitted = [], {}
    for name, m in models.items():
        m.fit(Xtr, y_train)
        p = m.predict(Xte)
        results.append({"model": name, "accuracy": accuracy_score(y_test, p),
                        "precision": precision_score(y_test, p), "recall": recall_score(y_test, p),
                        "f1": f1_score(y_test, p)})
        fitted[name] = m
    res = pd.DataFrame(results).sort_values("f1", ascending=False, kind="stable")
    print("\n", res.round(4).to_string(index=False))
    res.round(4).to_csv(os.path.join(REPORTS, "model_comparison.csv"), index=False)

    best_name = res.iloc[0]["model"]
    best = fitted[best_name]
    pred = best.predict(Xte)
    print(f"\nBest model: {best_name}\n", classification_report(y_test, pred, target_names=["Real", "Fake"]))

    ConfusionMatrixDisplay(confusion_matrix(y_test, pred), display_labels=["Real", "Fake"]).plot(cmap="Blues")
    plt.title(f"Confusion Matrix - {best_name}")
    plt.savefig(os.path.join(REPORTS, "confusion_matrix.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # most influential words for linear models
    if hasattr(best, "coef_"):
        words = tfidf.get_feature_names_out()
        top = pd.DataFrame({"word": words, "weight": best.coef_[0]}).sort_values("weight")
        top.tail(15).to_csv(os.path.join(REPORTS, "top_fake_words.csv"), index=False)
        top.head(15).to_csv(os.path.join(REPORTS, "top_real_words.csv"), index=False)
        fig, ax = plt.subplots(figsize=(8, 6))
        sel = pd.concat([top.head(10), top.tail(10)])
        ax.barh(sel["word"], sel["weight"], color=["tab:green"] * 10 + ["tab:red"] * 10)
        ax.set_title("Most influential TF-IDF terms (red = fake, green = real)")
        plt.savefig(os.path.join(REPORTS, "top_words.png"), dpi=150, bbox_inches="tight")
        plt.close()

    joblib.dump({"tfidf": tfidf, "model": best, "name": best_name}, os.path.join(MODELS, "fake_news_model.joblib"))
    print("Saved model to models/fake_news_model.joblib")


if __name__ == "__main__":
    main()
