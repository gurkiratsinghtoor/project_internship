"""Command line predictor:  python src/predict.py "your news text here" """
import os, sys
import joblib
import numpy as np
from preprocess import clean_text

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "fake_news_model.joblib")


def load():
    return joblib.load(MODEL_PATH)


def predict(text: str, bundle=None) -> dict:
    bundle = bundle or load()
    vec = bundle["tfidf"].transform([clean_text(text)])
    model = bundle["model"]
    if hasattr(model, "predict_proba"):
        p_fake = float(model.predict_proba(vec)[0][1])
    else:  # Passive Aggressive has no probabilities -> squash decision score
        p_fake = float(1 / (1 + np.exp(-model.decision_function(vec)[0])))
    reasons = []
    if hasattr(model, "coef_"):
        names = bundle["tfidf"].get_feature_names_out()
        idx = vec.nonzero()[1]
        contrib = sorted(((names[i], float(vec[0, i] * model.coef_[0][i])) for i in idx), key=lambda x: -abs(x[1]))
        reasons = contrib[:5]
    return {"label": "FAKE" if p_fake >= 0.5 else "REAL", "fake_probability": p_fake, "key_terms": reasons}


if __name__ == "__main__":
    text = " ".join(sys.argv[1:]) or input("Enter news text: ")
    r = predict(text)
    print(f"Prediction: {r['label']}  (fake probability {r['fake_probability']:.2%})")
    for w, c in r["key_terms"]:
        print(f"  {w:20s} {'-> fake' if c > 0 else '-> real'} ({c:+.3f})")
