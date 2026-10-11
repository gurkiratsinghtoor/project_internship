"""
Synthetic news generator (template based).

Purpose: create clearly FICTIONAL, labelled headlines + articles for
  1. a demo dataset when the Kaggle dataset is not downloaded,
  2. stress-testing the predictor with sensational vs neutral writing styles.
All names/places are invented. This is NOT meant to produce real-world misinformation.
"""
import os
import random
import pandas as pd

PLACES = ["Northvale", "Eastmere", "Lakeshore City", "Brookhaven", "Redstone County"]
PEOPLE = ["Dr. Alan Kest", "Mayor Lena Ortiz", "Senator Paul Rennick", "Prof. Mira Dolan", "Minister Ravi Nair"]
TOPICS = ["new tax policy", "water supply project", "school funding bill", "public transport plan",
          "healthcare reform", "energy grid upgrade", "city budget", "vaccination drive"]

FAKE_HEADLINES = [
    "SHOCKING: {person} SECRETLY hiding the truth about {topic}!!!",
    "You won't believe what {person} just did in {place} - media is silent!",
    "BREAKING!!! {topic} is a massive cover-up, insiders reveal everything",
    "Doctors HATE this: {place} residents expose the real {topic} scandal",
    "{person} caught lying about {topic}? Share before they delete this!",
]
FAKE_BODIES = [
    "Anonymous sources say the elite do not want you to know the truth. Wake up people! "
    "Everything they told you about the {topic} is a lie and the mainstream media is in on it. "
    "Share this with everyone you know before it gets censored.",
    "A secret leaked document PROVES that {person} planned the whole thing. "
    "No one is talking about it because they are all paid off. This is the biggest scandal ever. "
    "Click and share now, the government is trying to silence us.",
]
REAL_HEADLINES = [
    "{person} announces {topic} for {place}",
    "{place} council reviews {topic} after public consultation",
    "Officials in {place} outline timeline for {topic}",
    "{person} says {topic} will be debated next week",
    "Report: {topic} in {place} expected to cost less than projected",
]
REAL_BODIES = [
    "{person} told reporters on Tuesday that the {topic} would be reviewed by a committee in {place}. "
    "According to an official statement, the proposal will be open for public comment for 30 days, "
    "and a final vote is scheduled for next month.",
    "The {place} administration said the {topic} follows a report from an independent panel. "
    "{person} noted that funding details will be published and that officials will answer questions "
    "from residents at a public meeting.",
]


def _fill(template: str, rng: random.Random) -> str:
    return template.format(person=rng.choice(PEOPLE), place=rng.choice(PLACES), topic=rng.choice(TOPICS))


def generate(n: int = 400, seed: int = 42) -> pd.DataFrame:
    """Return DataFrame with columns title, text, label (1 = fake, 0 = real)."""
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        label = i % 2
        if label == 1:
            title, text = rng.choice(FAKE_HEADLINES), rng.choice(FAKE_BODIES)
        else:
            title, text = rng.choice(REAL_HEADLINES), rng.choice(REAL_BODIES)
        rows.append({"title": _fill(title, rng), "text": _fill(text, rng), "label": label})
    return pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)


if __name__ == "__main__":
    df = generate(400)
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "sample_data.csv")
    df.to_csv(out, index=False)
    print(df.head(6).to_string())
