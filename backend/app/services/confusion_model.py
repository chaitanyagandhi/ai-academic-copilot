import os
import pickle
from pathlib import Path
from typing import Optional, Tuple, List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


_BACKEND_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = _BACKEND_DIR / "data" / "confusion_model.pkl"


def _build_synthetic_dataset() -> Tuple[List[str], List[int]]:
    topics = [
        "gradient descent",
        "backpropagation",
        "regularization",
        "Bayes theorem",
        "eigenvalues",
        "binary search",
        "hash tables",
        "linked lists",
    ]

    confused_templates = [
        "I am confused about {}",
        "Can you explain {}?",
        "Why does {} work?",
        "How does {} actually work?",
        "This is not clear: {}",
        "What's the difference between {} and something else?",
        "I don't understand {}",
        "Can you help me with {}?",
    ]

    factual_templates = [
        "Define {}",
        "What is {}?",
        "Give an example of {}",
        "List the steps of {}",
        "Summarize {} in one sentence",
        "Provide a formula related to {}",
    ]

    texts: List[str] = []
    labels: List[int] = []

    for t in topics:
        for tmpl in confused_templates:
            texts.append(tmpl.format(t))
            labels.append(1)
        for tmpl in factual_templates:
            texts.append(tmpl.format(t))
            labels.append(0)

    return texts, labels


def _train_model() -> Tuple[TfidfVectorizer, LogisticRegression]:
    texts, labels = _build_synthetic_dataset()
    vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", max_features=8000)
    X = vec.fit_transform(texts)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, labels)
    return vec, clf


def _save_model(vec: TfidfVectorizer, clf: LogisticRegression) -> None:
    os.makedirs(MODEL_PATH.parent, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"vectorizer": vec, "model": clf}, f)


def _load_model() -> Optional[Tuple[TfidfVectorizer, LogisticRegression]]:
    if not MODEL_PATH.exists():
        return None
    try:
        with open(MODEL_PATH, "rb") as f:
            data = pickle.load(f)
        return data["vectorizer"], data["model"]
    except Exception:
        return None


def load_or_train_model() -> Optional[Tuple[TfidfVectorizer, LogisticRegression]]:
    loaded = _load_model()
    if loaded is not None:
        return loaded
    try:
        vec, clf = _train_model()
        _save_model(vec, clf)
        return vec, clf
    except Exception:
        return None


def predict_confusion(question: str) -> Optional[float]:
    """
    Returns confusion probability (0..1) if model available, else None.
    """
    model = load_or_train_model()
    if model is None:
        return None
    vec, clf = model
    X = vec.transform([question])
    proba = clf.predict_proba(X)[0][1]
    return float(proba)
