"""Tag-level aspect classification: rule baseline (from main.ipynb), zero-shot, supervised."""
import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from nlp_core import ASPECTS

# keyword lists copied from the original main.ipynb (substring matching, as in the notebook)
RULES = {
    "ambience": ["ambience", "environment", "place", "feel", "seating", "atmosphere", "music", "venue", "decor"],
    "value":    ["budget", "portion", "economical", "price", "worth", "money"],
    "food":     ["variety", "taste", "menu", "food", "dessert", "appetizer"],
    "service":  ["service", "staff", "behaviour"],
    "delivery": ["delivery", "packaging"],
}
DESCRIPTIONS = {
    "food": "food taste menu dishes quality portions", "service": "service staff waiters behaviour hospitality",
    "ambience": "ambience atmosphere decor music seating place crowd", "value": "price value for money affordable cheap cost",
    "delivery": "delivery packaging takeaway order", "other": "location parking payment",
}

def rule_predict(tags):
    out = []
    for t in tags:
        r = [int(any(k in t for k in RULES.get(a, []))) for a in ASPECTS[:-1]]
        out.append(r + [int(sum(r) == 0)])
    return np.array(out)

def zero_shot_predict(E_tags, E_desc, margin=0.02):
    S = E_tags @ E_desc.T
    return (S >= S.max(1, keepdims=True) - margin).astype(int)

class AspectModel:
    """One-vs-rest logistic regression on [char n-gram TF-IDF + sentence embedding]."""
    def __init__(self, embedder, C=10.0):
        self.E, self.C = embedder, C
    def _feats(self, texts, fit=False):
        emb = self.E.encode(texts)
        if fit:
            self.tf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True).fit(texts)
        return hstack([self.tf.transform(texts), csr_matrix(emb)]).tocsr()
    def fit(self, texts, Y):
        X = self._feats(list(texts), fit=True)
        self.models = [LogisticRegression(C=self.C, max_iter=3000, class_weight="balanced").fit(X, Y[:, j])
                       if Y[:, j].sum() else None for j in range(Y.shape[1])]
        return self
    def predict_proba(self, texts):
        X = self._feats(list(texts))
        return np.column_stack([m.predict_proba(X)[:, 1] if m else np.zeros(X.shape[0]) for m in self.models])
    def predict(self, texts, thr=0.5):
        return (self.predict_proba(texts) >= thr).astype(int)

def oof_proba(texts, Y, embedder, n_splits=5, seed=0):
    """Out-of-fold probabilities: every tag is scored by a model that never saw its label."""
    texts = np.asarray(texts, dtype=object)
    P = np.zeros(Y.shape, dtype=float)
    for tr, te in KFold(n_splits, shuffle=True, random_state=seed).split(texts):
        P[te] = AspectModel(embedder).fit(texts[tr], Y[tr]).predict_proba(texts[te])
    return P
