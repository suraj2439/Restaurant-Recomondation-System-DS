"""Semantic restaurant search: documents, three retrieval systems, query parsing, evaluation."""
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nlp_core import ASPECTS, bayes_rating, ndcg_at_k, precision_at_k

CHEAP_WORDS = ["cheap", "inexpensive", "affordable", "pocket friendly", "pocket-friendly", "budget", "low cost"]
PREMIUM_WORDS = ["upscale", "premium", "fine dining", "luxury", "expensive", "high end"]

def price_band(cost, q33, q67):
    return "budget cheap" if cost <= q33 else ("premium expensive" if cost >= q67 else "mid-range")

def build_docs(o, q33, q67):
    """One text document per outlet: cuisines, areas, price band and the review tags."""
    def doc(r):
        parts = [", ".join(r["cuisines"]), r["address"], price_band(r["cost_pp"], q33, q67), ". ".join(r["tags"])]
        return " ".join(p for p in parts if p)
    return o.apply(doc, axis=1).tolist()

def parse_query(q, areas, cuisines):
    ql = q.lower()
    area = next((a for a in areas if a.replace("-", " ") in ql), None)
    cuisine = next((c for c in sorted(cuisines, key=len, reverse=True) if c in ql), None)
    budget = "cheap" if any(w in ql for w in CHEAP_WORDS) else ("premium" if any(w in ql for w in PREMIUM_WORDS) else None)
    return dict(area=area, cuisine=cuisine, budget=budget)

class Searcher:
    def __init__(self, o, docs, embedder, q33, q67, tag_proba=None, aspect_model=None, w_prior=0.15, w_aspect=0.3):
        self.o, self.docs, self.E, self.q33, self.q67, self.w = o.reset_index(drop=True), docs, embedder, q33, q67, w_prior
        self.w_aspect, self.aspect_model = w_aspect, aspect_model
        # per-outlet aspect profile = mean predicted aspect probability over its tags (0 if no tags)
        self.profile = np.array([np.mean([tag_proba[t] for t in T], axis=0) if T else np.zeros(len(ASPECTS))
                                 for T in self.o["tags"]]) if tag_proba is not None else None
        self.tfidf = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2).fit(docs)
        self.Xt = self.tfidf.transform(docs)
        self.Xe = embedder.encode(docs)
        prior = bayes_rating(o["rating"].values, o["votes"].values)
        self.prior = (prior - prior.min()) / (prior.max() - prior.min())
        self.areas = sorted({a for L in o["areas"] for a in L})
        self.cuisines = sorted({c.lower() for L in o["cuisines"] for c in L})
        self.addr = o["address"].str.lower().values
    def _scores(self, q, system):
        """system = base ranker plus optional parts, e.g. 'embedding+filters+prior+aspect'.
        base: tfidf | embedding.  filters: hard area/cuisine/budget parsed from the query.
        prior: Bayesian rating.  aspect: query aspects x outlet aspect profile (from the tag classifier)."""
        base, *flags = system.split("+")
        if base == "tfidf":
            s = (self.Xt @ self.tfidf.transform([q]).T).toarray().ravel()
        elif base == "embedding":
            s = self.Xe @ self.E.encode([q])[0]
        else:
            raise ValueError(system)
        p = parse_query(q, self.areas, self.cuisines)
        if "prior" in flags:
            s = s + self.w * self.prior
        if "aspect" in flags and self.profile is not None:
            q_clean = q.lower().replace(p["area"].replace("-", " "), " ") if p["area"] else q
            qa = self.aspect_model.predict_proba([q_clean])[0][:-1]      # drop "other"
            s = s + self.w_aspect * (self.profile[:, :-1] @ qa)
        if "filters" in flags:
            mask = np.ones(len(s), dtype=bool)
            if p["area"]:
                mask &= np.array([p["area"].replace("-", " ") in a for a in self.addr])
            if p["cuisine"]:
                mask &= self.o["cuisines"].map(lambda L: p["cuisine"] in [c.lower() for c in L]).values
            if p["budget"] == "cheap":
                mask &= (self.o["cost_pp"] <= self.q33).values
            if p["budget"] == "premium":
                mask &= (self.o["cost_pp"] >= self.q67).values
            if mask.sum() >= 5:          # never filter down to nothing
                s = np.where(mask, s, s - 10)
        return s
    def search(self, q, system="embedding+filters+prior+aspect", k=10):
        s = self._scores(q, system)
        idx = np.argsort(-s)[:k]
        return self.o.iloc[idx][["name", "address", "cuisines", "cost_pp", "rating"]].assign(score=s[idx])
    def rank(self, q, system):
        return np.argsort(-self._scores(q, system))

# ---------- evaluation ----------
def relevance_mask(o, gold_counts, spec, q33, q67, min_tags=2):
    """Relevance derived from structured fields + hand-labeled tags (not seen by the retrieval systems)."""
    m = np.ones(len(o), dtype=bool)
    if spec.get("area"):
        m &= o["address"].str.lower().str.contains(spec["area"].replace("-", " ")).values
    if spec.get("cuisine"):
        m &= o["cuisines"].map(lambda L: spec["cuisine"] in [c.lower() for c in L]).values
    if spec.get("budget") == "cheap":
        m &= (o["cost_pp"] <= q33).values
    if spec.get("budget") == "premium":
        m &= (o["cost_pp"] >= q67).values
    if spec.get("aspect"):
        m &= (gold_counts[spec["aspect"]] >= min_tags).values
    return m

def evaluate(searcher, queries, gold_counts, systems, k=10, min_rel=5):
    rows = []
    for q in queries:
        rel = relevance_mask(searcher.o, gold_counts, q, searcher.q33, searcher.q67)
        if rel.sum() < min_rel:
            continue
        for sysname in systems:
            order = searcher.rank(q["text"], sysname)
            r = rel[order]
            rows.append(dict(query=q["text"], system=sysname, n_relevant=int(rel.sum()),
                             p_at_k=precision_at_k(r, k), ndcg_at_k=ndcg_at_k(r, k, int(rel.sum()))))
    return pd.DataFrame(rows)
