"""Core helpers for the restaurant NLP extension (cleaning, embeddings, evaluation)."""
import ast, os, re
import numpy as np
import pandas as pd

MOJIBAKE = {"ï¿½": "'", "�": "'", "�": "-"}
ASPECTS = ["food", "service", "ambience", "value", "delivery", "other"]

# ---------- cleaning ----------
def _num(x):
    """'3,705' -> 3705, '11.2K' -> 11200, '-', 'New', '0' -> NaN-safe."""
    s = str(x).strip().replace(",", "")
    if s in ("", "-", "New", "nan", "None"):
        return np.nan
    mult = 1000 if s.lower().endswith("k") else 1
    s = s.rstrip("kK")
    try:
        return float(s) * mult
    except ValueError:
        return np.nan

def clean_raw(path="data.csv"):
    df = pd.read_csv(path, encoding="cp1252")
    for bad, good in MOJIBAKE.items():
        for c in df.select_dtypes(include=["object", "string"]).columns:
            df[c] = df[c].astype(str).str.replace(bad, good, regex=False)
    df = df.rename(columns={"Dinning Rating": "dining_rating", "Number of dinning rating": "dining_votes",
                            "Delivery Rating": "delivery_rating", "Number of delivery rating": "delivery_votes",
                            "Restaurant Name": "name", "Area": "area", "Address": "address",
                            "Cuisines": "cuisines", "People Say": "people_say", "Average Cost": "avg_cost",
                            "Lattitude": "lat", "Longitude": "lon"})
    for c in ["dining_rating", "delivery_rating", "dining_votes", "delivery_votes"]:
        df[c] = df[c].map(_num)
    # a rating with zero votes is not a rating
    df.loc[df["dining_votes"].fillna(0) == 0, "dining_rating"] = np.nan
    df.loc[df["delivery_votes"].fillna(0) == 0, "delivery_rating"] = np.nan
    cost = df["avg_cost"].str.extract(r"([\d,]+)\s+for\s+(\w+)")
    df["cost_amount"] = cost[0].str.replace(",", "").astype(float)
    df["cost_basis"] = cost[1].map({"two": "two people", "one": "one order"})
    df["cost_pp"] = np.where(df["cost_basis"] == "two people", df["cost_amount"] / 2, df["cost_amount"])
    df["cuisines"] = df["cuisines"].map(lambda s: ast.literal_eval(s) if s.startswith("[") else [])
    ps = df["people_say"].str.strip()
    df["tags"] = ps.map(lambda s: [] if s in ("0", "nan", "-", "") else
                        [t.strip().lower() for t in s.split(",") if t.strip()])
    return df

def dedupe_outlets(df):
    """The same outlet appears once per area search; collapse to one row per (name, address)."""
    df = df.copy()
    # when scrapes disagree, trust the row backed by the most votes
    df = df.sort_values(["delivery_votes", "dining_votes"], ascending=False, na_position="last")
    df["outlet_id"] = (df["name"].str.lower().str.strip() + " | " + df["address"].str.lower().str.strip())
    def first_valid(s):
        s = s.dropna()
        return s.iloc[0] if len(s) else np.nan
    agg = {"name": "first", "address": "first", "lat": "first", "lon": "first", "cost_pp": first_valid,
           "cost_basis": first_valid, "cuisines": "first", "dining_rating": first_valid, "dining_votes": first_valid,
           "delivery_rating": first_valid, "delivery_votes": first_valid,
           "area": lambda s: sorted(set(s)),
           "tags": lambda s: sorted({t for L in s for t in L})}
    out = df.groupby("outlet_id", sort=False).agg(agg).reset_index()
    out = out.rename(columns={"area": "areas"})
    out["rating"] = out[["dining_rating", "delivery_rating"]].mean(axis=1)
    out["votes"] = out[["dining_votes", "delivery_votes"]].sum(axis=1, min_count=1)
    return out

def bayes_rating(rating, votes, m=50):
    """Shrink ratings with few votes toward the global mean."""
    C = np.nanmean(rating)
    v = np.nan_to_num(votes, nan=0.0)
    r = np.where(np.isnan(rating), C, rating)
    return (v * r + m * C) / (v + m)

# ---------- embeddings ----------
class Embedder:
    """backend='spacy' (word-vector average, works offline) or 'sbert' (sentence-transformers)."""
    def __init__(self, backend=None):
        backend = backend or os.environ.get("EMBED_BACKEND", "spacy")
        self.backend = backend
        if backend == "sbert":
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        else:
            import spacy
            path = os.environ.get("SPACY_MODEL_PATH", "en_core_web_md")
            self.nlp = spacy.load(path, disable=["parser", "ner", "lemmatizer", "attribute_ruler"])
    def encode(self, texts):
        texts = list(texts)
        if self.backend == "sbert":
            E = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            return np.asarray(E)
        out = np.zeros((len(texts), self.nlp.vocab.vectors.shape[1]), dtype="float32")
        for i, doc in enumerate(self.nlp.pipe(texts, batch_size=256)):
            vecs = [t.vector for t in doc if t.has_vector and not t.is_stop and t.is_alpha]
            if not vecs:
                vecs = [t.vector for t in doc if t.has_vector]
            if vecs:
                v = np.mean(vecs, axis=0)
                out[i] = v / (np.linalg.norm(v) + 1e-9)
        return out

# ---------- metrics ----------
def prf(y_true, y_pred):
    from sklearn.metrics import f1_score, precision_score, recall_score
    return dict(micro_f1=f1_score(y_true, y_pred, average="micro", zero_division=0),
                macro_f1=f1_score(y_true, y_pred, average="macro", zero_division=0),
                micro_p=precision_score(y_true, y_pred, average="micro", zero_division=0),
                micro_r=recall_score(y_true, y_pred, average="micro", zero_division=0))

def precision_at_k(rels, k):
    return float(np.mean(rels[:k])) if len(rels) else 0.0

def ndcg_at_k(rels, k, n_relevant):
    rels = np.asarray(rels[:k], dtype=float)
    dcg = np.sum(rels / np.log2(np.arange(2, len(rels) + 2)))
    ideal = np.sum(1.0 / np.log2(np.arange(2, min(k, n_relevant) + 2)))
    return float(dcg / ideal) if ideal > 0 else 0.0
