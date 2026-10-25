# Restaurant Insights: NLP Extension

Extends the original Zomato Pune EDA project (`main.ipynb`, untouched) with NLP and evaluation.
Open `restaurant_nlp_extension.ipynb` (already executed, so results are visible).

## Run it
```
pip install -r requirements.txt
python -m spacy download en_core_web_md
jupyter notebook restaurant_nlp_extension.ipynb
```
Optional: `EMBED_BACKEND=sbert` uses sentence-transformers (not run by me; numbers will differ).

## Files
- `nlp_core.py` cleaning, dedupe, embeddings, metrics. `aspects.py` aspect classifiers. `search.py` retrieval and evaluation.
- `labels/aspect_labels_draft.csv` hand labels for the 510 distinct tags. **Review them before quoting metrics.**
- `outputs/judgment_template.csv` fill the `relevant` column to evaluate search with your own judgments.
- `build_notebook.py` regenerates the notebook.

## Results (spaCy backend)
| Aspect classification of tags | CV micro-F1 | Frequency-split micro-F1 |
|---|---|---|
| Keyword rules (from main.ipynb) | 0.63 | 0.63 |
| Zero-shot embeddings | 0.76 | 0.73 |
| Supervised (char TF-IDF + embedding) | 0.91 | 0.88 |

Search (20 scored queries): embedding + filters + prior reaches P@10 0.62; adding the aspect component gives 0.74
(paired change +0.115, 95% bootstrap CI +0.065 to +0.170).

## Honest limits
Short templated tags (510 distinct, max 6 per outlet), one labeler, relevance derived from structured fields and the
same hand labels, 20 scored queries written by one person. Compares systems; not a measure of real user satisfaction.
The data was scraped from Zomato, which its terms generally prohibit.
