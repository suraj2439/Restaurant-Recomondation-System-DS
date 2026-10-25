"""Regenerates the README figures from outputs/ (banner, pipeline, stat tiles, charts, demo)."""
import os, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib.pyplot as plt, matplotlib as mpl
from matplotlib.patches import FancyBboxPatch
os.makedirs("figures", exist_ok=True)
INK, MUTED, TEAL, CORAL, SAND, GRID = "#1F2933", "#6B7785", "#0F8B8D", "#EC6B56", "#F6EFE6", "#E4DDD2"
mpl.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})

res = pd.read_csv("outputs/aspect_classification_results.csv", index_col=0)
sm = pd.read_csv("outputs/search_eval_summary.csv", index_col=0)
rules, sup = res.loc["Keyword rules (main.ipynb)"], res.loc["Supervised (char TF-IDF + embedding)"]
pe = pd.read_csv("outputs/search_eval_per_query.csv")
full, base = "embedding+filters+prior+aspect", "embedding+filters+prior"
a = pe[pe.system == full].set_index("query")["p_at_k"]; b = pe[pe.system == base].set_index("query")["p_at_k"]
d = (a - b).dropna().values; rng = np.random.default_rng(0)
boot = [rng.choice(d, len(d)).mean() for _ in range(5000)]; lo, hi = np.percentile(boot, [2.5, 97.5])

# ---------- banner ----------
open("figures/banner.svg", "w").write(f'''<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="300" viewBox="0 0 1200 300" role="img" aria-label="Restaurant Insights banner">
<defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#0B3C49"/><stop offset="1" stop-color="{TEAL}"/></linearGradient></defs>
<rect width="1200" height="300" rx="22" fill="url(#g)"/>
<g opacity="0.10" fill="none" stroke="#fff" stroke-width="3"><circle cx="1010" cy="150" r="140"/><circle cx="1010" cy="150" r="100"/><circle cx="1010" cy="150" r="60"/></g>
<g transform="translate(960 95)" fill="none" stroke="#fff" stroke-width="7" stroke-linecap="round" stroke-linejoin="round">
<circle cx="52" cy="55" r="44" stroke="{SAND}"/><circle cx="52" cy="55" r="28" stroke="{SAND}" opacity=".6"/>
<path d="M-18 8 V40 M-8 8 V40 M2 8 V40 M-18 40 Q-8 52 2 40 M-8 40 V108" stroke="#fff"/>
<path d="M122 8 Q142 30 122 60 V108 M122 8 V60" stroke="#fff"/></g>
<text x="64" y="128" font-family="Georgia, 'Times New Roman', serif" font-size="62" font-weight="700" fill="#fff">Restaurant Insights</text>
<text x="66" y="176" font-family="Helvetica, Arial, sans-serif" font-size="25" fill="{SAND}">NLP and semantic search on Zomato Pune data</text>
<g font-family="Helvetica, Arial, sans-serif" font-size="17" font-weight="600">
<rect x="66" y="208" width="200" height="38" rx="19" fill="#fff" opacity=".16"/><text x="166" y="233" text-anchor="middle" fill="#fff">Aspect classification</text>
<rect x="280" y="208" width="190" height="38" rx="19" fill="#fff" opacity=".16"/><text x="375" y="233" text-anchor="middle" fill="#fff">Semantic search</text>
<rect x="484" y="208" width="190" height="38" rx="19" fill="#fff" opacity=".16"/><text x="579" y="233" text-anchor="middle" fill="#fff">Measured results</text></g>
</svg>''')

# ---------- stat tiles ----------
tiles = [("2,560", "unique outlets", "from 9,110 scraped rows"), ("510", "hand-labeled tags", "five aspects plus other"),
         (f"{rules['freq-split micro-F1']:.2f} → {sup['freq-split micro-F1']:.2f}", "tag F1, rare tags", "keyword rules to supervised"),
         (f"+{d.mean():.3f}", "search P@10", "from the aspect component")]
w, g = 270, 20; t = ""
for i, (big, label, sub) in enumerate(tiles):
    x = i * (w + g)
    t += (f'<rect x="{x}" y="0" width="{w}" height="150" rx="18" fill="{SAND}" stroke="{GRID}"/>'
          f'<rect x="{x+20}" y="22" width="36" height="6" rx="3" fill="{CORAL if i%2 else TEAL}"/>'
          f'<text x="{x+20}" y="85" font-family="Georgia, serif" font-size="{44 if len(big)<8 else 38}" font-weight="700" fill="{INK}">{big}</text>'
          f'<text x="{x+20}" y="113" font-family="Helvetica, Arial, sans-serif" font-size="17" font-weight="700" fill="{TEAL}">{label}</text>'
          f'<text x="{x+20}" y="134" font-family="Helvetica, Arial, sans-serif" font-size="14" fill="{MUTED}">{sub}</text>')
open("figures/stats.svg", "w").write(f'<svg xmlns="http://www.w3.org/2000/svg" width="1160" height="150" viewBox="0 0 1160 150">{t}</svg>')

# ---------- pipeline ----------
steps = [("1", "Clean", "dedupe 9,110 rows\nto 2,560 outlets"), ("2", "Label", "510 distinct tags\nby hand"), ("3", "Classify", "rules vs zero-shot\nvs supervised"),
         ("4", "Profile", "aspect mix per\noutlet, out-of-fold"), ("5", "Search", "embeddings, filters,\nrating prior, aspects"), ("6", "Evaluate", "P@10, NDCG@10,\nbootstrap CI")]
bw, gap = 170, 28; p = ""
for i, (n, title, sub) in enumerate(steps):
    x = i * (bw + gap); col = TEAL if i < 3 else CORAL
    p += (f'<rect x="{x}" y="10" width="{bw}" height="150" rx="18" fill="{SAND}" stroke="{GRID}"/>'
          f'<circle cx="{x+30}" cy="46" r="17" fill="{col}"/><text x="{x+30}" y="52" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="17" font-weight="700" fill="#fff">{n}</text>'
          f'<text x="{x+56}" y="53" font-family="Georgia, serif" font-size="22" font-weight="700" fill="{INK}">{title}</text>')
    for j, line in enumerate(sub.split("\n")):
        p += f'<text x="{x+18}" y="{96+j*22}" font-family="Helvetica, Arial, sans-serif" font-size="14.5" fill="{MUTED}">{line}</text>'
    if i < len(steps) - 1:
        p += f'<path d="M{x+bw+4} 85 L{x+bw+gap-6} 85" stroke="{MUTED}" stroke-width="2.5" stroke-linecap="round"/><path d="M{x+bw+gap-14} 78 L{x+bw+gap-6} 85 L{x+bw+gap-14} 92" fill="none" stroke="{MUTED}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>'
open("figures/pipeline.svg", "w").write(f'<svg xmlns="http://www.w3.org/2000/svg" width="1160" height="170" viewBox="0 0 1160 170">{p}</svg>')

# ---------- charts ----------
ASP = ["food", "service", "ambience", "value", "delivery", "other"]
f1 = pd.DataFrame(index=ASP)
# per-aspect F1 recomputed so the chart always matches the code
import os as _os
_os.environ.setdefault("SPACY_MODEL_PATH", "en_core_web_md")
from nlp_core import *; from aspects import *; from search import *
from sklearn.metrics import f1_score
raw = clean_raw("data.csv"); outlets = dedupe_outlets(raw); q33, q67 = outlets["cost_pp"].quantile([.33, .67])
EMB = Embedder(); lab = pd.read_csv("labels/aspect_labels_draft.csv"); tags_l, Y = lab["tag"].tolist(), lab[ASPECTS].values
P_rules = rule_predict(tags_l); P_cv = (oof_proba(tags_l, Y, EMB, 5, 0) >= 0.5).astype(int)
E_t, E_d = EMB.encode(tags_l), EMB.encode([DESCRIPTIONS[x] for x in ASPECTS]); P_zero = zero_shot_predict(E_t, E_d)
f1 = pd.DataFrame({"Keyword rules": f1_score(Y, P_rules, average=None), "Zero-shot": f1_score(Y, P_zero, average=None),
                   "Supervised": f1_score(Y, P_cv, average=None)}, index=ASPECTS)
fig, ax = plt.subplots(figsize=(10, 4.6)); xs = np.arange(len(ASP)); wd = 0.26
for k, (c, col) in enumerate(zip(f1.columns, ["#C9C2B6", "#7FC4C5", TEAL])):
    bars = ax.bar(xs + (k - 1) * wd, f1[c], wd, label=c, color=col)
    if c == "Supervised":
        for r in bars: ax.text(r.get_x() + r.get_width() / 2, r.get_height() + .02, f"{r.get_height():.2f}", ha="center", fontsize=9, color=INK)
ax.set_xticks(xs, [s.capitalize() for s in ASP]); ax.set_ylim(0, 1.08); ax.set_ylabel("F1 (5-fold cross-validation)")
ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True); ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(.5, 1.13))
plt.tight_layout(); plt.savefig("figures/readme_aspect_f1.png", dpi=200, facecolor="white"); plt.close()

names = {"tfidf": "TF-IDF", "embedding": "Embedding", "tfidf+filters": "TF-IDF + filters", "embedding+filters": "Embedding + filters",
         "embedding+filters+prior": "+ rating prior", full: "+ aspect classifier"}
order = ["tfidf", "embedding", "tfidf+filters", "embedding+filters", "embedding+filters+prior", full]
vals = sm.loc[order, "P@10"].values; cols = ["#C9C2B6"] * 3 + ["#7FC4C5", "#7FC4C5", CORAL]
fig, ax = plt.subplots(figsize=(10, 4.4)); y = np.arange(len(order))[::-1]
ax.barh(y, vals, color=cols, height=.62)
for yi, v in zip(y, vals): ax.text(v + .012, yi, f"{v:.3f}", va="center", fontsize=10, color=INK)
ax.set_yticks(y, [names[k] for k in order]); ax.set_xlim(0, 1.08); ax.set_xlabel("mean precision@10 over 20 queries")
ax.xaxis.grid(True, color=GRID); ax.set_axisbelow(True)
ax.annotate(f"aspect part: {d.mean():+.3f}\n95% CI {lo:+.3f} to {hi:+.3f}", xy=(vals[-1], y[-1]), xytext=(.83, y[-1]),
            fontsize=10, color=CORAL, va="center", arrowprops=None)
plt.tight_layout(); plt.savefig("figures/readme_search_eval.png", dpi=200, facecolor="white"); plt.close()

# ---------- demo ----------
tag_proba = dict(zip(tags_l, oof_proba(tags_l, Y, EMB, 5, 0))); am = AspectModel(EMB).fit(tags_l, Y)
S = Searcher(outlets, build_docs(outlets, q33, q67), EMB, q33, q67, tag_proba=tag_proba, aspect_model=am)
def demo(q, fname):
    top = S.search(q, full, k=5)
    cut = lambda t, n: t if len(t) <= n else t[:n - 1] + "…"
    cells = [[cut(r["name"], 34), cut(", ".join(r["cuisines"][:3]), 36), f"{int(r['cost_pp'])}",
              "" if pd.isna(r["rating"]) else f"{r['rating']:.1f}"] for _, r in top.iterrows()]
    fig, ax = plt.subplots(figsize=(11, 3.4)); ax.axis("off")
    ax.text(0, 1.02, "Query", fontsize=11, color=MUTED, transform=ax.transAxes, va="bottom")
    ax.text(.07, 1.02, f"“{q}”", fontsize=14, color=INK, fontweight="bold", transform=ax.transAxes, va="bottom")
    tb = ax.table(cellText=cells, colLabels=["Restaurant", "Cuisines", "Cost (INR)", "Rating"], loc="center", cellLoc="left", colWidths=[.38, .42, .11, .09])
    tb.auto_set_font_size(False); tb.set_fontsize(10.5); tb.scale(1, 2.0)
    for (r, c), cell in tb.get_celld().items():
        cell.set_edgecolor(GRID); cell.PAD = 0.04
        if r == 0: cell.set_facecolor(TEAL); cell.get_text().set_color("white"); cell.get_text().set_fontweight("bold")
        else: cell.set_facecolor(SAND if r % 2 else "white")
    plt.savefig(fname, dpi=200, facecolor="white", bbox_inches="tight"); plt.close()
demo("friendly staff and quick service in wakad", "figures/demo_service.png")
demo("cozy place with a nice vibe in baner", "figures/demo_ambience.png")
print("figures done")
