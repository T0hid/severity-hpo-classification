"""The code-derived numbers of the manuscript and how each is recomputed.

Each check has: an id, where the number appears, what it is, the value as printed in the manuscript,
a function that recomputes it from the verification run (see verify.Results), and the output that
holds it. A computed value passes when, rounded the way the manuscript prints it, it equals the
printed value. Numbers taken from cited publications (for example the surveys by Lazarin et al. and
Arjunan et al.) are not code output and are not listed.
"""
import re

import pandas as pd

from hpo_severity import exact_p as XP

CHECKS = []
SCORES = "baseline/scores_full_precision.csv"
VAL = "figures/data_and_reports/validation_terms_scored.csv"
REPORT = "figures/data_and_reports/analysis_report_Figures3to6_S2_S3.txt"
VREPORT = "figures/data_and_reports/validation_report_Figure2.txt"
F7 = "gene_level/figure7_data.xlsx"
CLS = "published_run/agent_classifications.zip"
MS = "published_run/model_selection_summary.csv"
GLOG = "gene table and Figure 7 logs"
GT = "data/ground_truth.xlsx"
PT = "figures/data_and_reports/phenotype_table_Figures3to6_S2.csv"
EXACTP = "figures/data_and_reports/exact_p_values.csv"
TIERED = ("1", "2", "3", "4")


def add(id_, location, what, expected, get, source):
    base = re.sub(r"[^a-z0-9]+", "-", id_.lower()).strip("-")
    key, i = base, 2
    while any(c["id"] == key for c in CHECKS):
        key, i = f"{base}-{i}", i + 1
    CHECKS.append({"id": key, "location": location, "what": what, "expected": expected, "get": get, "source": source})


def pct(x):
    return 100 * x


def exact_p(R, key):
    """A P value as printed in the manuscript (two significant figures), from exact_p_values.csv."""
    if "exact_p" not in R._cache:
        R._cache["exact_p"] = pd.read_csv(R.work / EXACTP, encoding="utf-8").set_index("key")["p"]
    return R._cache["exact_p"][key]


def mcnemar_p(R, level):
    b = int(R.score(level, "main minus baseline", "accuracy", "mcnemar_main_right_base_wrong"))
    c = int(R.score(level, "main minus baseline", "accuracy", "mcnemar_main_wrong_base_right"))
    return XP.fmt(XP.mcnemar(b, c))


# ------------------------------------------------------------------ helpers on the analysis report
def tier_counts(R):
    return R.block("Distribution by Classification Tier:", "Distribution by Classification Category:")


def cat_counts(R):
    return R.block("Distribution by Classification Category:", "2. CLINICAL LANDSCAPE")


def evidence_tier(R, tier, kind):
    b = R.block(f"\nTier {tier}:\n  Direct Support", "\n\n")
    return float(re.search(kind + r": ([\d.]+)%", b).group(1))


def evidence_cat(R):
    return R.table(R.block("3B. EVIDENCE PROFILE BY CATEGORY", "Kruskal-Wallis (Support Score across Categories"))


def pt_exploded(R, col):
    """The phenotype table with one row per phenotype and listed value of col (QoL domain or intervention type)."""
    d = R.pt.assign(**{col: R.pt[col].fillna("").str.split(";")}).explode(col)
    return d[d[col] != ""]


def qol_cat_share(R, cat, domain):
    d = pt_exploded(R, "qol_domains")
    d = d[d.classification_category == cat]
    return 100 * (d.qol_domains == domain).mean()


def mgmt_qol_share(R, status, kind):
    d = pt_exploded(R, "management_categories")
    d = d[d.QoL_assessment == status]
    return 100 * (d.management_categories == kind).mean()


def support_iqr(R):
    d = R.pt[R.pt.classification_tier.isin(TIERED)]
    q = d.groupby("classification_category").statement_support_score.quantile([0.25, 0.75]).unstack()
    return q[0.75] - q[0.25]


def qol_tier(R):
    t = R.table(R.block("4A. QOL DOMAIN COUNTS BY TIER", "Chi-square (QoL Domains vs Tier)"))
    cols = ["Cognitive/School", "Emotional Impact", "Physical Functioning", "Physical Symptoms", "Social Functioning"]
    return {k: dict(zip(cols, v)) for k, v in t.items()}


def qol_share(R, tier, domain):
    row = qol_tier(R)[str(tier)]
    return 100 * row[domain] / sum(row.values())


def qol_cat(R):
    t = R.table(R.block("5A. QOL ASSESSMENT BY CATEGORY", "Chi-square: chi2="))
    return {k: {"Affected": v[0], "Not Affected": v[1]} for k, v in t.items()}


def category_total(R, cat):
    return R.count_line(cat_counts(R), cat)


def mgmt_tier(R, tier, col):
    t = R.table(R.block("6C. INTERVENTIONS BY TIER", "Kruskal-Wallis: H(3)"))
    return t[str(tier)][["count", "mean", "std"].index(col)]


def mgmt_cat(R, cat):
    return R.table(R.block("6D. INTERVENTIONS BY CATEGORY", "Kruskal-Wallis: H(12)"))[cat][0]


def described(R, start, stat):
    b = R.block(start, "\n\n") if "\n\n" in R.report[R.report.index(start):] else R.block(start)
    m = re.search(r"^" + re.escape(stat) + r"\s+(-?[\d.]+)", b, re.M)
    return float(m.group(1))


def ms_acc(R, model, temperature=None):
    d = R.ms[R.ms.model == model]
    if temperature is not None:
        d = d[d.temperature == temperature]
    return d.overall_accuracy.mean()


def best_models(R):
    d = R.ms
    g = d.groupby(["model", "temperature"]).overall_accuracy.mean().round(6)
    best = g[g == g.max()]
    names = []
    for model in sorted(best.index.get_level_values(0).unique()):
        temps = sorted(best.loc[model].index)
        all_t = len(temps) == d[d.model == model].temperature.nunique()
        names.append(model.split("/")[-1] + ("" if all_t else " at " + "/".join(str(t) for t in temps)))
    return "; ".join(names)


def removed_terms(R):
    out = R.gt_all[~R.gt_all.hpo_id.isin(R.cls.hpo_id)]
    moi = out.hpo_name.str.contains("inherit|genic|Mitochondrial|X-linked|Y-linked", case=False)
    return len(out), int(moi.sum()), int((out.hpo_name == "Chronic").sum()), int((~moi & (out.hpo_name != "Chronic")).sum())


def model_selection_terms_in_validation(R):
    from .model_selection import GROUND_TRUTH
    ids = [g["hpo_id"] for g in GROUND_TRUTH]
    v = R.val[~R.val.hpo_id.isin(ids)]
    return len(ids), int(R.val.hpo_id.isin(ids).sum()), (v.Final_classification_category == v.classification_category_pred).mean()


# ================================================================== Abstract and summary
add("abs-terms", "Abstract", "HPO terms classified", "10,652", lambda R: len(R.cls), CLS)
add("abs-gt", "Abstract", "expert-labelled terms", "898", lambda R: R.score("category", "main run", "accuracy", "n"), SCORES)
add("abs-acc", "Abstract", "agent category accuracy", "90.4%", lambda R: pct(R.score("category", "main run", "accuracy")), SCORES)
add("abs-acc-lo", "Abstract", "agent category accuracy, CI low", "88.4%", lambda R: pct(R.score("category", "main run", "accuracy", "ci95_low")), SCORES)
add("abs-acc-hi", "Abstract", "agent category accuracy, CI high", "92.3%", lambda R: pct(R.score("category", "main run", "accuracy", "ci95_high")), SCORES)
add("abs-base", "Abstract", "baseline category accuracy", "73.2%", lambda R: pct(R.score("category", "baseline", "accuracy")), SCORES)
add("abs-evid-lo", "Abstract", "claims direct or valid inference, lowest tier", "83.2%",
    lambda R: min(evidence_tier(R, t, "Direct Support") + evidence_tier(R, t, "Valid Inferences") for t in (1, 2, 3, 4)), REPORT)
add("abs-evid-hi", "Abstract", "claims direct or valid inference, highest tier", "91.5%",
    lambda R: max(evidence_tier(R, t, "Direct Support") + evidence_tier(R, t, "Valid Inferences") for t in (1, 2, 3, 4)), REPORT)
add("abs-pairs", "Abstract", "gene-disease pairs with a class", "6,709", lambda R: R.f7["7A"].pairs.sum(), F7)
add("abs-sp", "Abstract", "pairs Severe or Profound", "73.8%",
    lambda R: 100 * R.f7["7A"].set_index("severity").loc[["Profound", "Severe"], "pairs"].sum() / R.f7["7A"].pairs.sum(), F7)
add("abs-ar", "Abstract", "autosomal recessive pairs Severe or Profound", "78.9%",
    lambda R: (lambda r: 100 * (r.Profound + r.Severe) / r.pairs)(R.f7row("7C", "inheritance", "Autosomal recessive")), F7)
add("abs-genes", "Abstract", "genes in the analysis", "4,531", lambda R: R.f7row("7D", "gene_set", "All genes (base rate)").genes_labelled, F7)
add("abs-genes-sp", "Abstract", "genes with a Severe or Profound condition", "79.0%",
    lambda R: (lambda r: 100 * (r.Profound + r.Severe) / r.genes_labelled)(R.f7row("7D", "gene_set", "All genes (base rate)")), F7)
add("abs-cs", "Abstract", "ACMG carrier screening genes Severe or Profound", "90.2%",
    lambda R: (lambda r: 100 * (r.Profound + r.Severe) / r.genes_labelled)(R.f7row("7D", "gene_set", "ACMG carrier screening")), F7)
add("abs-mm", "Abstract", "Mackenzie's Mission genes Severe or Profound", "90.2%",
    lambda R: (lambda r: 100 * (r.Profound + r.Severe) / r.genes_labelled)(R.f7row("7D", "gene_set", "Mackenzie's Mission")), F7)
add("abs-intol", "Abstract", "Human Intolerome genes Severe or Profound", "92.9%",
    lambda R: (lambda r: 100 * (r.Profound + r.Severe) / r.genes_labelled)(R.f7row("7D", "gene_set", "Human Intolerome (fetal effect)")), F7)
add("sum-agree", "Summary", "agreement with experts", "90%", lambda R: pct(R.score("category", "main run", "accuracy")), SCORES)

# ================================================================== Results: model selection
MS_MODELS = {"openai/gpt-5": "GPT-5", "x-ai/grok-4": "Grok-4", "google/gemini-2.5-pro": "Gemini-2.5-pro",
             "anthropic/claude-3.5-sonnet": "Claude-3.5-sonnet", "openai/gpt-4o": "GPT-4o", "openai/gpt-4o-mini": "GPT-4o-mini"}
add("ms-models", "Methods, model selection", "LLMs benchmarked", "GPT-5, Grok-4, Gemini-2.5-pro, Claude-3.5-sonnet, GPT-4o, GPT-4o-mini",
    lambda R: ", ".join(MS_MODELS.get(m, m) for m in R.ms.model.unique()), MS)
add("ms-temperatures", "Methods, model selection", "temperatures", "0.0, 0.2, 0.5",
    lambda R: ", ".join(f"{t:.1f}" for t in sorted(R.ms.temperature.unique())), MS)
add("ms-top-p", "Methods, model selection", "top-p values", "0.90, 0.95, 1.0",
    lambda R: ", ".join(f"{t:.2f}" if t < 1 else f"{t:.1f}" for t in sorted(R.ms.top_p.unique())), MS)
add("ms-runs", "Methods, model selection", "settings per model (3 temperatures x 3 top-p)", "9",
    lambda R: int(R.ms.groupby("model").size().unique()[0]) if R.ms.groupby("model").size().nunique() == 1 else -1, MS)
add("ms-terms", "Results, model selection", "benchmark terms", "14", lambda R: model_selection_terms_in_validation(R)[0], "hpo_severity/model_selection.py")
add("ms-best", "Results, model selection", "models with the highest accuracy", "claude-3.5-sonnet; gpt-4o at 0.2", best_models, MS)
add("ms-acc", "Results, model selection", "highest accuracy (GPT-4o, temperature 0.2)", "92.9%", lambda R: pct(ms_acc(R, "openai/gpt-4o", 0.2)), MS)
add("ms-acc-claude", "Results, model selection", "highest accuracy (Claude 3.5 Sonnet)", "92.9%", lambda R: pct(ms_acc(R, "anthropic/claude-3.5-sonnet")), MS)
add("ms-13", "Results, model selection", "terms correct at the highest accuracy", "13", lambda R: ms_acc(R, "openai/gpt-4o", 0.2) * 14, MS)
add("ms-time-gpt", "Results, model selection", "GPT-4o time per term at 0.2 (s)", "7.6",
    lambda R: R.ms[(R.ms.model == "openai/gpt-4o") & (R.ms.temperature == 0.2)].avg_response_time.mean(), MS)
add("ms-time-claude", "Results, model selection", "Claude 3.5 Sonnet time per term (s)", "14",
    lambda R: R.ms[R.ms.model == "anthropic/claude-3.5-sonnet"].avg_response_time.mean(), MS)

# ================================================================== Results: validation (Figure 2)
add("val-n", "Results, validation", "validation terms", "898", lambda R: len(R.val), VAL)
add("val-psi", "Results, validation", "PSI", "0.0303", lambda R: float(re.search(r"PSI = ([\d.]+)", R.vreport).group(1)), VREPORT)
add("val-pw", "Results, validation", "weighted precision", "91.0%", lambda R: pct(R.cat_metrics()["precision_w"]), VAL)
add("val-rw", "Results, validation", "weighted recall", "90.4%", lambda R: pct(R.cat_metrics()["recall_w"]), VAL)
add("val-fw", "Results, validation", "weighted F1", "90.4%", lambda R: pct(R.cat_metrics()["f1_w"]), VAL)
add("val-mcc", "Results, validation", "MCC", "0.889", lambda R: R.cat_metrics()["mcc"], VAL)
add("val-kappa", "Results, validation", "Cohen's kappa", "0.888", lambda R: R.cat_metrics()["kappa"], VAL)
add("val-bal", "Results, validation", "balanced accuracy", "92.9%", lambda R: pct(R.cat_metrics()["balanced"]), VAL)
add("val-f1m", "Results, validation", "macro-averaged F1", "0.910", lambda R: R.cat_metrics()["f1_macro"], VAL)
add("val-err1", "Results, validation", "dysmorphic called internal malformation", "20",
    lambda R: R.cat_metrics()["confusion"].loc["Dysmorphic feature", "Internal physical malformations"], VAL)
add("val-err2", "Results, validation", "dysmorphic called impaired mobility", "16",
    lambda R: R.cat_metrics()["confusion"].loc["Dysmorphic feature", "Impaired mobility"], VAL)
add("val-prec-lo", "Results, validation", "lowest per-category precision", "73%", lambda R: pct(min(v["precision"] for v in R.cat_metrics()["per_class"].values())), VAL)
add("val-prec-hi", "Results, validation", "highest per-category precision", "100%", lambda R: pct(max(v["precision"] for v in R.cat_metrics()["per_class"].values())), VAL)
add("val-rec-lo", "Results, validation", "lowest per-category recall", "67%", lambda R: pct(min(v["recall"] for v in R.cat_metrics()["per_class"].values())), VAL)
add("val-rec-hi", "Results, validation", "highest per-category recall", "100%", lambda R: pct(max(v["recall"] for v in R.cat_metrics()["per_class"].values())), VAL)
add("val-lowest", "Results, validation", "category with the lowest precision and recall", "NFC",
    lambda R: (lambda pc: "NFC" if min(pc, key=lambda k: pc[k]["precision"]) == min(pc, key=lambda k: pc[k]["recall"]) == "NFC" else "other")(R.cat_metrics()["per_class"]), VAL)
add("val-tacc", "Results, validation", "tier accuracy", "91.2%", lambda R: pct(R.score("tier", "main run", "accuracy")), SCORES)
add("val-tacc-lo", "Results, validation", "tier accuracy, CI low", "89.2%", lambda R: pct(R.score("tier", "main run", "accuracy", "ci95_low")), SCORES)
add("val-tacc-hi", "Results, validation", "tier accuracy, CI high", "92.9%", lambda R: pct(R.score("tier", "main run", "accuracy", "ci95_high")), SCORES)
add("val-qwk", "Results, validation", "quadratic-weighted kappa", "0.926", lambda R: R.score("tier", "main run", "quadratic weighted kappa"), SCORES)
add("val-qwk-lo", "Results, validation", "quadratic-weighted kappa, CI low", "0.900", lambda R: R.score("tier", "main run", "quadratic weighted kappa", "ci95_low"), SCORES)
add("val-qwk-hi", "Results, validation", "quadratic-weighted kappa, CI high", "0.947", lambda R: R.score("tier", "main run", "quadratic weighted kappa", "ci95_high"), SCORES)
add("val-spear", "Results, validation", "Spearman correlation (tier)", "0.918", lambda R: R.tier_metrics()["spearman"], VAL)
add("val-kendall", "Results, validation", "Kendall's tau (tier)", "0.897", lambda R: R.tier_metrics()["kendall"], VAL)
add("val-mae", "Results, validation", "mean absolute error (tiers)", "0.129", lambda R: R.tier_metrics()["mae"], VAL)
add("val-within1", "Results, validation", "predictions within one tier", "97.1%", lambda R: pct(R.tier_metrics()["within1"]), VAL)
add("val-tf1-lo", "Results, validation", "lowest per-tier F1 (Tier 4)", "0.867", lambda R: R.tier_metrics()["f1"][4], VAL)
add("val-tf1-lo-which", "Results, validation", "tier with the lowest F1", "Tier 4", lambda R: f"Tier {min(R.tier_metrics()['f1'], key=R.tier_metrics()['f1'].get)}", VAL)
add("val-tf1-hi", "Results, validation", "highest per-tier F1 (non-tiered)", "0.970", lambda R: R.tier_metrics()["f1"][5], VAL)
add("val-tf1-hi-which", "Results, validation", "class with the highest F1", "non-tiered",
    lambda R: "non-tiered" if max(R.tier_metrics()["f1"], key=R.tier_metrics()["f1"].get) == 5 else "other", VAL)
add("val-cacc-lo", "Results, validation", "lowest category accuracy by tier (Tier 3)", "86.6%", lambda R: pct(R.tier_metrics()["cat_acc_by_tier"][3]), VAL)
add("val-cacc-hi", "Results, validation", "highest category accuracy by tier (Tier 1)", "100%", lambda R: pct(R.tier_metrics()["cat_acc_by_tier"][1]), VAL)
add("val-cacc-which", "Results, validation", "tiers with the lowest and highest category accuracy", "Tier 3, Tier 1",
    lambda R: (lambda a: f"Tier {min(a, key=a.get)}, Tier {max(a, key=a.get)}")(R.tier_metrics()["cat_acc_by_tier"]), VAL)

# ================================================================== Results: no-retrieval baseline
add("base-acc", "Results, baseline", "baseline category accuracy", "73.2%", lambda R: pct(R.score("category", "baseline", "accuracy")), SCORES)
add("base-acc-lo", "Results, baseline", "baseline category accuracy, CI low", "70.2%", lambda R: pct(R.score("category", "baseline", "accuracy", "ci95_low")), SCORES)
add("base-acc-hi", "Results, baseline", "baseline category accuracy, CI high", "76.1%", lambda R: pct(R.score("category", "baseline", "accuracy", "ci95_high")), SCORES)
add("base-tacc", "Results, baseline", "baseline tier accuracy", "72.2%", lambda R: pct(R.score("tier", "baseline", "accuracy")), SCORES)
add("base-tacc-lo", "Results, baseline", "baseline tier accuracy, CI low", "69.0%", lambda R: pct(R.score("tier", "baseline", "accuracy", "ci95_low")), SCORES)
add("base-tacc-hi", "Results, baseline", "baseline tier accuracy, CI high", "75.1%", lambda R: pct(R.score("tier", "baseline", "accuracy", "ci95_high")), SCORES)
add("base-qwk", "Results, baseline", "baseline quadratic-weighted kappa", "0.678", lambda R: R.score("tier", "baseline", "quadratic weighted kappa"), SCORES)
add("diff-cat", "Results, baseline", "category accuracy difference (points)", "17.3", lambda R: pct(R.score("category", "main minus baseline", "accuracy")), SCORES)
add("diff-cat-lo", "Results, baseline", "category difference, CI low", "14.5", lambda R: pct(R.score("category", "main minus baseline", "accuracy", "ci95_low")), SCORES)
add("diff-cat-hi", "Results, baseline", "category difference, CI high", "20.0", lambda R: pct(R.score("category", "main minus baseline", "accuracy", "ci95_high")), SCORES)
add("mcn-cat-p", "Results, baseline", "McNemar P (category)", "1.7 × 10⁻³⁰", lambda R: mcnemar_p(R, "category"), SCORES)
add("diff-tier", "Results, baseline", "tier accuracy difference (points)", "19.0", lambda R: pct(R.score("tier", "main minus baseline", "accuracy")), SCORES)
add("diff-tier-lo", "Results, baseline", "tier difference, CI low", "16.0", lambda R: pct(R.score("tier", "main minus baseline", "accuracy", "ci95_low")), SCORES)
add("diff-tier-hi", "Results, baseline", "tier difference, CI high", "21.8", lambda R: pct(R.score("tier", "main minus baseline", "accuracy", "ci95_high")), SCORES)
add("mcn-tier-p", "Results, baseline", "McNemar P (tier)", "1.2 × 10⁻³⁷", lambda R: mcnemar_p(R, "tier"), SCORES)

# ================================================================== Results: clinical landscape (Figure 3)
for t, n, p in (("2", "2,966", "27.8%"), ("3", "2,351", "22.1%"), ("4", "1,517", "14.2%"), ("1", "272", "2.6%"), ("NFC", "3,546", "33.3%")):
    label = "non-tiered" if t == "NFC" else f"Tier {t}"
    add(f"land-t{t}-n", "Results, landscape", f"{label} terms", n, lambda R, t=t: R.count_line(tier_counts(R), t), REPORT)
    add(f"land-t{t}-p", "Results, landscape", f"{label} share", p, lambda R, t=t: 100 * R.count_line(tier_counts(R), t) / len(R.cls), REPORT)
for cat, n in (("NFC - Clinical Signs", "2,414"), ("Out of Scope: Acquired/Multifactorial", "796"), ("NFC (Not Further Classified)", "331"),
               ("Internal physical malformations", "2,256"), ("Dysmorphic feature", "2,110"), ("Shortened life span: infancy", "129"),
               ("Shortened life span: childhood/adolescence", "50"), ("Shortened life span: adulthood", "61")):
    add(f"land-{cat[:18]}", "Results, landscape", f"terms: {cat.replace('NFC - Clinical Signs', 'Clinical Sign or Laboratory Abnormality')}", n,
        lambda R, cat=cat: R.count_line(cat_counts(R), cat), REPORT)
add("land-nocat", "Results, landscape", "terms with no category recorded", "5", lambda R: int(R.cls.classification_category.isna().sum()), CLS)
for hid, cat in (("HP:0002315", "Clinical Sign or Laboratory Abnormality"), ("HP:0004396", "Clinical Sign or Laboratory Abnormality"),
                 ("HP:0003236", "Clinical Sign or Laboratory Abnormality"), ("HP:0033235", "Out of Scope: Acquired/Multifactorial"),
                 ("HP:0031446", "Out of Scope: Acquired/Multifactorial"), ("HP:0032448", "Out of Scope: Acquired/Multifactorial"),
                 ("HP:0033454", "NFC (Not Further Classified)"), ("HP:0000118", "NFC (Not Further Classified)"), ("HP:0001470", "NFC (Not Further Classified)")):
    add(f"land-ex-{hid}", "Results, landscape", f"category of example {hid}", cat,
        lambda R, hid=hid: str(R.cls.set_index("hpo_id").loc[hid, "classification_category"]), CLS)
add("land-dys-t3", "Results, landscape", "dysmorphic features in Tier 3", "41.6%",
    lambda R: float(re.search(r"Dysmorphic feature:\n  Tier 3: ([\d.]+)%", R.report).group(1)), REPORT)
add("land-dys-t4", "Results, landscape", "dysmorphic features in Tier 4", "58.4%",
    lambda R: float(re.search(r"Dysmorphic feature:\n  Tier 3: [\d.]+%\n  Tier 4: ([\d.]+)%", R.report).group(1)), REPORT)
for cat, tier in (("Shortened life span: infancy", "Tier 1"), ("Shortened life span: childhood/adolescence", "Tier 1"), ("Shortened life span: adulthood", "Tier 2")):
    add(f"land-only-{cat[-10:]}", "Results, landscape", f"tier of all '{cat}' terms", tier,
        lambda R, cat=cat: (lambda m: f"Tier {m.group(1)}" if m else "mixed")(re.search(re.escape(cat) + r":\n  Tier (\w+): 100\.0%", R.report)), REPORT)

# ================================================================== Results: evidence quality (Figure 4)
for tier, kind, p in ((1, "Direct Support", "74.0%"), (1, "Unsupported", "5.7%"), (4, "Unsupported", "14.1%"), (4, "Direct Support", "72.9%"),
                      (2, "Direct Support", "73.6%"), (3, "Direct Support", "71.7%"), (2, "Unsupported", "6.7%"), (3, "Unsupported", "7.2%")):
    add(f"evid-t{tier}-{kind[:6]}", "Results, evidence", f"Tier {tier} {kind.lower()}", p, lambda R, tier=tier, kind=kind: evidence_tier(R, tier, kind), REPORT)
add("evid-cat-lo", "Results, evidence", "lowest direct evidence by category (hearing)", "66.6%", lambda R: evidence_cat(R)["Sensory Impairment – Hearing"][0], REPORT)
add("evid-cat-hi", "Results, evidence", "highest direct evidence by category (intellectual disability)", "79.7%", lambda R: evidence_cat(R)["Intellectual disability"][0], REPORT)
add("evid-cat-which", "Results, evidence", "categories with the lowest and highest direct evidence", "Sensory Impairment – Hearing, Intellectual disability",
    lambda R: (lambda t: f"{min(t, key=lambda k: t[k][0])}, {max(t, key=lambda k: t[k][0])}")(evidence_cat(R)), REPORT)
add("evid-cat-70", "Results, evidence", "categories with direct evidence above 70%", "9", lambda R: sum(v[0] > 70 for v in evidence_cat(R).values()), REPORT)
add("evid-cat-n", "Results, evidence", "categories", "13", lambda R: len(evidence_cat(R)), REPORT)
add("evid-cat-unsup", "Results, evidence", "categories with unsupported claims below 10%", "13", lambda R: sum(v[3] < 10 for v in evidence_cat(R).values()), REPORT)
for tier, med in ((1, "0.97"), (2, "0.97"), (3, "0.96"), (4, "0.95")):
    add(f"evid-med-t{tier}", "Results, evidence", f"median support score, Tier {tier}", med,
        lambda R, tier=tier: R.count_line(R.block("Median support score by tier", "3B."), str(tier)), REPORT)
add("evid-t4-low", "Results, evidence", "Tier 4 support scores below 0.5", "17.1%",
    lambda R: (lambda d: 100 * (d < 0.5).sum() / d.notna().sum())(R.pt[R.pt.classification_tier == "4"].statement_support_score), PT)
add("evid-iqr-lo", "Results, evidence", "tightest support-score distribution (intellectual disability), IQR", "0.07",
    lambda R: support_iqr(R)["Intellectual disability"], PT)
add("evid-iqr-lo-which", "Results, evidence", "category with the tightest support-score distribution", "Intellectual disability",
    lambda R: support_iqr(R).idxmin(), PT)
add("evid-iqr-hi", "Results, evidence", "widest support-score distribution (dysmorphic features), IQR", "0.31",
    lambda R: support_iqr(R)["Dysmorphic feature"], PT)
add("evid-iqr-hi-which", "Results, evidence", "category with the widest support-score distribution", "Dysmorphic feature",
    lambda R: support_iqr(R).idxmax(), PT)
add("evid-kw-tier", "Results, evidence", "Kruskal-Wallis H(3), support score by tier", "39.88",
    lambda R: float(R.stat(r"Kruskal-Wallis \(Support Score across Tiers\): H\(3\)=([\d.]+)")[0]), REPORT)
add("evid-kw-cat", "Results, evidence", "Kruskal-Wallis H(12), support score by category", "30.93",
    lambda R: float(R.stat(r"Kruskal-Wallis \(Support Score across Categories, Figure 4D\): H\(12\)=([\d.]+)")[0]), REPORT)
add("evid-kw-cat-p", "Results, evidence", "P, support score by category", "0.0020",
    lambda R: exact_p(R, "report: Kruskal-Wallis support score by category"), EXACTP)
add("evid-kw-tier-p", "Results, evidence", "P, support score by tier", "1.1 × 10⁻⁸",
    lambda R: exact_p(R, "report: Kruskal-Wallis support score by tier"), EXACTP)

# ================================================================== Results: quality of life (Figure 5, Table 1)
add("qol-any-p", "Results, QoL", "phenotypes with at least one affected QoL domain", "61.9%",
    lambda R: 100 * float(R.stat(r"QoL affected vs not affected: mean [\d.]+ \(n=(\d+)\)")[0]) / len(R.cls), REPORT)
add("qol-any-n", "Results, QoL", "phenotypes with an affected QoL domain", "6,593", lambda R: float(R.stat(r"QoL affected vs not affected: mean [\d.]+ \(n=(\d+)\)")[0]), REPORT)
add("qol-cs-aff", "Results, QoL", "Clinical Sign or Laboratory Abnormality terms rated QoL-affected", "963",
    lambda R: int(((R.pt.classification_category == "NFC - Clinical Signs") & (R.pt.QoL_assessment == "Affected")).sum()), PT)
add("qol-tiered-n", "Results, QoL", "tiered phenotypes", "7,106", lambda R: int(R.pt.classification_tier.isin(TIERED).sum()), PT)
add("qol-tiered-aff", "Results, QoL", "tiered phenotypes rated QoL-affected", "5,630",
    lambda R: int((R.pt.classification_tier.isin(TIERED) & (R.pt.QoL_assessment == "Affected")).sum()), PT)
add("qol-tiered-p", "Results, QoL", "share of tiered phenotypes rated QoL-affected", "79.2%",
    lambda R: 100 * (R.pt[R.pt.classification_tier.isin(TIERED)].QoL_assessment == "Affected").mean(), PT)
add("qol-chi2-tier", "Results, QoL", "chi-square, QoL domain by tier", "1,020.84", lambda R: float(R.stat(r"Chi-square \(QoL Domains vs Tier\): chi2=([\d.]+), df=12")[0]), REPORT)
TABLE1 = {1: ("272", "47.7", "31.0", "13.0", "4.2", "4.2"), 2: ("2,966", "71.9", "9.3", "15.1", "1.8", "1.8"),
          3: ("2,351", "49.8", "19.3", "9.2", "10.2", "11.5"), 4: ("1,517", "30.9", "2.5", "4.9", "38.3", "23.5")}
for tier, vals in TABLE1.items():
    add(f"t1-{tier}-n", "Table 1", f"Tier {tier}: phenotypes", vals[0], lambda R, tier=tier: R.count_line(tier_counts(R), str(tier)), REPORT)
    for dom, v in zip(("Physical Functioning", "Cognitive/School", "Physical Symptoms", "Emotional Impact", "Social Functioning"), vals[1:]):
        add(f"t1-{tier}-{dom[:6]}", "Table 1", f"Tier {tier}: {dom} %", v, lambda R, tier=tier, dom=dom: qol_share(R, tier, dom), REPORT)
add("t1-t4-nodomain", "Table 1", "Tier 4 phenotypes with no affected domain", "1,473",
    lambda R: int(((R.pt.classification_tier == "4") & (R.pt.num_qol_domains == 0)).sum()), PT)
add("qol-t4-infert", "Results, QoL", "QoL-affected Tier 4 phenotypes (all infertility)", "44", lambda R: qol_cat(R)["Infertility"]["Affected"], REPORT)
for cat, dom, v in (("Internal physical malformations", "Physical Functioning", "65.1"), ("Dysmorphic feature", "Physical Functioning", "63.0"),
                    ("Sensory impairment: touch, others", "Physical Functioning", "80.0"), ("Intellectual disability", "Cognitive/School", "57.9"),
                    ("Shortened life span: infancy", "Physical Functioning", "62.9"), ("Shortened life span: infancy", "Physical Symptoms", "22.6"),
                    ("Shortened life span: childhood/adolescence", "Cognitive/School", "21.8"), ("Shortened life span: childhood/adolescence", "Physical Functioning", "56.3"),
                    ("Mental illness", "Cognitive/School", "31.6"), ("Mental illness", "Emotional Impact", "31.4"),
                    ("Sensory Impairment – Vision", "Physical Functioning", "44.9"), ("Sensory Impairment – Vision", "Cognitive/School", "40.6"),
                    ("Sensory Impairment – Hearing", "Cognitive/School", "44.3"), ("Sensory Impairment – Hearing", "Social Functioning", "42.7")):
    add(f"qol-cat-{cat[:14]}-{dom[:6]}", "Results, QoL", f"{cat}: {dom} share of domain mappings %", v,
        lambda R, cat=cat, dom=dom: qol_cat_share(R, cat, dom), PT)
add("qol-chi2-cat", "Results, QoL", "chi-square, QoL domain by category", "3,621.84", lambda R: R.tests.find(48, col_has="Physical Functioning")["chi2"], "recorded while drawing Figure 5B")

# ================================================================== Results: management (Figure 6)
add("mgmt-0", "Results, management", "interventions with no QoL domain", "0.89", lambda R: float(R.stat(r"0 domains: ([\d.]+)")[0]), REPORT)
add("mgmt-n0", "Results, management", "phenotypes with no QoL domain listed", "4,198", lambda R: int((R.pt.num_qol_domains == 0).sum()), PT)
for k, v in ((1, "1.53"), (2, "1.73"), (3, "1.72"), (4, "1.95")):
    add(f"mgmt-{k}", "Results, management", f"mean intervention types with {k} QoL domain(s)", v,
        lambda R, k=k: R.pt[R.pt.num_qol_domains == k].num_interventions.mean(), PT)
add("mgmt-n5", "Results, management", "phenotypes with all five QoL domains", "3", lambda R: int((R.pt.num_qol_domains == 5).sum()), PT)
add("mgmt-5", "Results, management", "interventions with five QoL domains", "2.67", lambda R: float(R.stat(r"5 domains: ([\d.]+)")[0]), REPORT)
add("mgmt-r2", "Results, management", "R-squared, interventions by QoL breadth", "0.879", lambda R: float(R.stat(r"Linear Regression: R2=([\d.]+)")[0]), REPORT)
add("mgmt-r2-p", "Results, management; Methods, statistics", "P, regression", "0.0057",
    lambda R: exact_p(R, "report: linear regression of mean interventions on QoL domain count"), EXACTP)
for dom, n in (("Physical Functioning", "5,411"), ("Physical Symptoms", "1,349"), ("Cognitive/School", "1,312"), ("Emotional Impact", "606"), ("Social Functioning", "575")):
    add(f"mgmt-dom-{dom[:6]}", "Results, management", f"phenotypes affecting {dom}", n, lambda R, dom=dom: R.count_line(R.block("6B. QOL DOMAIN COUNTS", "Interventions, QoL"), dom), REPORT)
add("mgmt-aff", "Results, management", "mean interventions, QoL affected", "1.55", lambda R: float(R.stat(r"QoL affected vs not affected: mean ([\d.]+)")[0]), REPORT)
add("mgmt-unaff", "Results, management", "mean interventions, QoL not affected", "0.95", lambda R: float(R.stat(r"QoL affected vs not affected: mean [\d.]+ \(n=\d+\) vs ([\d.]+)")[0]), REPORT)
add("mgmt-mw-p", "Results, management", "Mann-Whitney P", "1.7 × 10⁻²⁷⁶",
    lambda R: exact_p(R, "report: Mann-Whitney interventions, QoL affected vs not affected"), EXACTP)
for tier, mean, sd in ((1, "1.71", "0.92"), (3, "1.55", "0.80"), (2, "1.48", "0.89"), (4, "0.90", "0.62")):
    add(f"mgmt-t{tier}-mean", "Results, management", f"Tier {tier} mean interventions", mean, lambda R, tier=tier: mgmt_tier(R, tier, "mean"), REPORT)
    add(f"mgmt-t{tier}-sd", "Results, management", f"Tier {tier} SD", sd, lambda R, tier=tier: mgmt_tier(R, tier, "std"), REPORT)
add("mgmt-kw-tier", "Results, management", "Kruskal-Wallis H(3), interventions by tier", "721.32", lambda R: float(R.stat(r"Kruskal-Wallis: H\(3\)=([\d.]+)")[0]), REPORT)
add("mgmt-life-lo", "Results, management", "lowest mean, shortened-lifespan categories", "1.89",
    lambda R: min(mgmt_cat(R, c) for c in ("Shortened life span: infancy", "Shortened life span: childhood/adolescence", "Shortened life span: adulthood")), REPORT)
add("mgmt-life-hi", "Results, management", "highest mean, shortened-lifespan categories", "2.00",
    lambda R: max(mgmt_cat(R, c) for c in ("Shortened life span: infancy", "Shortened life span: childhood/adolescence", "Shortened life span: adulthood")), REPORT)
add("mgmt-immuno", "Results, management", "mean interventions, immunodeficiency/cancer", "1.70", lambda R: mgmt_cat(R, "Immunodeficiency/cancer"), REPORT)
add("mgmt-sens-lo", "Results, management", "lowest mean, sensory impairments", "1.45",
    lambda R: min(mgmt_cat(R, c) for c in ("Sensory Impairment – Hearing", "Sensory Impairment – Vision", "Sensory impairment: touch, others")), REPORT)
add("mgmt-sens-hi", "Results, management", "highest mean, sensory impairments", "1.68",
    lambda R: max(mgmt_cat(R, c) for c in ("Sensory Impairment – Hearing", "Sensory Impairment – Vision", "Sensory impairment: touch, others")), REPORT)
add("mgmt-dys", "Results, management", "mean interventions, dysmorphic features (fewest)", "1.08", lambda R: mgmt_cat(R, "Dysmorphic feature"), REPORT)
add("mgmt-kw-cat", "Results, management", "Kruskal-Wallis H(12), interventions by category", "569.04", lambda R: float(R.stat(r"Kruskal-Wallis: H\(12\)=([\d.]+)")[0]), REPORT)

# ================================================================== Results: QoL status (Figure S2, Supplementary Table 2)
SUPP2 = (("Internal physical malformations", "2,256", "2,218", "98.3"), ("Impaired mobility", "687", "687", "100.0"),
         ("Shortened life span: adulthood", "61", "61", "100.0"), ("Shortened life span: infancy", "129", "129", "100.0"),
         ("Shortened life span: childhood/adolescence", "50", "50", "100.0"), ("Intellectual disability", "93", "93", "100.0"),
         ("Sensory impairment: touch, others", "65", "64", "98.5"), ("Sensory Impairment – Vision", "556", "524", "94.2"),
         ("Mental illness", "312", "292", "93.6"), ("Sensory Impairment – Hearing", "71", "64", "90.1"),
         ("Immunodeficiency/cancer", "623", "526", "84.4"), ("Infertility", "93", "44", "47.3"), ("Dysmorphic feature", "2,110", "878", "41.6"))
for cat, n, aff, p in SUPP2:
    add(f"s2-{cat[:14]}-n", "Supplementary Table 2", f"{cat}: phenotypes", n, lambda R, cat=cat: category_total(R, cat), REPORT)
    add(f"s2-{cat[:14]}-aff", "Supplementary Table 2", f"{cat}: QoL affected", aff, lambda R, cat=cat: qol_cat(R)[cat]["Affected"], REPORT)
    add(f"s2-{cat[:14]}-p", "Supplementary Table 2", f"{cat}: % QoL affected", p, lambda R, cat=cat: 100 * qol_cat(R)[cat]["Affected"] / category_total(R, cat), REPORT)
add("s2-chi2", "Results, QoL status", "chi-square, QoL status by category", "2,785.03", lambda R: float(R.stat(r"5A\. QOL ASSESSMENT BY CATEGORY[\s\S]*?Chi-square: chi2=([\d.]+), df=12")[0]), REPORT)
add("s2-chi2-mgmt", "Results, QoL status", "chi-square, management by QoL status", "2,312.92", lambda R: R.tests.find(7, row_has="Affected")["chi2"], "recorded while drawing Figure S2C")

for status, kind, v in (("Affected", "Surgical", "30.6%"), ("Affected", "Pharmacological", "24.2%"), ("Affected", "Monitoring", "21.5%"),
                        ("Affected", "Therapeutic Support", "16.7%"), ("Affected", "No Treatment", "1.7%"),
                        ("Not Affected", "No Treatment", "24.3%"), ("Not Affected", "Monitoring", "24.0%"), ("Not Affected", "Pharmacological", "23.1%")):
    add(f"s2-mgmt-{status[:3]}-{kind[:8]}", "Results, QoL status", f"QoL {status.lower()}: {kind} share of intervention types", v,
        lambda R, status=status, kind=kind: mgmt_qol_share(R, status, kind), PT)

# ================================================================== Results: operational (Figure S3)
add("op-scored", "Results, operational", "classifications with a support score", "10,651", lambda R: int(R.cls.statement_support_score.notna().sum()), CLS)
add("op-mean", "Results, operational", "mean support score", "0.82", lambda R: described(R, "Statement Support Score Statistics:", "mean"), REPORT)
add("op-sd", "Results, operational", "SD support score", "0.27", lambda R: described(R, "Statement Support Score Statistics:", "std"), REPORT)
add("op-median", "Results, operational", "median support score", "0.96", lambda R: described(R, "Statement Support Score Statistics:", "50%"), REPORT)
add("op-max", "Results, operational", "terms scoring 1.0", "34.9%", lambda R: float(R.stat(r"share scoring 1\.0 = ([\d.]+)%")[0]), REPORT)
add("op-steps", "Results, operational", "mean ReAct steps", "6.63", lambda R: R.cls.react_steps.mean(), CLS)
add("op-steps-sd", "Results, operational", "SD ReAct steps", "1.60", lambda R: R.cls.react_steps.std(), CLS)
add("op-steps6", "Results, operational", "terms that took 6 steps", "78%", lambda R: 100 * (R.cls.react_steps == 6).mean(), CLS)
add("op-steps10", "Results, operational", "terms that took 10 steps", "17%", lambda R: 100 * (R.cls.react_steps == 10).mean(), CLS)
add("op-cost", "Results, operational", "mean cost per term (USD)", "$0.101", lambda R: R.cls.total_cost_usd.mean(), CLS)
add("op-cost-min", "Results, operational", "lowest cost per term (USD)", "$0.036", lambda R: R.cls.total_cost_usd.min(), CLS)
add("op-cost-max", "Results, operational", "highest cost per term (USD)", "$0.197", lambda R: R.cls.total_cost_usd.max(), CLS)
add("op-cost-sd", "Results, operational", "SD cost (USD)", "$0.018", lambda R: R.cls.total_cost_usd.std(), CLS)
add("op-cost-q1", "Results, operational", "cost, 25th percentile (USD)", "$0.089", lambda R: R.cls.total_cost_usd.quantile(0.25), CLS)
add("op-cost-q3", "Results, operational", "cost, 75th percentile (USD)", "$0.112", lambda R: R.cls.total_cost_usd.quantile(0.75), CLS)
add("op-r", "Results, operational", "Pearson r, cost and steps", "0.40", lambda R: float(R.stat(r"Cost vs React Steps:\n  Pearson r=([\d.]+)")[0]), REPORT)
add("op-flag", "Results, operational", "classifications flagged for review", "1,169", lambda R: int((R.cls.review_status != "OK").sum()), CLS)
add("op-flag-p", "Results, operational", "share flagged", "11.0%", lambda R: 100 * (R.cls.review_status != "OK").mean(), CLS)

# ================================================================== Results: gene level (Figure 7, Table 2)
add("gl-pairs", "Results, gene level", "gene-disease pairs", "8,467", lambda R: R.log_int(r"Pairs kept: ([\d,]+) \(")[0], GLOG)
add("gl-mondo", "Results, gene level", "pairs from MONDO", "6,458", lambda R: R.log_int(r"sources \{'MONDO_germline': (\d+)")[0], GLOG)
add("gl-gencc", "Results, gene level", "pairs added from GenCC", "2,009", lambda R: R.log_int(r"'GenCC_DSM_only': (\d+)\}")[0], GLOG)
add("gl-limited", "Results, gene level", "pairs with only Limited GenCC support", "359", lambda R: R.log_int(r"flagged Limited-only (\d+)")[0], GLOG)
add("gl-analysed", "Results, gene level", "pairs analysed", "8,108", lambda R: R.log_int(r"Pairs in the analysis: ([\d,]+) of")[0], GLOG)
add("gl-labelled", "Results, gene level", "pairs with a severity class", "6,709", lambda R: R.log_int(r"labelled ([\d,]+);")[0], GLOG)
for sev, n, p in (("Profound", "2,436", "36.3%"), ("Severe", "2,518", "37.5%"), ("Moderate", "1,433", "21.4%"), ("Mild", "322", "4.8%")):
    add(f"gl-{sev}-n", "Results, gene level", f"{sev} pairs", n, lambda R, sev=sev: R.f7["7A"].set_index("severity").loc[sev, "pairs"], F7)
    add(f"gl-{sev}-p", "Results, gene level", f"{sev} share", p, lambda R, sev=sev: 100 * R.f7["7A"].set_index("severity").loc[sev, "pairs"] / R.f7["7A"].pairs.sum(), F7)
add("gl-sp-n", "Results, gene level", "Severe or Profound pairs", "4,954", lambda R: R.f7["7A"].set_index("severity").loc[["Profound", "Severe"], "pairs"].sum(), F7)
add("gl-systems", "Results, gene level", "body systems", "23", lambda R: len(R.f7["7B_Table2"]), F7)
TABLE2 = (("Abnormality of prenatal development or birth", "620", "69.5", "25.3", "3.5", "1.6"), ("Growth abnormality", "2,223", "57.2", "38.3", "4.3", "0.2"),
          ("Abnormality of the nervous system", "4,247", "55.1", "37.0", "7.4", "0.5"), ("Abnormality of the breast", "215", "54.9", "35.3", "9.3", "0.5"),
          ("Abnormality of head or neck", "3,294", "54.7", "35.5", "8.0", "1.8"), ("Abnormality of the ear", "1,750", "53.3", "35.0", "11.5", "0.2"),
          ("Abnormality of the eye", "2,906", "53.2", "35.1", "11.7", "0.1"), ("Abnormality of the digestive system", "2,517", "52.9", "38.1", "8.0", "1.0"),
          ("Abnormality of the musculoskeletal system", "4,249", "52.0", "41.0", "6.7", "0.3"), ("Abnormality of the respiratory system", "1,522", "50.1", "41.5", "8.2", "0.3"),
          ("Abnormality of the voice", "259", "49.4", "48.3", "2.3", "0.0"), ("Abnormality of limbs", "2,273", "48.4", "45.8", "5.4", "0.4"),
          ("Abnormality of the genitourinary system", "2,165", "46.1", "36.8", "10.6", "6.6"), ("Abnormal cellular phenotype", "641", "42.7", "44.9", "11.4", "0.9"),
          ("Abnormality of the integument", "2,310", "42.0", "38.1", "16.4", "3.5"), ("Abnormality of the endocrine system", "809", "41.4", "40.4", "13.3", "4.8"),
          ("Abnormality of the cardiovascular system", "2,269", "41.1", "44.6", "12.0", "2.3"), ("Abnormality of metabolism / homeostasis", "2,379", "40.8", "41.0", "14.1", "4.1"),
          ("Abnormality of the immune system", "1,583", "38.9", "42.5", "17.7", "0.9"), ("Constitutional symptom", "719", "30.5", "56.5", "11.7", "1.4"),
          ("Abnormality of blood and blood-forming tissues", "1,226", "29.4", "43.1", "19.7", "7.9"), ("Neoplasm", "502", "23.5", "44.2", "31.5", "0.8"),
          ("Abnormality of the thoracic cavity", "14", "21.4", "78.6", "0.0", "0.0"))
for name, n, *ps in TABLE2:
    add(f"t2-{name[:20]}-n", "Table 2", f"{name}: pairs", n, lambda R, name=name: R.f7row("7B_Table2", "body_system", name).pairs, F7)
    for sev, p in zip(("Profound", "Severe", "Moderate", "Mild"), ps):
        add(f"t2-{name[:20]}-{sev}", "Table 2", f"{name}: {sev} %", p,
            lambda R, name=name, sev=sev: (lambda r: 100 * r[sev] / r.pairs)(R.f7row("7B_Table2", "body_system", name)), F7)
for inh, p, a, b in (("X-linked", "49.1%", "172", "350"), ("Autosomal recessive", "42.6%", "1,493", "3,507"),
                     ("Autosomal dominant", "28.1%", "711", "2,531"), ("Complex", "19.4%", "55", "283")):
    add(f"inh-{inh[:12]}-p", "Results, gene level", f"{inh} pairs Profound", p, lambda R, inh=inh: (lambda r: 100 * r.Profound / r.pairs)(R.f7row("7C", "inheritance", inh)), F7)
    add(f"inh-{inh[:12]}-a", "Results, gene level", f"{inh} Profound pairs", a, lambda R, inh=inh: R.f7row("7C", "inheritance", inh).Profound, F7)
    add(f"inh-{inh[:12]}-b", "Results, gene level", f"{inh} pairs", b, lambda R, inh=inh: R.f7row("7C", "inheritance", inh).pairs, F7)
add("inh-ar-sp", "Results, gene level", "autosomal recessive pairs Severe or Profound", "2,767", lambda R: (lambda r: r.Profound + r.Severe)(R.f7row("7C", "inheritance", "Autosomal recessive")), F7)
add("inh-xl-sp", "Results, gene level", "X-linked pairs Severe or Profound", "286", lambda R: (lambda r: r.Profound + r.Severe)(R.f7row("7C", "inheritance", "X-linked")), F7)
add("inh-xl-sp-p", "Results, gene level", "X-linked pairs Severe or Profound, share", "81.7%", lambda R: (lambda r: 100 * (r.Profound + r.Severe) / r.pairs)(R.f7row("7C", "inheritance", "X-linked")), F7)
for gs, n, sp in (("All genes (base rate)", "4,531", "79.0%"), ("ACMG carrier screening", "112", "90.2%"),
                  ("Mackenzie's Mission", "1,242", "90.2%"), ("Human Intolerome (fetal effect)", "736", "92.9%"), ("ACMG SF v3.2", "81", "74.1%")):
    add(f"list-{gs[:10]}-n", "Results, gene level", f"{gs}: genes with a matched condition", n, lambda R, gs=gs: R.f7row("7D", "gene_set", gs).genes_labelled, F7)
    add(f"list-{gs[:10]}-sp", "Results, gene level", f"{gs}: Severe or Profound", sp, lambda R, gs=gs: (lambda r: 100 * (r.Profound + r.Severe) / r.genes_labelled)(R.f7row("7D", "gene_set", gs)), F7)
for gs, p in (("All genes (base rate)", "26.5%"), ("Mackenzie's Mission", "34.1%"), ("ACMG carrier screening", "42.9%"),
              ("ACMG SF v3.2", "43.2%"), ("Human Intolerome (fetal effect)", "52.2%")):
    add(f"early-{gs[:10]}", "Results, gene level", f"{gs}: shortened life span in infancy or childhood", p,
        lambda R, gs=gs: (lambda r: 100 * r.early_death_genes / r.genes_labelled)(R.f7row("7E", "gene_set", gs)), F7)

# ================================================================== Discussion
add("dis-unsup-hi", "Discussion", "unsupported claims, highest tier", "14.1%", lambda R: max(evidence_tier(R, t, "Unsupported") for t in (1, 2, 3, 4)), REPORT)
add("dis-unsup-lo", "Discussion", "unsupported claims, lowest tier", "5.7%", lambda R: min(evidence_tier(R, t, "Unsupported") for t in (1, 2, 3, 4)), REPORT)
for label, key, n, sp in (("lethal only before birth", "Intolerome, lethal before birth only", "177", "89.3%"),
                          ("lethal only after birth", "Intolerome, lethal after birth only", "395", "93.4%")):
    add(f"gl-intol-{label[-12:]}-n", "Results, gene level; Discussion", f"Human Intolerome genes {label}", n, lambda R, key=key: R.f7row("Sensitivity", "set", key).n, F7)
    add(f"gl-intol-{label[-12:]}-sp", "Results, gene level; Discussion", f"Human Intolerome genes {label}: Severe or Profound", sp,
        lambda R, key=key: R.f7row("Sensitivity", "set", key).Severe_or_Profound_pct, F7)
add("dis-diff", "Discussion", "category accuracy lost without PubMed evidence (points)", "17.3", lambda R: pct(R.score("category", "main minus baseline", "accuracy")), SCORES)
add("dis-flag", "Discussion", "classifications flagged by the verification step", "11.0%", lambda R: 100 * (R.cls.review_status != "OK").mean(), CLS)
add("dis-prenatal", "Discussion", "Intolerome genes lethal only before birth: Profound", "58.8%",
    lambda R: R.f7row("Sensitivity", "set", "Intolerome, lethal before birth only").Profound_pct, F7)
add("dis-postnatal", "Discussion", "Intolerome genes lethal only after birth: Profound", "64.1%",
    lambda R: R.f7row("Sensitivity", "set", "Intolerome, lethal after birth only").Profound_pct, F7)

# ================================================================== Methods
add("m-14", "Methods", "model-selection terms in the validation set", "14", lambda R: model_selection_terms_in_validation(R)[1], VAL)
add("m-14-acc", "Methods", "category accuracy without the 14 terms", "90.4%", lambda R: pct(model_selection_terms_in_validation(R)[2]), VAL)
add("m-abstracts", "Methods", "mean PubMed abstracts per term", "8.0", lambda R: R.cls.pubmed_results_count.mean(), CLS)
add("m-terms-first", "Methods, agent workflow", "terms classified with HPO release 2025-05-06", "10,211",
    lambda R: int((R.cls.hpo_release == "2025-05-06").sum()), CLS)
add("m-terms-later", "Methods, agent workflow", "terms added later, classified with HPO release 2026-09-01", "441",
    lambda R: int((R.cls.hpo_release == "2026-09-01").sum()), CLS)
add("m-pass", "Methods", "classifications that passed verification", "9,483", lambda R: int((R.cls.review_status == "OK").sum()), CLS)
add("m-pass-p", "Methods", "share that passed", "89.0%", lambda R: 100 * (R.cls.review_status == "OK").mean(), CLS)
add("m-psi", "Methods", "PSI of the final validation set", "0.0303", lambda R: float(re.search(r"PSI = ([\d.]+)", R.vreport).group(1)), VREPORT)
add("m-removed", "Methods", "labelled terms not among the classified terms", "15", lambda R: removed_terms(R)[0], "data/ground_truth_all_labelled.xlsx")
add("m-removed-moi", "Methods", "of these, mode-of-inheritance terms", "11", lambda R: removed_terms(R)[1], "data/ground_truth_all_labelled.xlsx")
add("m-removed-chronic", "Methods", "of these, the modifier Chronic", "1", lambda R: removed_terms(R)[2], "data/ground_truth_all_labelled.xlsx")
add("m-removed-other", "Methods", "of these, other phenotype terms", "3", lambda R: removed_terms(R)[3], "data/ground_truth_all_labelled.xlsx")
add("gl-disputed", "Results, gene level", "MONDO pairs removed for disputed or refuted GenCC evidence", "7", lambda R: R.log_int(r"Disputed/Refuted/No Known: (\d+) pairs")[0], GLOG)
add("gl-genes", "Results, gene level", "genes in the pairs kept", "5,015", lambda R: R.log_int(r"Pairs kept: [\d,]+ \(([\d,]+) genes")[0], GLOG)
add("gl-diseases", "Results, gene level", "diseases in the pairs kept", "7,639", lambda R: R.log_int(r"Pairs kept: [\d,]+ \([\d,]+ genes, ([\d,]+) diseases\)")[0], GLOG)
add("gl-unclass", "Results, gene level", "pairs without a class", "1,399", lambda R: sum(R.log_int(r"not labelled \{'no_phenotypes': (\d+), 'excluded_nfc_only': (\d+), 'no_phenotypes_after_frequency_filter': (\d+)\}")), GLOG)
add("gl-unclass-a", "Results, gene level", "of these, no phenotype annotation", "1,376", lambda R: R.log_int(r"not labelled \{'no_phenotypes': (\d+)")[0], GLOG)
add("gl-unclass-b", "Results, gene level", "of these, only NFC phenotypes", "17", lambda R: R.log_int(r"not labelled \{[^}]*'excluded_nfc_only': (\d+)")[0], GLOG)
add("gl-unclass-c", "Results, gene level", "of these, no phenotype after the frequency filter", "6", lambda R: R.log_int(r"not labelled \{[^}]*'no_phenotypes_after_frequency_filter': (\d+)")[0], GLOG)
add("gl-noinh", "Results, gene level; Figure 7C legend", "classified pairs without inheritance", "38", lambda R: R.log_int(r"pairs without inheritance annotation \(left out\): (\d+)")[0], GLOG)
add("m-list-sf", "Methods", "ACMG SF v3.2 genes", "81", lambda R: R.f7row("7D", "gene_set", "ACMG SF v3.2").genes_in_list, F7)
add("m-list-cs", "Methods", "ACMG carrier screening genes", "113", lambda R: R.f7row("7D", "gene_set", "ACMG carrier screening").genes_in_list, F7)
add("m-list-mm", "Methods", "Mackenzie's Mission genes", "1,300", lambda R: R.f7row("7D", "gene_set", "Mackenzie's Mission").genes_in_list, F7)
add("m-list-intol", "Methods", "Human Intolerome genes with a fetal effect", "824", lambda R: R.f7row("7D", "gene_set", "Human Intolerome (fetal effect)").genes_in_list, F7)
add("m-terms", "Methods", "HPO terms classified", "10,652", lambda R: len(R.cls), CLS)
for hp, what, tier in (("HP:0000202", "orofacial cleft", "3"), ("HP:0000574", "thick eyebrow", "4")):
    add(f"m-example-{hp}-cat", "Methods, QoL framework example", f"{what} ({hp}): category", "Dysmorphic feature",
        lambda R, hp=hp: R.cls.loc[R.cls.hpo_id == hp, "classification_category"].iloc[0], CLS)
    add(f"m-example-{hp}-tier", "Methods, QoL framework example", f"{what} ({hp}): tier", tier,
        lambda R, hp=hp: str(R.cls.loc[R.cls.hpo_id == hp, "classification_tier"].iloc[0]), CLS)
add("m-once", "Methods, statistics", "distinct terms (each term classified once)", "10,652", lambda R: R.cls.hpo_id.nunique(), CLS)

# ================================================================== exact P values of the other tests in Results
for id_, where, what, printed, key in (
        ("p-qol-tier", "Results, QoL", "P, QoL domain by tier (chi-square)", "6.2 × 10⁻²¹¹", "report: chi-square QoL domain x tier"),
        ("p-qol-cat", "Results, QoL", "P, QoL domain by category (chi-square)", "1.1 × 10⁻⁷³⁴", "figure: chi-square classification_category x qol_domains"),
        ("p-mgmt-tier", "Results, management", "P, interventions by tier (Kruskal-Wallis)", "5.0 × 10⁻¹⁵⁶", "report: Kruskal-Wallis interventions by tier"),
        ("p-mgmt-cat", "Results, management", "P, interventions by category (Kruskal-Wallis)", "4.3 × 10⁻¹¹⁴", "report: Kruskal-Wallis interventions by category"),
        ("p-s2-cat", "Results, QoL status", "P, QoL status by category (chi-square)", "7.6 × 10⁻⁵⁹²", "report: chi-square QoL status x category"),
        ("p-s2-mgmt", "Results, QoL status", "P, management by QoL status (chi-square)", "7.8 × 10⁻⁴⁹⁶", "figure: chi-square QoL_assessment x management_categories"),
        ("p-op-r", "Results, operational", "P, Pearson r of cost and steps", "2.5 × 10⁻⁴¹⁰", "report: Pearson Cost vs React Steps")):
    add(id_, where, what, printed, lambda R, key=key: exact_p(R, key), EXACTP)
