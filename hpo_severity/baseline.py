"""
No-retrieval baseline of the agent (Figure 2e and 2f).

Each ground-truth term is classified once with the baseline prompt in prompts/baseline/ (the
classification prompt with the sources and the rules about sources removed), with no retrieved
literature. The baseline answers and the agent output are then scored against the ground truth on
the same terms: accuracy and Cohen's kappa (category), accuracy and quadratic-weighted kappa (tier),
95% bootstrap confidence intervals (2,000 resamples of terms, seed 42) and exact McNemar tests.

Run it through the entry point: python run.py baseline (needs OPENROUTER_API_KEY). The scoring alone
runs without an API key as part of python run.py verify.
"""
import argparse
import json
import math
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests

from . import paths, prompts

# ---------------------------------------------------------------- settings
MODEL = "openai/gpt-4o"                     # same model as the agent; change with --model
PRICES = {                                   # USD per 1M tokens (input, output); others need --price-in/--price-out
    "openai/gpt-4o": (2.5, 10.0),
}
TEMPERATURE = 0.2                            # same as the agent (every call of the paper's runs used 0.2)
API_URL = "https://openrouter.ai/api/v1/chat/completions"
N_BOOT = 2000
SEED = 42

FOLDER = os.path.join(str(paths.ROOT), "outputs", "baseline")
OUTPUT = FOLDER                              # set by configure()
GT_FILE = str(paths.GROUND_TRUTH)
MAIN_FILE = str(paths.CLASSIFICATIONS)       # the agent classification file of the paper's run
OBO_FILE = str(paths.HPO_2025)               # HPO release used for the baseline term context
ENV_FILE = os.path.join(str(paths.ROOT), ".env")
RUN_CSV = os.path.join(OUTPUT, "baseline_run.csv")

CATEGORIES = [
    "Shortened life span: infancy", "Shortened life span: childhood/adolescence",
    "Shortened life span: adulthood", "Intellectual disability", "Internal physical malformations",
    "Impaired mobility", "Dysmorphic feature", "Sensory Impairment – Vision",
    "Sensory Impairment – Hearing", "Sensory impairment: touch, others", "Immunodeficiency/cancer",
    "Mental illness", "Infertility", "Clinical Sign or Laboratory Abnormality",
    "NFC (Not Further Classified)", "Out of Scope: Acquired/Multifactorial",
]
TIERS = ["1", "2", "3", "4", "NFC"]          # ordinal order for the weighted kappa (NFC after 4)


def key(text):
    """Letters and digits only, lower case. Makes 'Sensory Impairment - Vision' match 'Sensory Impairment – Vision'."""
    text = str(text).replace("â€“", "–")
    return re.sub(r"[^a-z0-9]", "", text.lower())


CAT_BY_KEY = {key(c): c for c in CATEGORIES}
CAT_BY_KEY["nfc"] = "NFC (Not Further Classified)"


def canon_cat(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    return CAT_BY_KEY.get(key(x))


def canon_tier(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    s = str(x).strip().upper().replace("TIER", "").strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s if s in TIERS else None


# ---------------------------------------------------------------- prompt and output check
SYSTEM_PROMPT = prompts.text("baseline/system.txt")


def build_prompt(hpo_context):
    """The baseline prompt (prompts/baseline/classification.json) for one term."""
    p = prompts.fill(prompts.load_json("baseline/classification.json"), {
        "hpo_id": hpo_context.get('id', 'N/A'),
        "hpo_name": hpo_context.get('name', 'N/A'),
        "hpo_definition": hpo_context.get('definition', 'N/A'),
        "hpo_synonyms": hpo_context.get('synonyms', []),
        "hpo_comment": hpo_context.get('comment', 'N/A'),
    })
    return json.dumps(p, indent=2)


def apply_classification_rules(classification_data: dict) -> dict:
    """
    Applies these rules to the final classification from the LLM:
    1. Prioritization of exclusion categories (NFC, Out of Scope).
    2. Enforcement of severity based on affected functional domains.
    3. An "Inherent Severity Override" to catch search failures for severe conditions.
    """
    if not isinstance(classification_data, dict):
        return classification_data

    classification = classification_data.get('classification', {})
    if not classification: # Handle case where classification dict is missing
        return classification_data
        
    reasoning_raw = classification_data.get('reasoning', {})
    
    initial_category = classification.get('category', '')

    # First, handle exclusion categories immediately.
    if initial_category in ['Out of Scope: Acquired/Multifactorial', 'NFC (Not Further Classified)']:
        classification['tier'] = 'NFC'
        classification['severity_assessment'] = 'Non-Severe' 
        classification_data['classification'] = classification
        return classification_data

    try:
        reasoning = json.loads(reasoning_raw) if isinstance(reasoning_raw, str) else reasoning_raw
    except (json.JSONDecodeError, TypeError):
        reasoning = {}
    
    # Normalize the category name for reliable matching
    parts = [part for part in re.split(r'[^a-zA-Z0-9]+', initial_category) if part]
    normalized_category = " ".join(parts)
        
    functional_impact = reasoning.get('functional_impact', {})
    affected_domains = functional_impact.get('affected_domains', [])
    
    # 1. Define the set of categories that are always considered severe by their nature.
    # These are the Tier 1 and Tier 2 categories.
    inherently_severe_categories = {
        'Intellectual disability',
        'Shortened life span infancy',
        'Shortened life span childhood adolescence',
        'Shortened life span adulthood',
        'Impaired mobility',
        'Internal physical malformations'
    }
    
    # 2. Apply the severity logic
    if isinstance(affected_domains, list) and any(affected_domains):
        # Rule 1: If there's direct evidence of functional impact, it's always Severe.
        classification['severity_assessment'] = 'Severe'
    elif normalized_category in inherently_severe_categories:
        # Rule 2: If the category itself is inherently severe, override to Severe.
        # This catches cases like Holoprosencephaly where search may have failed but the category is right.
        classification['severity_assessment'] = 'Severe'
        print(f"Inherent Severity Override triggered for category: {initial_category}")
    else:
        # Only classify as Non-Severe if there's NO domain impact AND the category is NOT inherently severe.
        classification['severity_assessment'] = 'Non-Severe'
    
        
    # 3. Derive the tier from the severity and the normalized category.
    if classification.get('severity_assessment') == 'Non-Severe':
        classification['tier'] = '4'
    else: # If it is 'Severe'
        tier_mapping = {
            'Intellectual disability': '1',
            'Shortened life span infancy': '1',
            'Shortened life span childhood adolescence': '1',
            'Shortened life span adulthood': '2',
            'Impaired mobility': '2',
            'Internal physical malformations': '2',
            'Sensory Impairment Vision': '3',
            'Sensory Impairment Hearing': '3',
            'Sensory impairment touch others': '3',
            'Immunodeficiency cancer': '3',
            'Mental illness': '3',
            'Dysmorphic feature': '3',
            'Infertility': '4',
            'Clinical Sign or Laboratory Abnormality': '4',
            'NFC Not Further Classified': '4'
        }
        expected_tier = tier_mapping.get(normalized_category, 'NFC')
        
        if classification.get('tier') != expected_tier:
            classification['tier'] = expected_tier

    classification_data['classification'] = classification
    return classification_data


# ---------------------------------------------------------------- ontology (same fields as get_rich_hpo_context)
def load_obo(path):
    """name, definition, EXACT synonyms and comment of every term, as the agent reads them."""
    terms, cur = {}, None
    unq = lambda s: s.replace('\\"', '"').replace("\\n", " ")
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if line.startswith("["):
            cur = {} if line == "[Term]" else None
            continue
        if cur is None or ": " not in line:
            continue
        tag, val = line.split(": ", 1)
        if tag == "id":
            cur.update(id=val, name="", definition="", synonyms=[], comment="")
            terms[val] = cur
        elif tag == "name":
            cur["name"] = val
        elif tag == "def":
            m = re.match(r'"((?:[^"\\]|\\.)*)"', val)
            cur["definition"] = unq(m.group(1)) if m else ""
        elif tag == "synonym":
            m = re.match(r'"((?:[^"\\]|\\.)*)"\s+(\w+)', val)
            if m and m.group(2) == "EXACT":
                cur["synonyms"].append(unq(m.group(1)))
        elif tag == "comment":
            cur["comment"] = val
    return terms


# ---------------------------------------------------------------- inputs
def read_env(path):
    env = {}
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def load_gt():
    """Unique GT terms. Identical duplicate rows are kept once; an ID with conflicting labels is left out."""
    gt = pd.read_excel(GT_FILE, dtype=str)
    gt = gt.rename(columns={"Final classification_tier": "gt_tier", "Final_classification_category": "gt_raw_cat"})
    gt["hpo_id"] = gt["hpo_id"].str.strip()
    gt["gt_tier"] = gt["gt_tier"].map(canon_tier)
    gt["gt_category"] = gt["gt_raw_cat"].map(canon_cat)
    left = []
    for hid, grp in gt[gt["hpo_id"].duplicated(keep=False)].groupby("hpo_id"):
        if grp[["gt_tier", "gt_category"]].drop_duplicates().shape[0] > 1:
            left.append((hid, grp["hpo_name"].iloc[0], f"duplicate in GT with different labels ({len(grp)} rows)"))
    gt = gt[~gt["hpo_id"].isin([h for h, _, _ in left])].drop_duplicates("hpo_id")
    for _, r in gt[gt["gt_tier"].isna() | gt["gt_category"].isna()].iterrows():
        left.append((r.hpo_id, r.hpo_name, "GT tier or category missing or not recognised"))
    return gt[["hpo_id", "hpo_name", "gt_tier", "gt_category"]], left


# ---------------------------------------------------------------- one term
def classify(session, api_key, model, prompt, system_prompt, validate, max_tries):
    """Returns (result dict, None) or (None, last error). Retries API errors, bad JSON and missing categories."""
    payload = {"model": model, "response_format": {"type": "json_object"}, "temperature": TEMPERATURE,
               "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]}
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json",
               "HTTP-Referer": "http://localhost", "X-Title": "HPO Classification Agent"}
    err, tok_in, tok_out = None, 0, 0
    for attempt in range(1, max_tries + 1):
        try:
            r = session.post(API_URL, json=payload, headers=headers, timeout=180)
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
            data = r.json()
            usage = data.get("usage") or {}
            tok_in += usage.get("prompt_tokens", 0)
            tok_out += usage.get("completion_tokens", 0)
            raw = data["choices"][0]["message"]["content"] or ""
            raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
            resp = json.loads(raw)
            cls = resp.get("classification") if isinstance(resp, dict) else None
            if not isinstance(cls, dict) or canon_cat(cls.get("category")) is None:
                raise ValueError(f"no valid category in reply: {str(cls)[:150]}")
            llm_tier = cls.get("tier")
            cls["category"] = canon_cat(cls["category"])      # exact spelling, so the tier code maps it
            checked = validate(resp)["classification"]
            tier = canon_tier(checked.get("tier"))
            if tier is None:
                raise ValueError(f"tier code gave no valid tier: {checked.get('tier')}")
            return {"classification_tier": tier, "classification_category": checked["category"],
                    "severity_assessment": checked.get("severity_assessment"), "llm_tier": llm_tier,
                    "tries": attempt, "input_tokens": tok_in, "output_tokens": tok_out,
                    "response_json": json.dumps(resp, ensure_ascii=False)}, None
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            time.sleep(min(60, 3 * 2 ** (attempt - 1)))
    return None, err


def run(args, validate, system_prompt, ontology, gt):
    api_key = read_env(args.env).get("OPENROUTER_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit(f"OPENROUTER_API_KEY not found in the environment or in {args.env}")
    done = set(pd.read_csv(RUN_CSV, usecols=["hpo_id"])["hpo_id"]) if os.path.exists(RUN_CSV) else set()
    todo = [h for h in gt["hpo_id"] if h in ontology and h not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(gt)} GT terms, {len(done)} done, {len(todo)} to do now | {args.model} | temperature {TEMPERATURE}")
    lock, failed = threading.Lock(), []

    def one(h):
        return classify(session, api_key, args.model, build_prompt(ontology[h]),
                        system_prompt, validate, args.max_tries)

    with requests.Session() as session, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(one, h): h for h in todo}
        for i, f in enumerate(as_completed(futs), 1):
            h = futs[f]
            res, err = f.result()
            with lock:
                if res is None:
                    failed.append(h)
                    print(f"[{i}/{len(todo)}] {h} FAILED after {args.max_tries} tries: {err}")
                    continue
                row = {"hpo_id": h, "hpo_name": ontology[h]["name"], "model": args.model, **res}
                pd.DataFrame([row]).to_csv(RUN_CSV, mode="a", index=False, header=not os.path.exists(RUN_CSV))
                print(f"[{i}/{len(todo)}] {h} {row['hpo_name']}: tier {res['classification_tier']}, "
                      f"{res['classification_category']} (try {res['tries']})")
    if failed:
        print(f"\n{len(failed)} terms still have no answer. Run the same command again to retry them: {failed}")
    return failed


# ---------------------------------------------------------------- statistics
def cohen_kappa(a, b, labels, weights=None):
    idx = {l: i for i, l in enumerate(labels)}
    n = len(labels)
    obs = np.zeros((n, n))
    for x, y in zip(a, b):
        obs[idx[x], idx[y]] += 1
    obs /= obs.sum()
    exp = np.outer(obs.sum(1), obs.sum(0))
    i, j = np.indices((n, n))
    w = ((i - j) ** 2) / ((n - 1) ** 2) if weights == "quadratic" else (i != j).astype(float)
    denom = (w * exp).sum()
    return 1.0 if denom == 0 else 1 - (w * obs).sum() / denom


def mcnemar_exact(c_main, c_base):
    """Exact two-sided McNemar test. b = main right & baseline wrong, c = main wrong & baseline right."""
    b = int(np.sum(c_main & ~c_base))
    c = int(np.sum(~c_main & c_base))
    n = b + c
    if n == 0:
        return b, c, 1.0
    p = 2 * sum(math.comb(n, k) for k in range(0, min(b, c) + 1)) / 2 ** n
    return b, c, min(1.0, p)


def score_one(gt_v, main_v, base_v, labels, weights, name, rng):
    gt_v, main_v, base_v = map(np.asarray, (gt_v, main_v, base_v))
    n = len(gt_v)
    kname = "quadratic weighted kappa" if weights else "Cohen's kappa"
    stat = lambda g, p: ((g == p).mean(), cohen_kappa(g, p, labels, weights))
    acc_m, k_m = stat(gt_v, main_v)
    acc_b, k_b = stat(gt_v, base_v)
    boots = []
    for _ in range(N_BOOT):
        s = rng.integers(0, n, n)
        am, km = stat(gt_v[s], main_v[s])
        ab, kb = stat(gt_v[s], base_v[s])
        boots.append((am, km, ab, kb, am - ab, km - kb))
    lo, hi = np.percentile(np.array(boots), [2.5, 97.5], axis=0)
    b, c, p = mcnemar_exact(main_v == gt_v, base_v == gt_v)
    point = [acc_m, k_m, acc_b, k_b, acc_m - acc_b, k_m - k_b]
    rows = []
    labels_out = [("main run", "accuracy"), ("main run", kname), ("baseline", "accuracy"), ("baseline", kname),
                  ("main minus baseline", "accuracy"), ("main minus baseline", kname)]
    for (who, metric), v, l, h in zip(labels_out, point, lo, hi):
        row = {"level": name, "run": who, "metric": metric, "n": n, "value": round(v, 4),
               "ci95_low": round(l, 4), "ci95_high": round(h, 4),
               "_exact": {"value": v, "ci95_low": l, "ci95_high": h}}
        if who == "main minus baseline" and metric == "accuracy":
            row.update(mcnemar_main_right_base_wrong=b, mcnemar_main_wrong_base_right=c, mcnemar_p_exact=p)
        rows.append(row)
    return rows


def score(gt, gt_left, ontology):
    if not os.path.exists(RUN_CSV):
        sys.exit("No baseline results yet. Run python run.py baseline first.")
    base = pd.read_csv(RUN_CSV, dtype=str).drop_duplicates("hpo_id", keep="last")
    base["base_tier"] = base["classification_tier"].map(canon_tier)
    base["base_category"] = base["classification_category"].map(canon_cat)
    main = pd.read_csv(MAIN_FILE, dtype=str, usecols=["hpo_id", "classification_tier", "classification_category"])
    main["hpo_id"] = main["hpo_id"].str.strip()
    main = main.drop_duplicates("hpo_id")
    main["main_tier"] = main["classification_tier"].map(canon_tier)
    main["main_category"] = main["classification_category"].map(canon_cat)

    df = gt.merge(main[["hpo_id", "main_tier", "main_category"]], on="hpo_id", how="left")
    df = df.merge(base[["hpo_id", "base_tier", "base_category", "severity_assessment", "tries",
                        "input_tokens", "output_tokens"]], on="hpo_id", how="left")
    left = list(gt_left)
    main_ids = set(main["hpo_id"])
    for _, r in df.iterrows():
        if r.hpo_id not in main_ids:
            left.append((r.hpo_id, r.hpo_name, "not in the main run file"))
        elif pd.isna(r.main_tier) or pd.isna(r.main_category):
            left.append((r.hpo_id, r.hpo_name, "main run tier or category empty or not recognised"))
        elif r.hpo_id not in ontology:
            left.append((r.hpo_id, r.hpo_name, "not in the ontology file, so the baseline could not run it"))
        elif pd.isna(r.base_tier) or pd.isna(r.base_category):
            left.append((r.hpo_id, r.hpo_name, "baseline has no valid answer yet (run again to retry)"))
    left_df = pd.DataFrame(left, columns=["hpo_id", "hpo_name", "reason"])
    s = df[~df["hpo_id"].isin(left_df["hpo_id"])].copy()
    # Clinical Sign or Laboratory Abnormality is a non-tiered category: its tier is NFC on each side,
    # set from that side's own category (the classification files store tier 4 for it).
    for w in ("gt", "main", "base"):
        s.loc[s[f"{w}_category"] == "Clinical Sign or Laboratory Abnormality", f"{w}_tier"] = "NFC"
    for lvl in ("tier", "category"):
        s[f"main_{lvl}_correct"] = s[f"main_{lvl}"] == s[f"gt_{lvl}"]
        s[f"base_{lvl}_correct"] = s[f"base_{lvl}"] == s[f"gt_{lvl}"]

    rng = np.random.default_rng(SEED)
    rows = score_one(s.gt_category, s.main_category, s.base_category, CATEGORIES, None, "category", rng)
    rows += score_one(s.gt_tier, s.main_tier, s.base_tier, TIERS, "quadratic", "tier", rng)
    exact = pd.DataFrame([{**{k: v for k, v in r.items() if k != "_exact"}, **r["_exact"]} for r in rows])
    scores = pd.DataFrame([{k: v for k, v in r.items() if k != "_exact"} for r in rows])

    s.to_csv(os.path.join(OUTPUT, "per_term_results.csv"), index=False, encoding="utf-8-sig")
    scores.to_csv(os.path.join(OUTPUT, "scores.csv"), index=False, encoding="utf-8-sig")
    exact.to_csv(os.path.join(OUTPUT, "scores_full_precision.csv"), index=False, encoding="utf-8-sig")
    left_df.to_csv(os.path.join(OUTPUT, "left_out_terms.csv"), index=False, encoding="utf-8-sig")

    model = base["model"].dropna().iloc[0] if "model" in base and base["model"].notna().any() else "?"
    tin = pd.to_numeric(base["input_tokens"], errors="coerce").sum()
    tout = pd.to_numeric(base["output_tokens"], errors="coerce").sum()
    pin, pout = PRICES.get(model, (float("nan"), float("nan")))
    L = [f"No-retrieval baseline vs main run, scored against the ground truth",
         f"Baseline model: {model}, temperature {TEMPERATURE}, prompt = prompts/baseline/ (the classification prompt with "
         f"zero sources: sources block, citation rules and source-only rules removed, one line added telling the model to use "
         f"its own knowledge).",
         f"Note: the main run used openai/gpt-4o with retrieval, so if the baseline model differs, "
         f"the difference mixes model and retrieval.",
         f"Ground truth: data/ground_truth.xlsx, columns Final classification_tier and Final_classification_category.",
         f"Terms in GT (unique): {len(gt) + sum(1 for x in gt_left if 'duplicate' in x[2])}, scored on both runs: {len(s)}, "
         f"left out: {len(left_df)} (see left_out_terms.csv).", ""]
    for reason, n in left_df["reason"].value_counts().items():
        L.append(f"  left out, {reason}: {n}")
    L.append("")
    for lvl in ("category", "tier"):
        t = scores[scores.level == lvl]
        L.append(lvl.upper())
        for _, r in t.iterrows():
            L.append(f"  {r.run:<20} {r.metric:<25} {r.value:.3f}  (95% CI {r.ci95_low:.3f} to {r.ci95_high:.3f})")
        r = t[(t.run == "main minus baseline") & (t.metric == "accuracy")].iloc[0]
        L.append(f"  McNemar (exact): main right and baseline wrong {int(r.mcnemar_main_right_base_wrong)}, "
                 f"main wrong and baseline right {int(r.mcnemar_main_wrong_base_right)}, p = {r.mcnemar_p_exact:.4g}")
        L.append("")
    L.append(f"Bootstrap: {N_BOOT} resamples of terms, seed {SEED}, percentile CI. Tier kappa uses the order 1, 2, 3, 4, NFC.")
    L.append(f"Baseline tokens: {int(tin):,} in, {int(tout):,} out, about ${tin / 1e6 * pin + tout / 1e6 * pout:.2f} "
             f"at ${pin}/{pout} per 1M (all tries counted).")
    text = "\n".join(L)
    open(os.path.join(OUTPUT, "summary.txt"), "w", encoding="utf-8").write(text + "\n")
    print("\n" + text)
    plot(scores, s)


# ---------------------------------------------------------------- figures
def plot(scores, s):
    """fig_accuracy.png: accuracy and kappa with 95% CI, main vs baseline, for category and tier.
    fig_per_category.png: accuracy per GT category (also saved as per_category_accuracy.csv)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, so no figures (pip install matplotlib).")
        return
    colors = {"main run": "#2b6cb0", "baseline": "#a0aec0"}

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, lvl in zip(axes, ("category", "tier")):
        t = scores[(scores.level == lvl) & scores.run.isin(colors)]
        metrics = list(dict.fromkeys(t.metric))
        x = np.arange(len(metrics))
        for i, run_name in enumerate(colors):
            r = t[t.run == run_name].set_index("metric").loc[metrics]
            pos = x + (i - 0.5) * 0.38
            ax.bar(pos, r.value, 0.36, color=colors[run_name], label=run_name,
                   yerr=[r.value - r.ci95_low, r.ci95_high - r.value], capsize=4)
            for xx, v in zip(pos, r.value):
                ax.text(xx, 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=9, color="white")
        ax.set_xticks(x, [m.replace("quadratic weighted kappa", "weighted kappa") for m in metrics])
        ax.set_title(f"{lvl.capitalize()} (n = {int(t.n.iloc[0])})")
        ax.set_ylim(0, 1)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Agreement with ground truth")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, loc="upper center", ncol=2)
    fig.text(0.5, 0.01, "Error bars: 95% bootstrap CI", ha="center", fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    fig.savefig(os.path.join(OUTPUT, "fig_accuracy.png"), dpi=200)
    plt.close(fig)

    per = (s.groupby("gt_category")
           .agg(n=("hpo_id", "size"), main_run=("main_category_correct", "mean"),
                baseline=("base_category_correct", "mean"))
           .sort_values("n"))
    per.round(4).to_csv(os.path.join(OUTPUT, "per_category_accuracy.csv"), encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(per) + 1.2))
    y = np.arange(len(per))
    ax.barh(y + 0.2, per.main_run, 0.38, color=colors["main run"], label="main run")
    ax.barh(y - 0.2, per.baseline, 0.38, color=colors["baseline"], label="baseline")
    ax.set_yticks(y, [f"{c} (n = {n})" for c, n in zip(per.index, per.n)], fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Category accuracy against ground truth")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT, "fig_per_category.png"), dpi=200)
    plt.close(fig)
    print(f"Wrote fig_accuracy.png, fig_per_category.png and per_category_accuracy.csv in {OUTPUT}")


# ---------------------------------------------------------------- main
def configure(output, classifications=None):
    """Sets the output folder (and optionally the agent classification file to score)."""
    global OUTPUT, RUN_CSV, MAIN_FILE
    OUTPUT = str(output)
    RUN_CSV = os.path.join(OUTPUT, "baseline_run.csv")
    if classifications:
        MAIN_FILE = str(classifications)
    os.makedirs(OUTPUT, exist_ok=True)


def score_published(output):
    """Scores the published baseline answers against the ground truth, without API calls."""
    configure(output)
    import shutil
    shutil.copyfile(paths.BASELINE_ANSWERS, RUN_CSV)
    gt, gt_left = load_gt()
    score(gt, gt_left, load_obo(OBO_FILE))


def main(argv=None):
    p = argparse.ArgumentParser(prog="run.py baseline", description="No-retrieval baseline of the HPO classification agent")
    p.add_argument("--output", default=os.path.join("outputs", "baseline"), help="output folder (default outputs/baseline)")
    p.add_argument("--model", default=MODEL, help=f"OpenRouter model ID (default {MODEL})")
    p.add_argument("--limit", type=int, help="classify only the first N terms still to do")
    p.add_argument("--workers", type=int, default=8, help="terms at the same time (default 8)")
    p.add_argument("--max-tries", type=int, default=10, help="tries per term in one run (default 10)")
    p.add_argument("--price-in", type=float, help="USD per 1M input tokens (for the cost line)")
    p.add_argument("--price-out", type=float, help="USD per 1M output tokens")
    p.add_argument("--env", default=ENV_FILE, help="path of the .env with OPENROUTER_API_KEY")
    p.add_argument("--dry-run", action="store_true", help="no API calls: write one prompt and a token estimate")
    args = p.parse_args(argv)
    if args.price_in is not None and args.price_out is not None:
        PRICES[args.model] = (args.price_in, args.price_out)
    configure(args.output)
    print(f"Output folder: {OUTPUT}")

    system_prompt = SYSTEM_PROMPT
    ontology = load_obo(OBO_FILE)
    gt, gt_left = load_gt()

    # The exact prompt of one term, written on each run so the output folder documents it.
    h = next(x for x in gt["hpo_id"] if x in ontology)
    prompt = build_prompt(ontology[h])
    with open(os.path.join(OUTPUT, "prompt_example.txt"), "w", encoding="utf-8") as f:
        f.write(f"SYSTEM PROMPT:\n{system_prompt}\n\nUSER PROMPT ({h}):\n{prompt}\n")
    if args.dry_run:
        n = sum(1 for x in gt["hpo_id"] if x in ontology)
        est = len(prompt + system_prompt) / 4
        print(f"{len(gt)} GT terms, {n} in the ontology. Prompt for {h}: {len(prompt):,} characters, "
              f"about {est:,.0f} input tokens per term (characters / 4, a rough guess).")
        print("Wrote prompt_example.txt")
        return

    if args.model not in PRICES:
        print(f"Note: no price known for {args.model}; pass --price-in and --price-out for the cost line.")
    failed = run(args, apply_classification_rules, system_prompt, ontology, gt)
    if args.limit:
        print("Test run done (--limit). Run without --limit for all terms.")
        return
    if failed:
        print("Scoring anyway; terms without an answer are listed in left_out_terms.csv.")
    score(gt, gt_left, ontology)


if __name__ == "__main__":
    main()
