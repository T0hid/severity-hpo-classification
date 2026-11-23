"""Recomputes every code-derived number in the manuscript from the saved outputs and reports PASSED or
FAILED for each one. No language model is called.

Steps:
  1. unpack the data and check every file against data/checksums.json;
  2. score the published baseline answers and the agent output against the ground truth;
  3. rebuild the gene-disease table from the inputs and check that it is identical to the published one;
  4. redraw Figure 7 and Figures 2 to 6 and S1 to S3 (the statistical tests the figure code runs are
     recorded while it runs);
  5. compare each manuscript number with the recomputed value, rounded the way the manuscript prints it.

The expected values are in manuscript_numbers.py. The report is written to verify_report.csv and
verify_report.md in the output folder.
"""
import contextlib
import io
import json
import re
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np
import pandas as pd

from . import paths


# ---------------------------------------------------------------- comparison
def _decimals(core):
    return len(core.split(".")[1]) if "." in core else 0


def format_like(value, expected):
    """Formats a computed value the way the expected string is written (decimals, %, $, thousands comma)."""
    if isinstance(value, str):
        return value
    s = expected.strip()
    if s.startswith("<"):
        return f"{value:.3g}"
    prefix = "$" if s.startswith("$") else ""
    suffix = "%" if s.endswith("%") else ""
    core = s.lstrip("$").rstrip("%")
    d = _decimals(core)
    q = Decimal(repr(float(value))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP)
    text = f"{q:,.{d}f}" if "," in core else f"{q:.{d}f}"
    return prefix + text + suffix


def matches(value, expected):
    if isinstance(value, str):
        return value == expected
    if expected.strip().startswith("<"):
        return float(value) < float(expected.strip()[1:])
    return format_like(value, expected) == expected.strip()


# ---------------------------------------------------------------- recorded statistical tests
class TestRecorder:
    """Records the results of scipy.stats.chi2_contingency while the figure code runs."""

    def __init__(self):
        self.chi2 = []

    @contextlib.contextmanager
    def active(self):
        import scipy.stats as st
        original = st.chi2_contingency

        def recorded(table, *args, **kwargs):
            res = original(table, *args, **kwargs)
            rows = list(map(str, getattr(table, "index", [])))
            cols = list(map(str, getattr(table, "columns", [])))
            self.chi2.append({"chi2": float(res[0]), "p": float(res[1]), "dof": int(res[2]), "rows": rows, "columns": cols})
            return res

        st.chi2_contingency = recorded
        try:
            yield self
        finally:
            st.chi2_contingency = original

    def find(self, dof, row_has=None, col_has=None):
        hits = [t for t in self.chi2 if t["dof"] == dof
                and (row_has is None or row_has in t["rows"]) and (col_has is None or col_has in t["columns"])]
        if not hits:
            raise KeyError(f"no chi-square test with {dof} degrees of freedom was recorded")
        return hits[-1]


# ---------------------------------------------------------------- results of one verification run
class Results:
    def __init__(self, work: Path, gene_log: str, tests: TestRecorder):
        self.work = work
        self.tests = tests
        self.gene_log = gene_log
        self.cls = pd.read_csv(paths.CLASSIFICATIONS, low_memory=False)
        sc = pd.read_csv(work / "baseline" / "scores_full_precision.csv", encoding="utf-8-sig")
        self.scores = {(r.level, r.run, r.metric): r for r in sc.itertuples()}
        self.per_term = pd.read_csv(work / "baseline" / "per_term_results.csv", dtype=str, encoding="utf-8-sig")
        rep = work / "figures" / "data_and_reports"
        self.val = pd.read_csv(rep / "validation_terms_scored.csv", low_memory=False)
        self.pt = pd.read_csv(rep / "phenotype_table_Figures3to6_S2.csv", dtype={"classification_tier": str},
                              keep_default_na=False, na_values=[""])
        self.report = (rep / "analysis_report_Figures3to6_S2_S3.txt").read_text(encoding="utf-8")
        self.vreport = (rep / "validation_report_Figure2.txt").read_text(encoding="utf-8")
        self.f7 = pd.read_excel(work / "gene_level" / "figure7_data.xlsx", sheet_name=None)
        self.ms = pd.read_csv(paths.MODEL_SELECTION_SUMMARY)
        self.gt = pd.read_excel(paths.GROUND_TRUTH)
        self.gt_all = pd.read_excel(paths.GROUND_TRUTH_ALL_LABELLED)
        self.gene_table_identical = (paths.sha256(work / "gene_table" / "gene_disease_table.csv") ==
                                     paths.sha256(paths.GENE_TABLE))
        self._cache = {}

    # --- scores.csv (baseline comparison)
    def score(self, level, run, metric, field="value"):
        return getattr(self.scores[(level, run, metric)], field)

    # --- analysis report
    def block(self, start, end=None):
        i = self.report.index(start)
        j = self.report.index(end, i + len(start)) if end else len(self.report)
        return self.report[i:j]

    def count_line(self, block, label):
        m = re.search(r"^" + re.escape(label) + r"\s{2,}(-?[\d.]+)\s*$", block, re.M)
        if not m:
            raise KeyError(label)
        return float(m.group(1))

    def table(self, block):
        """Rows of a pandas table printed in the report: {row label: [numbers]}"""
        out = {}
        for line in block.splitlines():
            m = re.match(r"^(\S.*?)\s{2,}(-?[\d.]+(?:\s+-?[\d.]+)*)\s*$", line)
            if m:
                out[m.group(1).strip()] = [float(x) for x in m.group(2).split()]
        return out

    def stat(self, pattern):
        m = re.search(pattern, self.report)
        if not m:
            raise KeyError(pattern)
        return m.groups()

    # --- validation (Figure 2)
    def cat_metrics(self):
        if "cat" not in self._cache:
            from sklearn.metrics import (balanced_accuracy_score, classification_report, cohen_kappa_score,
                                         f1_score, matthews_corrcoef, precision_score, recall_score)
            yt, yp = self.val["Final_classification_category"], self.val["classification_category_pred"]
            labels = sorted(yt.unique())
            rep = classification_report(yt, yp, labels=labels, zero_division=0, output_dict=True)
            self._cache["cat"] = {
                "precision_w": precision_score(yt, yp, average="weighted", zero_division=0),
                "recall_w": recall_score(yt, yp, average="weighted", zero_division=0),
                "f1_w": f1_score(yt, yp, average="weighted", zero_division=0),
                "f1_macro": f1_score(yt, yp, average="macro", zero_division=0),
                "mcc": matthews_corrcoef(yt, yp), "kappa": cohen_kappa_score(yt, yp),
                "balanced": balanced_accuracy_score(yt, yp),
                "per_class": {k: rep[k] for k in labels},
                "confusion": pd.crosstab(yt, yp),
            }
        return self._cache["cat"]

    def tier_metrics(self):
        if "tier" not in self._cache:
            from scipy.stats import kendalltau, spearmanr
            from sklearn.metrics import classification_report
            yt, yp = self.val["Final classification_tier"].astype(int), self.val["classification_tier_pred"].astype(int)
            rep = classification_report(yt, yp, zero_division=0, output_dict=True)
            acc_by_tier = {t: (g["Final_classification_category"] == g["classification_category_pred"]).mean()
                           for t, g in self.val.groupby("Final classification_tier")}
            self._cache["tier"] = {
                "spearman": spearmanr(yt, yp)[0], "kendall": kendalltau(yt, yp)[0],
                "mae": float(np.mean(np.abs(yt - yp))), "within1": float(np.mean(np.abs(yt - yp) <= 1)),
                "f1": {int(k): v["f1-score"] for k, v in rep.items() if k.isdigit()},
                "cat_acc_by_tier": acc_by_tier,
            }
        return self._cache["tier"]

    # --- Figure 7 workbook
    def f7row(self, sheet, key_col, key):
        df = self.f7[sheet]
        hit = df[df[key_col].map(_key) == _key(key)]
        if hit.empty:
            raise KeyError((sheet, key))
        return hit.iloc[0]

    # --- gene table log
    def log_int(self, pattern):
        m = re.search(pattern, self.gene_log)
        if not m:
            raise KeyError(pattern)
        return [int(x.replace(",", "")) for x in m.groups()]


def _key(text):
    return re.sub(r"[^a-z0-9]", "", str(text).lower().replace("â€“", "-"))


# ---------------------------------------------------------------- run
def run(output: str) -> int:
    from . import baseline, figures, gene_level, gene_table
    from .manuscript_numbers import CHECKS

    out = Path(output)
    work = out / "work"
    print("Checking data files against data/checksums.json ...")
    paths.prepare()
    print(f"  {len(json.loads(paths.CHECKSUMS.read_text()))} files match their SHA-256.")
    steps = []

    def quiet(name, fn, *args):
        print(f"{name} ...")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            fn(*args)
        (work / "logs").mkdir(parents=True, exist_ok=True)
        (work / "logs" / f"{name.split()[0].lower()}.log").write_text(buf.getvalue(), encoding="utf-8")
        steps.append(name)
        return buf.getvalue()

    quiet("Baseline scoring", baseline.score_published, work / "baseline")
    gene_log = quiet("Gene table (rebuilt from the inputs)", gene_table.build, work / "gene_table" / "gene_disease_table.csv")
    fig7_log = quiet("Figure 7", gene_level.run, work / "gene_level", work / "gene_table" / "gene_disease_table.csv")
    tests = TestRecorder()
    with tests.active():
        quiet("Figures 2 to 6 and S1 to S3", figures.run, work / "figures", work / "baseline" / "per_term_results.csv",
              work / "gene_level")

    R = Results(work, gene_log + "\n" + fig7_log, tests)
    rows = []
    for c in CHECKS:
        try:
            value = c["get"](R)
            ok = matches(value, c["expected"])
            shown = format_like(value, c["expected"])
        except Exception as e:      # a number that cannot be computed is a failure, never a pass
            ok, shown = False, f"error: {type(e).__name__}: {e}"
        rows.append({"id": c["id"], "location": c["location"], "number": c["what"], "manuscript": c["expected"],
                     "recomputed": shown, "result": "PASSED" if ok else "FAILED", "source": c["source"]})
    df = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "verify_report.csv", index=False)
    _write_markdown(df, out / "verify_report.md", R.gene_table_identical)

    n_fail = int((df.result == "FAILED").sum())
    width = max(len(x) for x in df.number) if len(df) else 10
    print()
    for r in df.itertuples():
        print(f"{r.result:6}  {r.location:22} {r.number:{width}}  manuscript {r.manuscript:>12}  recomputed {r.recomputed}")
    print()
    print(f"Gene-disease table rebuilt from the inputs is identical to the published one: {R.gene_table_identical}")
    print(f"{len(df) - n_fail} of {len(df)} numbers PASSED, {n_fail} FAILED. Report: {out / 'verify_report.md'}")
    return n_fail + (0 if R.gene_table_identical else 1)


def _write_markdown(df, path, identical):
    lines = ["# Verification report", "",
             f"{(df.result == 'PASSED').sum()} of {len(df)} numbers PASSED. "
             f"Gene-disease table rebuilt from the inputs is identical to the published one: {identical}.", "",
             "| Result | Location | Number | Manuscript | Recomputed | Source |", "|---|---|---|---|---|---|"]
    for r in df.itertuples():
        lines.append(f"| {r.result} | {r.location} | {r.number} | {r.manuscript} | {r.recomputed} | {r.source} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
