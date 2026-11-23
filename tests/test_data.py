"""Data files: checksums, sizes and content of the published outputs."""
import json

import pandas as pd


def test_checksums(prepared):
    n = prepared.verify_checksums()
    assert n == len(json.loads(prepared.CHECKSUMS.read_text()))


def test_published_classifications(prepared):
    d = pd.read_csv(prepared.CLASSIFICATIONS, low_memory=False)
    terms = pd.read_csv(prepared.HPO_TERMS)
    assert len(d) == 10652 and d.hpo_id.is_unique
    assert set(d.hpo_id) == set(terms.hpo_id)
    for col in ("classification_tier", "classification_category", "reasoning", "all_found_references",
                "cited_references", "verification_log", "statement_support_score"):
        assert col in d.columns


def test_ground_truth(prepared):
    gt = pd.read_excel(prepared.GROUND_TRUTH)
    assert len(gt) == 898 and gt.hpo_id.is_unique
    assert {"Final classification_tier", "Final_classification_category"} <= set(gt.columns)


def test_baseline_answers(prepared):
    b = pd.read_csv(prepared.BASELINE_ANSWERS)
    gt = pd.read_excel(prepared.GROUND_TRUTH)
    assert set(b.hpo_id) == set(gt.hpo_id)
    assert (b.model == "openai/gpt-4o").all()
