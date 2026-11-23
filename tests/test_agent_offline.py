"""Parts of the agent that run without a model."""
import json
import math

import pandas as pd


def test_support_score_is_the_weighted_mean_of_the_claims(prepared):
    """The saved support score is the weighted mean of the claim ratings in the verification log."""
    d = pd.read_csv(prepared.CLASSIFICATIONS, low_memory=False, usecols=["statement_support_score", "verification_log"])
    w = {"direct_support": 1.0, "valid_inferences": 0.90, "weak_inferences": 0.60, "unsupported": 0.0}
    checked = 0
    for score, log in zip(d.statement_support_score, d.verification_log):
        summary = json.loads(log)[0]
        n = summary.get("total_claims", 0)
        if not n or math.isnan(score) or not set(w) <= set(summary):
            continue
        assert abs(sum(w[k] * summary[k] for k in w) / n - score) < 1e-6
        checked += 1
    assert checked > 9000


def test_classification_rules():
    from hpo_severity import agent

    def rule(category, domains):
        r = agent.validate_and_correct_classification(
            {"classification": {"category": category, "tier": "?"}, "reasoning": {"functional_impact": {"affected_domains": domains}}})
        return r["classification"]["severity_assessment"], r["classification"]["tier"]

    assert rule("Dysmorphic feature", ["Social Functioning"]) == ("Severe", "3")
    assert rule("Dysmorphic feature", []) == ("Non-Severe", "4")
    assert rule("Internal physical malformations", []) == ("Severe", "2")
    assert rule("Intellectual disability", ["School/Cognitive Functioning"]) == ("Severe", "1")
    assert rule("Out of Scope: Acquired/Multifactorial", ["Physical Functioning"]) == ("Non-Severe", "NFC")


def test_claim_extraction():
    from hpo_severity import agent
    reasoning = {"conclusion": "Based on the cited evidence: [1] It is inherited [Ref 2]. "
                               "[3] It belongs in the Dysmorphic feature category due to 'a visible change' [Ref 3].",
                 "functional_impact": {"domain_evidence": "Feeding is difficult [Ref 1]."}}
    claims = agent.extract_claims_from_reasoning(reasoning)
    assert any("inherited [Ref 2]" in c for c in claims)
    assert any("Feeding is difficult [Ref 1]" in c for c in claims)
    assert not any("belongs in the" in c for c in claims)


def test_no_api_call_without_key(monkeypatch):
    from hpo_severity import agent
    monkeypatch.setattr(agent, "OPENAI_API_KEY", None)
    assert agent.call_openai_api(None, "x", "y") == (None, None)
