"""The prompts are unchanged: tests/fixtures/expected_prompts.json holds the prompts for the contexts in
tests/fixtures/contexts.json, and the code must write them character for character."""
import copy
import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"
EXPECTED = json.loads((FIXTURES / "expected_prompts.json").read_text(encoding="utf-8"))
CTX = json.loads((FIXTURES / "contexts.json").read_text(encoding="utf-8"))

# Rules of the classification prompt that refer only to the retrieved sources; the baseline prompt leaves them out.
SOURCE_RULES = ("**RULE 1 (NON-NEGOTIABLE): CITE ONLY THE DIRECT SOURCE.**", "NEGATIVE PERSONA:", "DIRECT QUOTES & CITATIONS:",
                "NO INFERENCE BEYOND SOURCES:", "VERIFICATION CHECK:", "ADMIT UNCERTAINTY:", "NO EXTERNAL KNOWLEDGE:",
                "EVIDENCE-ONLY PRINCIPLE:", "**FINAL CHECK (MANDATORY):**")
NO_SOURCES_LINE = ("No sources are provided for this term. Wherever these instructions refer to sources or evidence, "
                   "use your own clinical knowledge instead.")


@pytest.fixture(scope="module")
def agent():
    from hpo_severity import agent
    return agent


def make_context(agent, hpo_id, n_articles):
    c = agent.ReActContext(hpo_id=hpo_id)
    c.hpo_context = copy.deepcopy(CTX["contexts"][hpo_id])
    c.pubmed_articles = copy.deepcopy(CTX["articles"][:n_articles])
    return c


@pytest.mark.parametrize("hpo_id", list(CTX["contexts"]))
@pytest.mark.parametrize("n", [0, 2])
def test_agent_prompts(agent, hpo_id, n):
    c = make_context(agent, hpo_id, n)
    assert agent.create_final_classification_prompt(c) == EXPECTED[f"classification|{hpo_id}|{n}"]
    assert json.dumps(c._reference_urls, ensure_ascii=False) == EXPECTED[f"references|{hpo_id}|{n}"]
    c.mesh_term_searched, c.mesh_term = True, "Example MeSH"
    assert agent.create_react_planning_prompt(c) == EXPECTED[f"planner|{hpo_id}|{n}"]


@pytest.mark.parametrize("hpo_id", list(CTX["contexts"]))
def test_baseline_is_the_classification_prompt_without_sources(agent, hpo_id):
    """The baseline prompt is the agent's classification prompt with the source list and the rules that refer only
    to sources removed, one line added, and a phenotype definition section in the output."""
    from hpo_severity import baseline
    c = CTX["contexts"][hpo_id]
    a = json.loads(agent.create_final_classification_prompt(make_context(agent, hpo_id, 0)))
    b = json.loads(baseline.build_prompt({k: c[k] for k in ("id", "name", "definition", "synonyms", "comment")}))
    del a["input_data"]["retrieved_clinical_sources"]
    g = a["guidelines_and_instructions"]
    g["core_principles"] = [NO_SOURCES_LINE] + [p for p in g["core_principles"] if not p.startswith(SOURCE_RULES)]
    del g["CRITICAL_FINAL_CITATION_CHECK"]
    g["reasoning_approach"] = [r for r in g["reasoning_approach"] if r["rule"] not in ("Rule of Direct Evidence", "No Assumptions")]
    del a["required_json_output_format"]["cited_references"]
    del a["FINAL_INSTRUCTION"]
    del b["reasoning_and_output_structure"]["1_phenotype_definition"]
    del b["required_json_output_format"]["reasoning"]["phenotype_definition"]
    assert a == b


@pytest.mark.parametrize("hpo_id", list(CTX["contexts"]))
def test_baseline_and_model_selection_prompts(hpo_id):
    from hpo_severity import baseline, model_selection
    c = CTX["contexts"][hpo_id]
    bc = {k: c[k] for k in ("id", "name", "definition", "synonyms", "comment")}
    assert baseline.build_prompt(bc) == EXPECTED[f"baseline|{hpo_id}"]
    ms = model_selection.create_classification_prompt({k: c[k] for k in ("id", "name", "definition", "synonyms")},
                                                       ["Abstract one.", "Abstract two."])
    assert ms == EXPECTED[f"model_selection|{hpo_id}"]


def test_system_prompts(agent):
    from hpo_severity import baseline
    assert baseline.SYSTEM_PROMPT == EXPECTED["baseline_system"]
    assert agent.VERIFIER_SYSTEM_PROMPT == EXPECTED["verifier_system"]


def test_temperature_is_0_2(agent):
    from hpo_severity import baseline
    assert agent.REACT_TEMPERATURE == 0.2 and baseline.TEMPERATURE == 0.2
