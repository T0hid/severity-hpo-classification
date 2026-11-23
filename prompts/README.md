# Prompts

The baseline and model-selection prompts are the files below. Placeholders are written as `${name}` and are filled by the code; the rest of the text is sent unchanged. Text files end with one newline that is not part of the prompt. The agent's prompts are built in `hpo_severity/agent.py`. `tests/test_prompts.py` checks the planner, classification, verification, baseline and model-selection prompts, character for character, against `tests/fixtures/expected_prompts.json`.

## Agent (`hpo_severity/agent.py`)

| Function | Step |
|---|---|
| `create_react_planning_prompt()` | ReAct planner: chooses the next action from the state of the term. |
| `transform_query_with_llm()` | Common clinical names of the phenotype, used to write the search queries. |
| `create_final_classification_prompt()` | Classification: role, rules, decision tree, PedsQL domains, category definitions and the required output format, with the numbered sources. |
| `execute_action()` (expand_search) | Related genetic syndromes, when fewer than 5 sources were found. |
| `repair_and_parse_json()` | Repair of a reply that is not valid JSON. |
| `VERIFIER_SYSTEM_PROMPT` | Verification agent: rates one claim against its cited source. |

## Baseline (`baseline/`, used by `hpo_severity/baseline.py`)

`system.txt` and `classification.json`: the classification prompt with no sources. The source list and the rules that refer only to sources are removed, and one line is added: "No sources are provided for this term. Wherever these instructions refer to sources or evidence, use your own clinical knowledge instead." The required output also has a short phenotype definition section (`phenotype_definition`). Placeholders: `hpo_id`, `hpo_name`, `hpo_definition`, `hpo_synonyms`, `hpo_comment`.

## Model selection (`model_selection/`, used by `hpo_severity/model_selection.py`)

`system.txt` and `classification.txt`. Placeholders: `hpo_context` (name, identifier, definition and synonyms of the term) and `retrieved_context` (up to two PubMed abstracts).
