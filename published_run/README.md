# Saved outputs of the published run

These are the outputs analysed in the paper. `run.py verify` recomputes every code-derived number of the manuscript from them, so the results can be checked without calling a language model.

| File | Content |
|---|---|
| `agent_classifications.zip` | `agent_classifications.csv`: the agent output for all 10,652 HPO terms, one row per term, with the reasoning, the references, the verification of every cited claim and the cost (columns below). |
| `baseline_answers.csv` | The no-retrieval baseline's answer for each of the 898 ground-truth terms: `classification_tier`, `classification_category`, `severity_assessment`, the tier the model wrote (`llm_tier`), the number of tries, the tokens used and the full model reply (`response_json`). |
| `model_selection_summary.csv` | Model selection (Figure S1): one row per model, temperature and top_p with the share of the 14 terms whose tier, category and severity all matched the expert label (`overall_accuracy`), the mean response time per term in seconds (`avg_response_time`) and the cost in USD (`total_cost`). Six models, as in the manuscript. |
| `gene_disease_table.zip` | `gene_disease_table.csv`: one row per gene, disease and HPO phenotype, with the severity class of each gene-disease pair (written by `gene_table.py`; `run.py verify` rebuilds it from the inputs and checks that it is identical). |

## Columns of `agent_classifications.csv`

| Column | Content |
|---|---|
| `hpo_id`, `hpo_name` | The HPO term. |
| `mesh_term` | MeSH heading matched to the term (`Not Found` if none). |
| `classification_category` | Category chosen by the agent. |
| `classification_tier` | Tier (1 to 4, or NFC). Clinical Sign or Laboratory Abnormality is stored with tier 4 and treated as non-tiered in the phenotype-level analyses. |
| `QoL_assessment` | `Affected` or `Not affected`; `NFC` for Out of Scope and NFC terms. |
| `management_profile` | Management categories found in the sources (JSON). |
| `reasoning` | The model's reasoning (JSON), with a `[Ref #]` tag after each claim taken from a source. |
| `all_found_references` | Every source given to the model, by reference number, with its link (JSON). |
| `cited_references` | The references the model cited (JSON). |
| `source_breakdown` | Sources by type (JSON). |
| `rag_results_count`, `pubmed_results_count`, `total_sources` | Number of HPO-index, PubMed and total sources. |
| `statement_support_score` | Weighted support score of the cited claims (0 to 1): direct 1.0, valid inference 0.90, weak inference 0.60, unsupported 0. |
| `supported_statements`, `total_statements` | Claims rated direct or valid inference, and all claims checked. |
| `verification_log` | Every checked claim with its rating and the verifier's reason (JSON). |
| `review_status`, `review_reason` | `OK` or `FLAGGED FOR REVIEW`, and why. |
| `acog_quality_of_life`, `acog_impairment`, `acog_early_onset`, `acog_intervention`, `acog_criteria_met_count`, `acog_criteria_met_names` | ACOG criteria derived from the classification. |
| `react_steps` | Number of planner steps. |
| `input_tokens`, `output_tokens` | Tokens used for the term, verification included. |
| `verification_cost_usd`, `total_cost_usd` | Cost in USD of the verification and of the whole term, at USD 2.50 and 10 per million input and output tokens. |
| `hpo_release` | HPO release the agent read for the term: `2025-05-06` (10,211 terms) or `2026-09-01` (441 terms). |
