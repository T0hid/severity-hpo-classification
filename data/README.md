# Data

Every file below is checked against `data/checksums.json` (SHA-256) by `run.py` before use. The zip archives are unpacked into `data/inputs/` on first use.

## Made for this study

| File | Content | Licence |
|---|---|---|
| `hpo_terms.csv` | The 10,652 HPO terms classified in the paper (`hpo_id`, `hpo_name`). Input of `run.py classify`. | MIT (this repository); term identifiers and names are from HPO |
| `ground_truth.xlsx` | Expert labels of the 898 validation terms. Columns: `Tier` and `category-reviewer 1` (label proposed by the curator), `reviewer 2` and `reviewer 3` (agreement of the two clinical and laboratory geneticists, Y or N), `Final` (Y if the proposed label was kept), `Final classification_tier` and `Final_classification_category` (the label used for scoring). | MIT (this repository) |
| `ground_truth_all_labelled.xlsx` | The 913 labelled terms before the 15 terms that are not among the classified terms were removed (Methods). Same columns. | MIT (this repository) |

## Gene lists (`gene_lists/`)

The gene lists and the Human Intolerome table are copied from the cited publications so that the gene-level results can be reproduced; reuse is subject to the terms of those publications.

| File | Content | Source |
|---|---|---|
| `SF_ACMG_v3.2_genes_MIM.txt` | ACMG secondary findings v3.2: gene symbol, MIM gene number, MIM disorder numbers (Table 1). | Miller et al. 2023, Genet. Med. 25:100866, doi:10.1016/j.gim.2023.100866 |
| `CS_Gregg2021_tier3_genes_MIM.txt` | ACMG Tier 3 carrier screening genes and the MIM numbers printed in each gene's row (Tables 1 to 6). In two rows the MIM number does not match the condition named beside it; `gene_level.py` uses the number of the named condition (BCKDHB 248600, MCPH1 251200). | Gregg et al. 2021, Genet. Med. 23:1793-1806, doi:10.1038/s41436-021-01203-z |
| `MackenziesMission_genes.csv` | Mackenzie's Mission genes with their condition, OMIM and MONDO identifiers. | Kirk et al. 2021, Eur. J. Hum. Genet. 29:79-87, doi:10.1038/s41431-020-0685-x (supplementary gene list) |

## Ontologies (`ontologies.zip`)

| File | Content | Source and licence |
|---|---|---|
| `hp_2025-05-06.obo` | HPO release 2025-05-06. Read by the agent for 10,211 of the 10,652 terms and by the no-retrieval baseline; the default release of `run.py classify`. | Human Phenotype Ontology, https://hpo.jax.org; HPO licence (https://hpo.jax.org/app/license): free to use with citation of the HPO and its version |
| `hp_2026-09-01.obo` | HPO release 2026-09-01. Read by the agent for the other 441 terms; used by the gene-level analysis. | as above |
| `phenotype_2026-09-01.hpoa` | HPO disease annotations of release 2026-09-01 (annotation version 2026-09-02). | as above |
| `mondo_2026-09-01.json` | MONDO release 2026-09-01. | Mondo Disease Ontology, https://mondo.monarchinitiative.org; CC BY 4.0 |

## Gene-disease sources (`gene_disease_sources.zip`)

| File | Content | Source and licence |
|---|---|---|
| `gencc-submissions_2026-11-15.csv` | GenCC submissions export: 30,383 submissions from 19 submitters, latest submission run date 15 November 2026. | Gene Curation Coalition, https://thegencc.org; CC0 1.0 |
| `hgnc_complete_set_2026-11-18.txt` | HGNC complete set, downloaded 18 November 2026. | HUGO Gene Nomenclature Committee, https://www.genenames.org; CC0 |

## Human Intolerome (`human_intolerome.zip`)

| File | Content | Source |
|---|---|---|
| `human_intolerome_genes.tsv` | Human Intolerome gene table, 934 rows. | Yatsenko et al. 2026, Genet. Med. 28:102546 |

GenCC and HGNC change over time, so a new download gives slightly different gene-disease pairs and counts than the paper. Please cite the HPO, MONDO, GenCC and HGNC when you reuse their files.
