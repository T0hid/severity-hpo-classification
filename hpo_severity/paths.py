"""File locations, and unpacking and checking of the data files.

Large inputs are stored as zip archives in data/ and the published outputs as zip archives in
published_run/. They are unpacked on first use into data/inputs/ and published_run/unpacked/ (both
ignored by git). Every unpacked or small data file is checked against data/checksums.json (SHA-256).
"""
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
INPUTS = DATA / "inputs"
PUBLISHED = ROOT / "published_run"
UNPACKED = PUBLISHED / "unpacked"
CHECKSUMS = DATA / "checksums.json"

# Inputs (data/)
HPO_TERMS = DATA / "hpo_terms.csv"
GROUND_TRUTH = DATA / "ground_truth.xlsx"
GROUND_TRUTH_ALL_LABELLED = DATA / "ground_truth_all_labelled.xlsx"
GENE_LISTS = DATA / "gene_lists"
HPO_2025 = INPUTS / "hp_2025-05-06.obo"
HPO_2026 = INPUTS / "hp_2026-09-01.obo"
HPO_ANNOTATIONS = INPUTS / "phenotype_2026-09-01.hpoa"
MONDO = INPUTS / "mondo_2026-09-01.json"
GENCC = INPUTS / "gencc-submissions_2026-11-15.csv"
HGNC = INPUTS / "hgnc_complete_set_2026-11-18.txt"
INTOLEROME = INPUTS / "human_intolerome_genes.tsv"
HPO_RELEASES = {"2025-05-06": HPO_2025, "2026-09-01": HPO_2026}

# Published run (published_run/)
CLASSIFICATIONS = UNPACKED / "agent_classifications.csv"
GENE_TABLE = UNPACKED / "gene_disease_table.csv"
BASELINE_ANSWERS = PUBLISHED / "baseline_answers.csv"
MODEL_SELECTION_SUMMARY = PUBLISHED / "model_selection_summary.csv"

ARCHIVES = {
    DATA / "ontologies.zip": INPUTS,
    DATA / "gene_disease_sources.zip": INPUTS,
    DATA / "human_intolerome.zip": INPUTS,
    PUBLISHED / "agent_classifications.zip": UNPACKED,
    PUBLISHED / "gene_disease_table.zip": UNPACKED,
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def prepare(check: bool = True) -> None:
    """Unpacks the archives (once) and checks every file listed in data/checksums.json."""
    for archive, target in ARCHIVES.items():
        if not archive.exists():
            continue
        with zipfile.ZipFile(archive) as bundle:
            for name in bundle.namelist():
                if not (target / name).exists():
                    bundle.extract(name, target)
    if check:
        verify_checksums()


def verify_checksums() -> int:
    expected = json.loads(CHECKSUMS.read_text())
    missing, wrong = [], []
    for rel, digest in expected.items():
        path = ROOT / rel
        if not path.exists():
            missing.append(rel)
        elif sha256(path) != digest:
            wrong.append(rel)
    if wrong:
        raise SystemExit("Checksum mismatch (file differs from the one used for the paper): " + ", ".join(wrong))
    if missing:
        raise SystemExit("Missing data files: " + ", ".join(missing) +
                         ". See data/README.md for how to obtain them.")
    return len(expected)
