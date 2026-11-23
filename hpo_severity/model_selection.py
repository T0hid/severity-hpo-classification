"""
Model selection (Supplementary Figure S1).

Each candidate model is run with every combination of temperature (0.0, 0.2, 0.5) and top_p (1.0,
0.95, 0.9) on 14 expert-labelled HPO terms (GROUND_TRUTH below), with the prompt in
prompts/model_selection/. For each term the prompt includes up to two PubMed review or clinical-trial
abstracts. A configuration is scored by the share of terms whose tier, category and quality-of-life rating all
match. The summary of the paper's run is published_run/model_selection_summary.csv.

Run it through the entry point: python run.py model-selection (needs OPENROUTER_API_KEY and NCBI_EMAIL).
"""
import os
import json
import time
import pickle
import requests
import pandas as pd
import numpy as np
import pronto
import re
import warnings
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product

from Bio import Entrez

from . import paths, prompts

# --- CONFIGURATION ---
ROOT = str(paths.ROOT)


def _load_dotenv(path):
    if os.path.exists(path):
        for raw in open(path, encoding="utf-8"):
            line = raw.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv(os.path.join(ROOT, ".env"))
HPO_ONTOLOGY_PATH = os.environ.get("HPO_ONTOLOGY_PATH", str(paths.HPO_2025))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", os.path.join(ROOT, "outputs", "model_selection"))
CACHE_DIR = os.path.join(OUTPUT_DIR, "cache")


# OpenRouter Configuration
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")   # never write the key in this file
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# The six models compared in the manuscript, with the prices (USD per million input and output
# tokens) used for the cost column.
MODELS_CONFIG = {
    "openai/gpt-5": {"input": 1.25, "output": 10.00}, 
    "x-ai/grok-4": {"input": 3.00, "output": 15.00},
    "google/gemini-2.5-pro": {"input": 2.5, "output": 10.00},
    "anthropic/claude-3.5-sonnet": {"input": 3.00, "output": 15.00},
    "openai/gpt-4o": {"input": 2.50, "output": 10.00},
    "openai/gpt-4o-mini": {"input": 0.15, "output": 0.60},   

}

# Sampling settings compared for every model
TEMPERATURES = [0.0, 0.2, 0.5]
TOP_P_VALUES = [1.0, 0.95, 0.9]
MAX_TOKENS_REASONING = 12000 # token limit for the reasoning template

# Ground Truth Dataset
GROUND_TRUTH = [
    {"hpo_id": "HP:0000738", "hpo_name": "Hallucinations", "tier": 3, "category": "Mental illness", "severity": "Severe"},
    {"hpo_id": "HP:0001402", "hpo_name": "Hepatocellular carcinoma", "tier": 3, "category": "Immunodeficiency/cancer", "severity": "Severe"},
    {"hpo_id": "HP:0003811", "hpo_name": "Neonatal death", "tier": 1, "category": "Shortened life span: infancy", "severity": "Severe"},
    {"hpo_id": "HP:0003819", "hpo_name": "Death in childhood", "tier": 1, "category": "Shortened life span: childhood/adolescence", "severity": "Severe"},
    {"hpo_id": "HP:0002342", "hpo_name": "Intellectual disability, moderate", "tier": 1, "category": "Intellectual disability", "severity": "Severe"},
    {"hpo_id": "HP:0002066", "hpo_name": "Gait ataxia", "tier": 2, "category": "Impaired mobility", "severity": "Severe"},
    {"hpo_id": "HP:0000407", "hpo_name": "Sensorineural hearing impairment", "tier": 3, "category": "Sensory Impairment – Hearing", "severity": "Severe"},
    {"hpo_id": "HP:0000027", "hpo_name": "Azoospermia", "tier": 4, "category": "Infertility", "severity": "Severe"},
    {"hpo_id": "HP:0000543", "hpo_name": "Optic disc pallor", "tier": 3, "category": "Sensory Impairment – Vision", "severity": "Severe"},
    {"hpo_id": "HP:0002414", "hpo_name": "Spina bifida", "tier": 2, "category": "Internal physical malformations", "severity": "Severe"},
    {"hpo_id": "HP:0005321", "hpo_name": "Mandibulofacial dysostosis", "tier": 3, "category": "Dysmorphic feature", "severity": "Severe"},
    {"hpo_id": "HP:0000579", "hpo_name": "Nasolacrimal duct obstruction", "tier": 4, "category": "Internal physical malformations", "severity": "Non-Severe"},
    {"hpo_id": "HP:0000805", "hpo_name": "Enuresis", "tier": 4, "category": "Internal physical malformations", "severity": "Non-Severe"},
    {"hpo_id": "HP:0005280", "hpo_name": "Depressed nasal bridge", "tier": 4, "category": "Dysmorphic feature", "severity": "Non-Severe"}
]


# --- HELPER FUNCTIONS ---
def load_hpo_ontology(obo_path: str) -> Optional[pronto.Ontology]:
    if not os.path.exists(obo_path):
        print(f"HPO ontology file not found at '{obo_path}'")
        return None
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            return pronto.Ontology(obo_path)
    except Exception as e:
        print(f"Error loading ontology: {e}")
        return None

def get_hpo_context(hpo_id: str, ontology: pronto.Ontology) -> Dict:
    try:
        term = ontology[hpo_id]
        return {"id": hpo_id, "name": term.name, "definition": term.definition or "No definition available.", "synonyms": [syn.description for syn in term.synonyms if syn.scope == 'EXACT']}
    except KeyError:
        return {"id": hpo_id, "name": "Unknown", "definition": "No definition available.", "synonyms": []}

def search_pubmed_simple(hpo_name: str, max_results: int = 2) -> List[str]:
    if not hpo_name: return []
    try:
        Entrez.email = os.environ.get("NCBI_EMAIL", "")  # NCBI asks every client to identify itself
        handle = Entrez.esearch(db="pubmed", term=f'"{hpo_name}"[Title/Abstract] AND (review[Publication Type] OR clinical trial[Publication Type])', retmax=max_results)
        record = Entrez.read(handle)
        handle.close()
        if not record.get("IdList"): return []
        fetch_handle = Entrez.efetch(db="pubmed", id=record["IdList"], rettype="abstract", retmode="text")
        abstracts = fetch_handle.read().strip().split("\n\n\n")
        fetch_handle.close()
        return [re.sub(r'\s+', ' ', abstract).strip()[:1000] for abstract in abstracts if abstract]
    except Exception as e:
        print(f"PubMed search error: {e}")
        return []

# --- PROMPT ENGINEERING ---
def create_classification_prompt(hpo_context: Dict, pubmed_abstracts: List[str]) -> str:
    """Creates the detailed, structured classification prompt."""

    hpo_context_str = f"HPO Term: {hpo_context['name']} ({hpo_context['id']})\nDefinition: {hpo_context['definition']}\nSynonyms: {', '.join(hpo_context['synonyms']) or 'None'}"

    retrieved_context_str = ""
    source_count = 1
    for abstract in pubmed_abstracts:
        retrieved_context_str += f"Source [{source_count}] (PubMed): {abstract}\n\n"
        source_count += 1
    if not retrieved_context_str:
        retrieved_context_str = "No external clinical context was retrieved."

    prompt_template = prompts.render_text("model_selection/classification.txt", hpo_context=hpo_context_str, retrieved_context=retrieved_context_str)
    return prompt_template

def extract_json_from_text(text: str) -> Optional[Dict]:
    """Extracts the final JSON object from a model's response. This is robust."""
    if not text or not text.strip():
        return None
    # Prioritize finding the 'FINAL_JSON:' marker for strict compliance.
    match = re.search(r'FINAL_JSON:\s*(\{.*\})', text, re.DOTALL | re.IGNORECASE)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            # If parsing fails, fall through to the more general search.
            pass
    
    # Fallback: Find the last valid JSON object in the string.
    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    all_matches = list(re.finditer(json_pattern, text))
    if all_matches:
        for match in reversed(all_matches):
            try:
                result = json.loads(match.group())
                # Verify it has the expected structure to avoid grabbing other JSONs.
                if all(key in result for key in ['tier', 'category', 'severity']):
                    return result
            except json.JSONDecodeError:
                continue
    return None

# --- API AND EVALUATION ---
def call_openrouter_api(prompt: str, model: str, temperature: float, top_p: float) -> Tuple[Optional[Dict], float, int, int]:
    """Calls the OpenRouter API."""
    start_time = time.time()
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "X-Title": "HPO Classification Agent"
    }
    
    system_message = (
        prompts.text("model_selection/system.txt")
    )
    
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system_message}, {"role": "user", "content": prompt}],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": MAX_TOKENS_REASONING,
    }
    try:
        response = requests.post(OPENROUTER_API_URL, json=payload, headers=headers, timeout=180)
        response.raise_for_status()
        result = response.json()
        elapsed_time = time.time() - start_time
        content = result.get('choices', [{}])[0].get('message', {}).get('content', '')
        
        parsed = extract_json_from_text(content)
        
        if not parsed:
            print(f"Could not extract JSON from {model}.")
            return None, elapsed_time, 0, 0
            
        usage = result.get('usage', {})
        prompt_tokens = usage.get('prompt_tokens', 0)
        completion_tokens = usage.get('completion_tokens', 0)
        
        return parsed, elapsed_time, prompt_tokens, completion_tokens

    except requests.exceptions.Timeout:
        print(f"Timeout for {model}.")
        return None, 180.0, 0, 0
    except requests.exceptions.RequestException as e:
        error_content = e.response.text if e.response else "No response body."
        print(f"API error for {model}: {e}. Body: {error_content[:300]}")
        return None, time.time() - start_time, 0, 0
    except Exception as e:
        print(f"Unexpected error for {model}: {e}")
        return None, time.time() - start_time, 0, 0

def evaluate_prediction(prediction: Optional[Dict], ground_truth: Dict) -> Dict:
    """Compares a model's prediction to the ground truth."""
    if prediction is None:
        return {'tier_correct': False, 'category_correct': False, 'severity_correct': False, 'all_correct': False}
    
    pred_tier = str(prediction.get('tier', 'N/A')).strip()
    true_tier = str(ground_truth['tier']).strip()
    
    pred_cat = str(prediction.get('category', 'N/A')).strip()
    true_cat = str(ground_truth['category']).strip()
    
    pred_sev = str(prediction.get('severity', 'N/A')).strip()
    true_sev = str(ground_truth['severity']).strip()
    
    is_correct = (pred_tier == true_tier and pred_cat == true_cat and pred_sev == true_sev)
    
    return {
        'tier_correct': pred_tier == true_tier,
        'category_correct': pred_cat == true_cat,
        'severity_correct': pred_sev == true_sev,
        'all_correct': is_correct
    }

# --- MAIN EXECUTION ---
def run_hyperparameter_tuning():
    """Main hyperparameter tuning function."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    if not OPENROUTER_API_KEY:
        raise SystemExit("OPENROUTER_API_KEY is not set. Copy .env.example to .env and fill it in.")
    print("=" * 80 + "\nHYPERPARAMETER TUNING WITH ADVANCED PROMPT & CACHING\n" + "=" * 80)
    ontology = load_hpo_ontology(HPO_ONTOLOGY_PATH)
    if not ontology:
        return

    print("\n1. Preparing contexts for ground truth terms...")
    contexts = {gt['hpo_id']: {'hpo_context': get_hpo_context(gt['hpo_id'], ontology), 'ground_truth': gt} for gt in GROUND_TRUTH}
    for hpo_id, data in contexts.items():
        hpo_name = data['hpo_context']['name']
        data['pubmed_abstracts'] = search_pubmed_simple(hpo_name)

    all_results = []
    print("\n2. Starting tuning...")
    for model_name, config in MODELS_CONFIG.items():
        print(f"\n\nTesting model: {model_name}")
        print("=" * 60)
        pricing = {"input": config['input'], "output": config['output']}
        
        for temperature, top_p in product(TEMPERATURES, TOP_P_VALUES):
            print(f"\n  Temp={temperature}, Top-p={top_p}")
            config_results = {
                'model': model_name, 
                'temperature': temperature, 
                'top_p': top_p, 
                'predictions': [], 
                'evaluations': []
            }
            total_in_tok, total_out_tok, total_cost, total_time = 0, 0, 0.0, 0.0

            for i, (hpo_id, context_data) in enumerate(contexts.items()):
                print(f"    Testing {i+1}/{len(contexts)}: {hpo_id} ({context_data['hpo_context']['name']})", end=" ... ")
                
                # reuse a saved answer for this model, setting and term if there is one
                safe_model_name = model_name.replace('/', '_')
                safe_hpo_id = hpo_id.replace(':', '_')
                cache_file = os.path.join(CACHE_DIR, f"{safe_model_name}_t{temperature}_p{top_p}_{safe_hpo_id}.pkl")

                if os.path.exists(cache_file):
                    with open(cache_file, 'rb') as f:
                        prediction, elapsed_time, in_tok, out_tok = pickle.load(f)
                    print("[CACHE]", end=" ... ")
                else:
                    prompt = create_classification_prompt(context_data['hpo_context'], context_data['pubmed_abstracts'])
                    prediction, elapsed_time, in_tok, out_tok = call_openrouter_api(prompt, model_name, temperature, top_p)
                    # Save result to cache
                    with open(cache_file, 'wb') as f:
                        pickle.dump((prediction, elapsed_time, in_tok, out_tok), f)
                # --- End Caching Logic ---

                evaluation = evaluate_prediction(prediction, context_data['ground_truth'])

                # Log results for this run
                config_results['predictions'].append(prediction)
                config_results['evaluations'].append(evaluation)
                total_time += elapsed_time
                total_in_tok += in_tok
                total_out_tok += out_tok
                total_cost += ((in_tok / 1_000_000) * pricing['input']) + ((out_tok / 1_000_000) * pricing['output'])
                
                print("Correct" if evaluation['all_correct'] else ("Incorrect" if prediction else "Failed"))

            # Aggregate results for this configuration
            num_evals = len(config_results['evaluations'])
            if num_evals > 0:
                config_results['overall_accuracy'] = sum(e['all_correct'] for e in config_results['evaluations']) / num_evals
                config_results['avg_time'] = total_time / num_evals
            else:
                config_results['overall_accuracy'] = 0
                config_results['avg_time'] = 0
            
            config_results['total_cost'] = total_cost
            print(f"  Configuration Summary: Accuracy: {config_results['overall_accuracy']:.2%}, Avg Time: {config_results['avg_time']:.2f}s, Cost: ${total_cost:.6f}")
            all_results.append(config_results)

    # Save results
    print("\n\n3. Saving results...")
    dt_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    summary_data = [{
        'model': r['model'], 
        'temperature': r['temperature'], 
        'top_p': r['top_p'], 
        'overall_accuracy': r['overall_accuracy'], 
        'avg_response_time': r['avg_time'], 
        'total_cost': r['total_cost']
    } for r in all_results]
    
    summary_df = pd.DataFrame(summary_data)
    summary_file = os.path.join(OUTPUT_DIR, f"tuning_summary_{dt_str}.csv")
    summary_df.to_csv(summary_file, index=False)
    print(f"  Summary saved to: {summary_file}")

    results_file = os.path.join(OUTPUT_DIR, f"tuning_results_detailed_{dt_str}.json")
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"  Detailed results saved to: {results_file}")

if __name__ == "__main__":
    run_hyperparameter_tuning()
    print("\nAll done. Results are in the output directory.")