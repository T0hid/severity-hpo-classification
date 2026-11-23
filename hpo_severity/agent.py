"""
HPO severity classification agent.

For each HPO term the agent reads the term's HPO record, maps the term to a MeSH heading, writes search queries,
finds the best-matching record in a local index of HPO term records, searches PubMed, and asks the language model
to classify the term from these sources. A verification step then checks every cited claim against its source.

Settings are read from the environment (see .env.example); run.py sets the paths. Run with: python run.py classify
"""
import argparse
import gc
import hashlib
import json
import os
import pickle
import re
import threading
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import pronto
import requests
from Bio import Entrez

ROOT = Path(__file__).resolve().parents[1]


def env(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


# --- Paths ---
INPUT_CSV_PATH = env("INPUT_CSV_PATH", str(ROOT / "data" / "hpo_terms.csv"))
HPO_ONTOLOGY_PATH = env("HPO_ONTOLOGY_PATH", str(ROOT / "data" / "inputs" / "hp_2025-05-06.obo"))
# Folder with docs.faiss, bm25.pkl and documents.pkl: the index of HPO term records.
RAG_INDEX_PATH = env("HPO_INDEX_PATH", str(ROOT / "data" / "inputs" / "hpo_record_index"))
# The embedding model used to build the index. Must match.
RAG_EMBEDDING_MODEL_NAME = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'

RUN_VERSION = env("RUN_VERSION", "run1")
OUTPUT_DIR = env("OUTPUT_DIR", str(ROOT / "outputs" / "classify"))
CACHE_DIR = os.path.join(OUTPUT_DIR, "cache")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
OUTPUT_CSV_PATH = os.path.join(OUTPUT_DIR, f"hpo_classifications_{RUN_VERSION}.csv")
CACHE_FILE = os.path.join(CACHE_DIR, f"classification_cache_{RUN_VERSION}.pkl")
CHECKPOINT_FILE = os.path.join(CHECKPOINT_DIR, f"completed_{RUN_VERSION}.json")
REACT_LOG_FILE = os.path.join(OUTPUT_DIR, f"reasoning_log_{RUN_VERSION}.jsonl")

# --- Language model (OpenRouter, OpenAI-compatible chat completions endpoint) ---
OPENAI_API_URL = env("OPENROUTER_API_URL", "https://openrouter.ai/api/v1/chat/completions")
OPENAI_API_KEY = env("OPENROUTER_API_KEY")
MODEL_NAME = env("MODEL_NAME", "openai/gpt-4o")

# --- NCBI ---
NCBI_EMAIL = env("NCBI_EMAIL")
NCBI_API_KEY = env("NCBI_API_KEY")       # optional; raises the NCBI rate limit
if NCBI_API_KEY:
    Entrez.api_key = NCBI_API_KEY

# --- Cost calculation (USD per million tokens, gpt-4o) ---
INPUT_TOKEN_COST_PER_MILLION = float(env("INPUT_TOKEN_COST_PER_MILLION", "2.5"))
OUTPUT_TOKEN_COST_PER_MILLION = float(env("OUTPUT_TOKEN_COST_PER_MILLION", "10"))

# --- ReAct and index search settings ---
MAX_REACT_ITERATIONS = 12
REACT_TEMPERATURE = 0.2
RAG_TOP_K = 5
RAG_MINIMUM_RESULTS = 5
RAG_SCORE_THRESHOLD = 0.75
MAX_WORKERS = int(env("MAX_WORKERS", "9"))


# --- 2. INDEX OF HPO TERM RECORDS ---
class HPORecordIndex:
    """Hybrid search (PubMedBERT embeddings with FAISS, and BM25 keywords) over the index of HPO term records."""
    def __init__(self, index_path: str):
        print("Loading the HPO record index...")
        self.index_path = index_path
        self.documents = None
        self.index_faiss = None
        self.bm25 = None
        self.model = None
        self.authority_weights = {"GeneReviews": 1.2, "Orphanet": 1.1, "StatPearls": 1.05, "MedlinePlus": 1.0}
        self.severity_markers = ["severe", "fatal", "life-threatening", "degenerative", "progressive", "disabling"]
        self._load_components()

    def _load_components(self):
        try:
            import faiss
            import rank_bm25  # noqa: F401  (needed to unpickle bm25.pkl)
            from sentence_transformers import SentenceTransformer
            print(f"  - Loading index files from '{self.index_path}'...")
            self.index_faiss = faiss.read_index(os.path.join(self.index_path, 'docs.faiss'))
            with open(os.path.join(self.index_path, 'bm25.pkl'), 'rb') as f:
                self.bm25 = pickle.load(f)
            with open(os.path.join(self.index_path, 'documents.pkl'), 'rb') as f:
                self.documents = pickle.load(f)
            self.model = SentenceTransformer(RAG_EMBEDDING_MODEL_NAME)
            print(f"Index loaded with {len(self.documents):,} documents.")
        except (FileNotFoundError, ImportError) as e:
            print(f"ERROR: the HPO record index could not be loaded: {e}")
            self.documents = None

    def is_available(self) -> bool:
        return all([self.documents, self.index_faiss, self.bm25, self.model])


    def _calculate_composite_score(self, chunk: Dict, vector_dist: float, keyword_score: float) -> float:
        source_bonus = self.authority_weights.get(chunk['metadata']['source'], 0.9)
        entity_bonus = 0.1 if any(marker in chunk['content'].lower() for marker in self.severity_markers) else 0.0
        vector_similarity = 1 / (1 + vector_dist)
        final_score = (0.6 * vector_similarity) + (0.4 * (keyword_score / 10))
        final_score = final_score * source_bonus + entity_bonus
        return final_score

    def search(self, queries: List[str]) -> List[Dict[str, Any]]:
        """
        Hybrid search (vector and keyword) of the index for a list of queries.
        """
        if not self.is_available():
            print("  - Index not available, skipping search.")
            return []
        
        print(f"  - Searching the HPO record index with {len(queries)} queries...")
        
        # --- 1. BATCH VECTOR SEARCH (FAISS) ---
        print(f"    - Batch encoding {len(queries)} queries for FAISS...")
        query_embeddings = self.model.encode(queries, convert_to_tensor=False).astype('float32')
        
        print(f"    - Performing batched FAISS search...")
        num_faiss_candidates = RAG_TOP_K * 5 
        distances, indices = self.index_faiss.search(query_embeddings, num_faiss_candidates)
        
        candidate_docs = {}
        for query_idx in range(len(queries)):
            for i, doc_idx in enumerate(indices[query_idx]):
                if doc_idx < 0 or doc_idx >= len(self.documents): continue
                
                if doc_idx not in candidate_docs:
                    candidate_docs[doc_idx] = {'vector_dist': distances[query_idx][i], 'keyword_score': 0.0}
                else:
                    candidate_docs[doc_idx]['vector_dist'] = min(candidate_docs[doc_idx]['vector_dist'], distances[query_idx][i])
        
        print(f"    - Found {len(candidate_docs)} unique candidates from vector search.")
        
        # --- 2. KEYWORD SEARCH (BM25) ---
        print(f"    - Performing BM25 keyword search...")
        num_bm25_candidates = RAG_TOP_K * 5
        
        for query in queries:
            tokenized_query = query.lower().split(" ")
            doc_scores = self.bm25.get_scores(tokenized_query)
            
            top_n_indices = np.argpartition(doc_scores, -num_bm25_candidates)[-num_bm25_candidates:]
        
            for doc_idx in top_n_indices:
                score = doc_scores[doc_idx]
                if score <= 0: continue
    
                if doc_idx in candidate_docs:
                    candidate_docs[doc_idx]['keyword_score'] += score
                else:
                    candidate_docs[doc_idx] = {'vector_dist': 999.0, 'keyword_score': score}
        
        print(f"    - Merged keyword results. Total unique candidates: {len(candidate_docs)}")
        
        # --- 3. RERANK AND FINALIZE ---
        print(f"    -  Reranking top candidates...")
        reranked_results = []
        for idx, scores in candidate_docs.items():
            doc = self.documents[idx]
            final_score = self._calculate_composite_score(doc, scores['vector_dist'], scores['keyword_score'])
            reranked_results.append({'doc': doc, 'score': final_score})
        
        reranked_results.sort(key=lambda x: x['score'], reverse=True)
        
        final_results_formatted = []
        seen_articles = set()
        for res in reranked_results:
            if len(final_results_formatted) >= RAG_TOP_K:
                break
                
            doc = res['doc']
            meta = doc['metadata']
            article_id = f"{meta.get('source', 'Unknown')}_{meta.get('article_title', 'Unknown')}"
    
            if article_id not in seen_articles:
                url = meta.get('url', f"Local-RAG-Source_{meta.get('source', 'Unknown')}")
                formatted_result = {
                    'source': f"RAG-{meta.get('source', 'Unknown')}",
                    'content': doc.get('content', 'N/A'),
                    'metadata': meta,
                    'similarity_score': res.get('score', 0.0),
                    'article_title': meta.get('article_title', 'N/A'),
                    'section_title': meta.get('section_title', 'N/A'),
                    'url': url
                }
                final_results_formatted.append(formatted_result)
                seen_articles.add(article_id)
        
        print(f"  - Found {len(final_results_formatted)} relevant, unique documents from batched search.")
        return final_results_formatted

# --- 3. REACT AGENT CLASSES AND ENUMS ---
class AgentAction(Enum):
    GET_HPO_CONTEXT = "get_hpo_context"
    GET_MESH_TERM = "get_mesh_term" 
    TRANSFORM_QUERY = "transform_query"
    SEARCH_RAG = "search_rag"
    SEARCH_PUBMED = "search_pubmed"
    SEARCH_PUBMED_BROAD = "search_pubmed_broad"
    EXPAND_SEARCH = "expand_search" 
    CLASSIFY = "classify"
    INSUFFICIENT_INFO = "insufficient_info"
@dataclass
class ReActStep:
    iteration: int
    thought: str
    action: AgentAction
    action_input: Dict[str, Any]
    observation: str
    raw_response: Optional[Dict] = None


@dataclass
class ReActContext:
    hpo_id: str
    hpo_context: Optional[Dict] = None
    mesh_term: Optional[str] = None 
    mesh_term_searched: bool = False  
    mesh_term_attempted: bool = False  
    transformed_queries: List[str] = field(default_factory=list)
    queries_transformed: bool = False 
    rag_articles: List[Dict] = field(default_factory=list)
    pubmed_articles: List[Dict] = field(default_factory=list)
    steps: List[ReActStep] = field(default_factory=list)
    final_classification: Optional[Dict] = None
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    _reference_urls: List[str] = field(default_factory=list)
    rag_searched: bool = False
    rag_broad_searched: bool = False
    pubmed_searched: bool = False
    pubmed_broad_searched: bool = False
    search_expanded: bool = False


# --- 4. MANAGEMENT CLASSES ---
class ClassificationCache:
    def __init__(self, cache_path: str):
        self.cache_path = cache_path
        self.cache = self._load_cache()
        if os.path.exists(self.cache_path):
            print(f"Cache loaded with {len(self.cache)} items from '{cache_path}'.")
        else:
            print(f"No existing cache found at '{cache_path}'. A new one will be created.")

    def _load_cache(self) -> Dict:
        if not os.path.exists(self.cache_path): return {}
        try:
            with open(self.cache_path, 'rb') as f: return pickle.load(f)
        except (pickle.UnpicklingError, EOFError, FileNotFoundError) as e:
            print(f"Warning: Could not load cache file. It might be corrupted or empty. Error: {e}")
            return {}

    def _save_cache(self):
        try:
            with open(self.cache_path, 'wb') as f:
                pickle.dump(self.cache, f)
        except IOError as e:
            print(f"CRITICAL: Could not write to cache file at '{self.cache_path}'. Error: {e}")
            print("  - Please check directory permissions and available disk space.")

    def _get_key(self, hpo_id: str) -> str:
        return hashlib.md5(f"{hpo_id}-{MODEL_NAME}".encode()).hexdigest()

    def get(self, hpo_id: str) -> Optional[Dict]:
        return self.cache.get(self._get_key(hpo_id))

    def set(self, hpo_id: str, result: Dict):
        key = self._get_key(hpo_id)
        self.cache[key] = result
        self._save_cache()

class CheckpointManager:
    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = checkpoint_path
        self.completed_ids = self._load_checkpoint()
        print(f"Checkpoint loaded. {len(self.completed_ids)} terms already completed.")

    def _load_checkpoint(self) -> Set[str]:
        if not os.path.exists(self.checkpoint_path): return set()
        try:
            with open(self.checkpoint_path, 'r') as f: return set(json.load(f))
        except (json.JSONDecodeError, FileNotFoundError): return set()

    def add_completed(self, hpo_id: str):
        self.completed_ids.add(hpo_id)
        with open(self.checkpoint_path, 'w') as f: json.dump(list(self.completed_ids), f)

    def is_completed(self, hpo_id: str) -> bool:
        return hpo_id in self.completed_ids

class ResultsManager:
    def __init__(self, output_path: str):
        self.output_path = output_path
        self.header_written = os.path.exists(output_path)
        if not self.header_written:
            print(f"A new results file will be created at '{output_path}'.")
        else:
            print(f"Will append to existing results file at '{output_path}'.")

    def save_result(self, result_df: pd.DataFrame):
        try:
            result_df.to_csv(self.output_path, mode='a', index=False, header=not self.header_written)
            self.header_written = True
        except IOError as e:
            print(f"CRITICAL: Could not write to results file at '{self.output_path}'. Error: {e}")


class ReActLogger:
    def __init__(self, log_path: str, lock: threading.Lock):
        self.log_path = log_path
        self.lock = lock
        print(f"ReActLogger initialized. Will append logs to '{log_path}'.")

    def log_react_trace(self, hpo_id: str, context: ReActContext):
        # The lock ensures only one thread writes to the file at a time
        with self.lock:
            trace = {
                "hpo_id": hpo_id,
                "total_steps": len(context.steps),
                "rag_results_count": len(context.rag_articles),
                "pubmed_results_count": len(context.pubmed_articles),
                "steps": [
                    {
                        "iteration": step.iteration,
                        "thought": step.thought,
                        "action": step.action.value,
                        "action_input": step.action_input,
                        "observation": step.observation[:500] + "..." if len(step.observation) > 500 else step.observation
                    }
                    for step in context.steps
                ],
                "final_classification": context.final_classification
            }
            try:
                # Open the file in 'append' mode ('a') and write the new trace as one line
                with open(self.log_path, 'a') as f:
                    f.write(json.dumps(trace) + '\n') 
            except IOError as e:
                print(f"CRITICAL: Could not append to log file at '{self.log_path}'. Error: {e}")
# --- 5. TOOL IMPLEMENTATIONS ---
def load_hpo_ontology(obo_path: str) -> Optional[pronto.Ontology]:
    print(f"Loading HPO ontology from '{obo_path}'...")
    if not os.path.exists(obo_path):
        print(f"Error: HPO ontology file not found at '{obo_path}'")
        return None
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UnicodeWarning)
            return pronto.Ontology(obo_path)
    except Exception as e:
        print(f"Error loading ontology with pronto: {e}")
        return None

def get_rich_hpo_context(hpo_id: str, ontology: pronto.Ontology) -> Dict[str, Any]:
    try:
        term = ontology[hpo_id]
        synonyms = [syn.description for syn in term.synonyms if syn.scope == 'EXACT']
        parents = [p.name for p in term.superclasses(distance=1, with_self=False)]
        definition = term.definition if term.definition is not None else ""
        comment = term.comment if term.comment is not None else ""
        return {
            "id": hpo_id, "name": term.name, "definition": definition,
            "synonyms": synonyms, "comment": comment, "parents": parents,
        }
    except KeyError:
        return {"error": f"Term {hpo_id} not found in ontology."}


def get_mesh_term(hpo_name: str, synonyms: List[str]) -> Optional[str]:
    """
    Robust MeSH term lookup with proper error handling
    """
    print(f"  - Searching MeSH database for '{hpo_name}'...")
    Entrez.email = NCBI_EMAIL
    
    search_terms = [hpo_name] + synonyms
    
    for term in search_terms:
        try:
            # Clean the search term
            clean_term = term.strip()
            
            # Try a simple search first
            handle = Entrez.esearch(db="mesh", term=clean_term, retmax=1)
            search_result = Entrez.read(handle)
            handle.close()
            
            if search_result.get("IdList"):
                mesh_id = search_result["IdList"][0]
                
                try:
                    # Try to fetch the actual term name
                    fetch_handle = Entrez.efetch(db="mesh", id=mesh_id, retmode="xml")
                    xml_data = fetch_handle.read()
                    fetch_handle.close()
                    
                    # Only try to parse if we got valid XML
                    if xml_data and not xml_data.startswith(b'<!DOCTYPE'):
                        root = ET.fromstring(xml_data)
                        # Look for the descriptor name
                        desc_element = root.find(".//DescriptorName/String")
                        if desc_element is not None and desc_element.text:
                            mesh_term = desc_element.text
                            print(f"  - Found MeSH term: '{mesh_term}'")
                            return mesh_term
                    
                    # If we can't parse the XML, just return that we found a match
                    print(f"  - Found MeSH ID: {mesh_id}")
                    return f"MeSH_ID:{mesh_id}"
                    
                except ET.ParseError:
                    # If XML parsing fails, just use the ID
                    print(f"  - Found MeSH ID: {mesh_id} (couldn't parse details)")
                    return f"MeSH_ID:{mesh_id}"
                except Exception:
                    # Any other error in fetching, still use the ID
                    return f"MeSH_ID:{mesh_id}"
                    
        except Exception as e:
            # Skip to next term if this one fails
            continue
    
    print(f"  - No MeSH term found for '{hpo_name}'")
    return None


def search_pubmed(hpo_name: str, synonyms: List[str], mesh_term: Optional[str] = None) -> List[Dict[str, str]]:
    """
    Searches PubMed with a robust cascading strategy to find relevant medical literature.

    The function follows a prioritized search hierarchy:
    1.  It first attempts several highly specific queries to find an INHERITED or
        CONGENITAL context for the given HPO term, prioritizing MeSH terms if available.
    2.  If all specific searches fail to yield results, it automatically falls back
        to broader searches for the term itself to ensure some context is retrieved.
    3.  It fetches article details and filters for those with meaningful abstracts.

    Args:
        hpo_name: The primary name of the HPO term (e.g., "Bladder polyp").
        synonyms: A list of synonyms for the HPO term.
        mesh_term: An optional, precise MeSH term to improve search accuracy.

    Returns:
        A list of dictionaries, where each dictionary represents a relevant PubMed
        article with its source, ID, URL, title, and abstract. Returns an empty
        list if no articles are found after all strategies are exhausted.
    """
    print(f"Searching PubMed for '{hpo_name}' with cascading logic...")
    Entrez.email = NCBI_EMAIL
    article_ids = []

    # Define a standard set of keywords for finding inherited disease context
    INHERITED_DISEASE_KEYWORDS = (
        "(congenital[Title/Abstract] OR "
        "inherited[Title/Abstract] OR "
        "hereditary[Title/Abstract] OR "
        "genetic[Title/Abstract] OR "
        "familial[Title/Abstract] OR "
        "syndrome[Title/Abstract])"
    )

    # Combine the HPO name and its synonyms into a single, unique list for querying
    all_terms = list(set([hpo_name] + synonyms))
    # Format the terms for a PubMed Title/Abstract search query
    term_query = " OR ".join([f'"{t}"[Title/Abstract]' for t in all_terms])

    # --- Define the sequence of search strategies, from most specific to broadest ---
    strategies = []

    # Strategy 1 (Highest Priority): Precise MeSH term + inherited context. This is the gold standard.
    if mesh_term and "MeSH_ID" not in mesh_term:
        strategies.append(
            ("MeSH Term + Inherited Context", f'("{mesh_term}"[MeSH Terms]) AND {INHERITED_DISEASE_KEYWORDS}')
        )

    # Strategy 2: All known terms (HPO name + synonyms) + inherited context.
    strategies.append(
        ("All Terms + Inherited Context", f'({term_query}) AND {INHERITED_DISEASE_KEYWORDS}')
    )

    # Strategy 3 (Fallback): Just the precise MeSH Term, if available.
    if mesh_term and "MeSH_ID" not in mesh_term:
        strategies.append(
            ("Fallback: Precise MeSH Term", f'"{mesh_term}"[MeSH Terms]')
        )

    # Strategy 4 (Final Fallback): All terms without the inherited context filter.
    strategies.append(
        ("Fallback: All Terms (Broad Search)", f'({term_query})')
    )

    # --- Execute strategies sequentially until one succeeds ---
    for name, query in strategies:
        if article_ids:  # Stop if we found results from a higher-priority strategy
            break

        try:
            print(f"  - Attempting strategy: {name}")
            handle = Entrez.esearch(db="pubmed", term=query, retmax="20", sort="relevance")
            record = Entrez.read(handle)
            handle.close()
            time.sleep(0.4) # Be polite to the NCBI API

            found_ids = record.get("IdList", [])
            if found_ids:
                print(f"  - Success! Found {len(found_ids)} potential article IDs.")
                article_ids = found_ids
        except Exception as e:
            print(f"  - Error during '{name}' search: {e}. Trying next strategy.")
            continue

    # --- Fetch and Parse article details ---
    if not article_ids:
        print("  - All PubMed search strategies failed to find articles.")
        return []

    try:
        print(f"  - Fetching details for up to {len(article_ids)} articles...")
        fetch_handle = Entrez.efetch(db="pubmed", id=article_ids, rettype="abstract", retmode="xml")
        xml_data = fetch_handle.read()
        fetch_handle.close()
        time.sleep(0.4)

        if not xml_data or xml_data.strip() == b'':
            print("  - PubMed returned an empty response during fetch.")
            return []

        root = ET.fromstring(xml_data)
        articles_data = root.findall("PubmedArticle")

        retrieved_articles = []
        for article_element in articles_data:
            medline_citation = article_element.find('MedlineCitation')
            if medline_citation is None: continue
            
            pmid_element = medline_citation.find('PMID')
            pmid = pmid_element.text if pmid_element is not None else 'N/A'
            
            article_info = medline_citation.find('Article')
            if article_info is None: continue

            title_element = article_info.find('ArticleTitle')
            title = title_element.text if title_element is not None else 'No Title'
            
            # Join all parts of the abstract together
            abstract_texts = [el.text for el in article_info.findall('.//AbstractText') if el.text]
            abstract = " ".join(abstract_texts)

            # Simple filter for meaningful abstracts to ensure quality of results
            if abstract and len(abstract) > 100:
                retrieved_articles.append({
                    "source": "PubMed",
                    "id": pmid,
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    "title": title,
                    "abstract": abstract
                })

        print(f"  - Successfully fetched and filtered {len(retrieved_articles)} articles.")
        return retrieved_articles[:10] # Return a maximum of 10 high-quality articles

    except Exception as e:
        print(f"  - An unexpected error occurred during PubMed article fetching: {e}")
        return []


def transform_query_with_llm(session: requests.Session, hpo_context: Dict[str, Any], mesh_term: Optional[str] = None) -> List[str]:
    """
    Generates targeted search queries by first brainstorming related clinical terms
    and then applying a systematic template to find congenital links.
    """
    print("Transforming HPO term with enhanced clinical term expansion...")
    
    hpo_name = hpo_context.get('name', 'N/A')
    
    # --- STEP 1: Brainstorm related clinical terms with a targeted LLM call ---
    brainstorm_prompt = f"""
The HPO term is "{hpo_name}". What are the 1-3 most common clinical diagnoses or synonyms for this phenotype?
For example, for "Constrictive median neuropathy," the answer should include "Carpal Tunnel Syndrome."
For "Ectopia lentis," the answer should include "dislocated lens."

Respond with a JSON object with a single key "related_terms" containing a list of strings.
Example: {{"related_terms": ["Carpal Tunnel Syndrome", "Median nerve entrapment"]}}
"""
    system_prompt = "You are a medical terminologist providing common names for technical phenotypes."
    
    related_terms = []
    try:
        response_json, _ = call_openai_api(session, brainstorm_prompt, system_prompt, temperature=REACT_TEMPERATURE)
        if response_json and "related_terms" in response_json:
            related_terms = response_json["related_terms"]
            print(f"  - Found related clinical terms: {', '.join(related_terms)}")
    except Exception as e:
        print(f"  - LLM brainstorming failed: {e}. Proceeding with base term only.")

    # --- STEP 2: Create a comprehensive list of all terms to search ---
    all_search_terms = list(dict.fromkeys([hpo_name] + related_terms + hpo_context.get('synonyms', [])))
    if mesh_term and "MeSH" not in str(mesh_term):
        all_search_terms.append(mesh_term)
    
    print(f"  - Assembling queries for: {', '.join(all_search_terms)}")

    # --- STEP 3: Apply a systematic template to all gathered terms ---
    # This ensures we are always looking for the genetic/congenital connection.
    final_queries = []
    for term in all_search_terms:
        final_queries.extend([
            f'"{term}" congenital OR genetic syndrome',
            f'"{term}" AND (hereditary OR familial)',
            f'"{term}" pediatric OR neonatal onset',
            f'"{term}" prognosis when presenting in childhood',
            f'"{term}" management in inherited disorders'
        ])
    
    # --- STEP 4: Deduplicate and return the final list ---
    final_queries = list(dict.fromkeys(final_queries))
    print(f"Generated {len(final_queries)} unique, targeted queries.")
    return final_queries[:15] # Return a generous but capped number of queries



# --- 6. REACT AGENT PROMPTS ---
def create_react_planning_prompt(context: ReActContext) -> str:
    """Creates the planning prompt with explicit state tracking to prevent loops."""
    state_summary = f"HPO Term to classify: {context.hpo_id}\n\n"
    if context.hpo_context:
        state_summary += f"✓ HPO Context retrieved: Name: {context.hpo_context.get('name', 'N/A')}\n"
        if context.mesh_term_searched:
            if context.mesh_term:
                state_summary += f"✓ MeSH Term Found: {context.mesh_term}\n"
            else:
                state_summary += f"✓ MeSH Term Searched: None Found\n"
        else:
            state_summary += f"✗ MeSH Term Not Yet Searched\n"
        if context.transformed_queries:
            state_summary += f"✓ Targeted Queries Generated: {len(context.transformed_queries)} questions created.\n"
        else:
            state_summary += f"✗ Targeted Queries Not Generated\n"
    else:
        state_summary += "✗ HPO Context not yet retrieved\n"
        state_summary += f"MeSH Search Attempted: {'✓' if context.mesh_term_attempted else '✗'}\n"
        if context.mesh_term_attempted:
            state_summary += f"MeSH Term Found: {context.mesh_term if context.mesh_term else 'None'}\n"
    
    state_summary += "\nSearch Actions Performed:\n"
    state_summary += f"- RAG Search: {'✓' if context.rag_searched else '✗'}\n"
    state_summary += f"- PubMed Search: {'✓' if context.pubmed_searched else '✗'}\n"
    state_summary += f"- Broad PubMed Search: {'✓' if context.pubmed_broad_searched else '✗'}\n"
    state_summary += f"- Search Expanded: {'✓' if context.search_expanded else '✗'}\n"

    total_sources = len(context.rag_articles) + len(context.pubmed_articles)
    state_summary += f"\nSources Found: {total_sources} (RAG: {len(context.rag_articles)}, PubMed: {len(context.pubmed_articles)})\n"
    
    if context.steps:
        state_summary += "\nPrevious step:\n"
        last_step = context.steps[-1]
        state_summary += f"  - Step {last_step.iteration}: {last_step.action.value} -> {last_step.observation[:100]}...\n"
    
    return f"""You are a strategic and efficient ReAct agent specialized in classifying HPO (Human Phenotype Ontology) terms. Your goal is to gather sufficient information to make an accurate classification in the minimum number of steps. You must strictly follow the Decision Process.

SEARCH STRATEGY PRIORITY:
1. Get HPO context first
2. Search for MeSH term to improve search quality
3. Transform queries using HPO and MeSH information
4. ALWAYS search RAG database first (local medical knowledge base).
5. If RAG provides insufficient results (< {RAG_MINIMUM_RESULTS} relevant sources), then search external sources like PubMed.
6. External searches are supplementary to RAG, not replacements.

Current State:
{state_summary}

Available Actions (use EXACTLY these names):
- get_hpo_context
- get_mesh_term
- transform_query
- search_rag
- search_pubmed
- search_pubmed_broad
- expand_search
- classify
- insufficient_info

Decision Process:
Follow these steps in strict order. Do not skip steps.

1.  **Information Gathering:**
    - If HPO context is not retrieved, you MUST use `get_hpo_context`.
    - If HPO context is retrieved but MeSH term is not searched, you MUST use `get_mesh_term`.
    - If MeSH term is searched but queries are not transformed, you MUST use `transform_query`.

2.  **Initial Evidence Search:**
    - If queries are transformed but RAG has not been searched, you MUST use `search_rag`.
    - If RAG search is done but total sources are < {RAG_MINIMUM_RESULTS} AND PubMed has not been searched, you MUST use `search_pubmed`.
    - If PubMed search is done but total sources are < {RAG_MINIMUM_RESULTS} AND Broad PubMed has not been searched, you MUST use `search_pubmed_broad`.

3.  **Search Expansion (The Critical Checkpoint):**
   - If ALL initial searches are complete, AND total sources are < {RAG_MINIMUM_RESULTS}, AND you have NOT yet used `expand_search` (Search Expanded is ✗), you MUST use `expand_search`. This is your ONLY attempt to gather more evidence.

4.  **Final Action:**
    - If the conditions for steps 1, 2, and 3 are not met, it's time to conclude.
    - You MUST use the `classify` action. This is the mandatory final step.
    - The ONLY exception is if you have already attempted `expand_search` and STILL have ZERO sources. In that specific case, you may use `insufficient_info`.

Respond with a JSON object containing:
{{
  "thought": "Your reasoning about what to do next based on the Current State and Decision Process.",
  "action": "One of the exact action names listed above",
  "action_input": {{}}
}}

IMPORTANT: Do not repeat a search action if its flag is already '✓'. Move to the next step in the Decision Process.
"""


def create_final_classification_prompt(context: ReActContext) -> str:
    """
    Builds the final classification prompt with the exclusion rules and decision trees.
    References are listed in full, with URLs for PubMed articles.
    """
    rich_context = context.hpo_context
    all_articles = context.rag_articles + context.pubmed_articles
    
    valid_articles = [a for a in all_articles if a.get('content') or a.get('abstract')]
    retrieved_sources_list = []
    reference_dict = {}

    # Numbered list of the sources given to the model
    for i, article in enumerate(valid_articles):
        ref_id = i + 1
        citation_tag = f"[Ref {ref_id}]"
        source_name = article.get('source', 'Unknown')
        title = article.get('article_title') or article.get('title', 'N/A')
        url = article.get('url', '')
        
        # Get metadata
        metadata = article.get('metadata', {})
        
        # Generate searchable URLs for sources that don't have them
        if not url or url == 'N/A' or url == '' or url.startswith('Local-RAG'):
            
            if 'StatPearls' in source_name or 'GeneReviews' in source_name:
                # ALWAYS use the HPO term being classified for search
                hpo_name = context.hpo_context.get('name', '')
                if hpo_name:
                    search_term = hpo_name.replace(' ', '+')
                else:
                    # Fallback only if HPO name somehow missing
                    search_term = "medical+condition"
                
                if 'StatPearls' in source_name:
                    url = f"https://www.ncbi.nlm.nih.gov/books/?term={search_term}+StatPearls"
                elif 'GeneReviews' in source_name:
                    url = f"https://www.ncbi.nlm.nih.gov/books/?term={search_term}+GeneReviews"
                    
            elif 'Orphanet' in source_name:
                # Orphanet should have proper URLs
                orpha_code = metadata.get('orpha_code', '')
                if orpha_code and orpha_code != 'Unknown':
                    url = f"https://www.orpha.net/consor/cgi-bin/OC_Exp.php?lng=EN&Expert={orpha_code}"
                else:
                    # Use HPO term for search
                    hpo_name = context.hpo_context.get('name', '')
                    search_term = hpo_name.replace(' ', '+') if hpo_name else title.replace(' ', '+')
                    url = f"https://www.orpha.net/consor/cgi-bin/Disease_Search.php?lng=EN&search={search_term}"
                    
            elif 'MedlinePlus' in source_name:
                # Generate MedlinePlus URL using HPO term
                hpo_name = context.hpo_context.get('name', '')
                condition = hpo_name if hpo_name else title
                slug = condition.lower().replace(' ', '-').replace('/', '-')
                url = f"https://medlineplus.gov/genetics/condition/{slug}/"
                
            elif source_name == 'PubMed':
                # PubMed URLs should already be good
                pmid = article.get('id', '')
                if pmid:
                    url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        
        # Store the reference with URL
        reference_string = f"{source_name}: {url}"
        reference_dict[citation_tag] = reference_string
        
        # Continue with source info
        source_info = {
            "reference_id": citation_tag,
            "source_type": source_name,
            "article_title": title,
            "content_snippet": article.get('content') or article.get('abstract', 'N/A')
        }
        retrieved_sources_list.append(source_info)

    # Store references for CSV
    context._reference_urls = reference_dict
    max_ref_number = len(valid_articles)

    prompt_dict = {
        "role_and_objective": {
            "role": "You are a board-certified clinical geneticist specializing in dysmorphology and the annotation of rare congenital disorders. Your primary responsibility is to ensure every classification is conservative and rigorously backed by the provided evidence.",
            "objective": "Your task is to analyze the provided HPO term and clinical sources to classify the term's severity and functional impact according to a strict set of rules. You must produce a single JSON object as your final output."
        },
        "UNIVERSAL_CLASSIFICATION_RULES": [
            "Rule 1: The phenotype must be evaluated IN ISOLATION, not as part of a syndrome",
            "Rule 2: Medical interventions/procedures are NOT phenotypes - classify as NFC",
            "Rule 3: Each category is MUTUALLY EXCLUSIVE - a term can only belong to ONE",
            "Rule 4: When in doubt between severe and non-severe, check if functional domains are affected",
            "Rule 5: Evidence must be SPECIFIC to this phenotype, not to related conditions",
            "Rule 6: Distinguish between gross motor (walking/locomotion) and fine motor (hand/arm) impairments"
        ],
        "input_data": {
            "hpo_term_to_classify": {
                "id": rich_context.get('id', 'N/A'),
                "name": rich_context.get('name', 'N/A'),
                "definition": rich_context.get('definition', 'N/A'),
                "synonyms": rich_context.get('synonyms', []),
                "comment": rich_context.get('comment', 'N/A')
            },
            "retrieved_clinical_sources": retrieved_sources_list if retrieved_sources_list else [{"reference_id": "N/A", "content_snippet": "No relevant clinical articles were found."}]
        },
        "guidelines_and_instructions": {
            "core_principles": [
                "**RULE 1 (NON-NEGOTIABLE): CITE ONLY THE DIRECT SOURCE.** Every factual claim you write that ends with a `[Ref #]` tag MUST be found within that single reference. You are strictly forbidden from writing a sentence that combines information from multiple sources. If Ref 1 has the definition and Ref 2 has the impact, you MUST write two separate sentences. Violation of this rule will result in an incorrect analysis.",
                "NEGATIVE PERSONA: Your knowledge is strictly limited to the text in the `retrieved_clinical_sources`. Do not invent, assume, or use any external information whatsoever.",
                "DIRECT QUOTES & CITATIONS: Every factual claim in your reasoning MUST be supported by a direct quote from the source text and end with the appropriate `[Ref #]` tag.",
                "NO INFERENCE BEYOND SOURCES: Do not make logical inferences that go beyond what is explicitly stated in the text.",
                "VERIFICATION CHECK: Before finalizing, verify each claim can be found verbatim or near-verbatim in the cited source.",
                "ADMIT UNCERTAINTY: If the provided sources do not contain information for a specific part of the analysis (e.g., management, onset, prognosis), you MUST explicitly state 'Information not available in the provided sources for this topic.' Do not infer or invent information to fill gaps.",
                "MORTALITY OVERRIDE: If the HPO term or its definition explicitly describes mortality, death, or a life-ending event, you MUST bypass the functional domain assessment. Immediately classify the `severity_assessment` as 'Severe' and the `tier` as '1'. The reasoning should state that mortality is the ultimate severe outcome.",
                "CONGENITAL FIRST PRINCIPLE: Your clinical focus is on congenital conditions. Therefore, when evaluating a phenotype's origin, you MUST prioritize its congenital nature. If the sources mention that the condition CAN be congenital, even if it is rare or less common than acquired forms, you MUST proceed with a full severity and functional impact assessment. Only classify the term as 'Out of Scope: Acquired/Multifactorial' if the evidence confirms it is **exclusively** acquired, with no mention of congenital or neonatal onset.",
                "SURGICAL INTERVENTION OVERRIDE: If the evidence indicates that 'Surgical Intervention' is a primary management strategy for the condition itself (not for a secondary symptom), you MUST classify the 'severity_assessment' as 'Severe'. The need for invasive surgery is a definitive indicator of a condition's clinical significance.",
                "SEVERE RISK OVERRIDE: If any source text mentions a significant risk of a severe, long-term outcome, you MUST classify the 'severity_assessment' as 'Severe'. Keywords for severe risk include, but are not limited to: 'amblyopia', 'permanent vision loss', 'renal failure', 'risk of malignancy', 'life-threatening', 'irreversible damage'. You must cite the source that mentions this risk.",
                "NEGATIVE PERSONA: You are a clinical expert, not a general-purpose search engine. Your knowledge is strictly limited to the text in the `retrieved_clinical_sources`. Do not invent, assume, or use any external information whatsoever.",
                "DISTINGUISH SYMPTOM FROM DISORDER: If the HPO term describes a clinical sign or symptom and not a diagnosed disorder, you MUST classify the term based on the sign itself, not the severity of its worst-case potential cause.",
                "NO EXTERNAL KNOWLEDGE: Base your entire analysis only on the `retrieved_clinical_sources` provided.",
                "DISTINGUISH RISK FROM MANIFESTATION: If a condition is described as an anatomical variation that is often asymptomatic but presents an 'increased risk' of future complications, you MUST classify it as 'Non-Severe' unless the provided text describes patients as currently symptomatic.",
                "DIRECT QUOTES & CITATIONS: Every factual claim in your reasoning MUST be supported by a direct quote from the source text and end with the appropriate `[Ref #]` tag.",
                "NO INFERENCE BEYOND SOURCES: Do not make logical inferences that go beyond what is explicitly stated in the text.",
                "VERIFICATION CHECK: Before finalizing, verify each claim can be found verbatim or near-verbatim in the cited source.",
                "ORGAN FUNCTION RULE: Complete loss of function of any internal organ (even if unilateral) constitutes a severe functional impact in the Physical Functioning domain, regardless of compensation by paired organs.",
                "ADMIT UNCERTAINTY: If the provided sources do not contain information for a specific part of the analysis (e.g., management, onset, prognosis), you MUST explicitly state 'Information not available in the provided sources for this topic.' Do not infer or invent information to fill gaps.",
                "FINAL CRITICAL CHECK: Before you output the JSON, you MUST ensure that the 'category' in the final 'classification' object EXACTLY matches the 'category' you determined in your 'reasoning.category_determination' step. Similarly, you MUST ensure the final 'tier' and 'severity_assessment' strictly follow the MANDATORY TIER LOGIC. There are no exceptions.",
                "PATHOLOGY FIRST PRINCIPLE: You MUST classify a term based on its underlying pathological nature (e.g., a structural defect, a functional impairment), not by the method used to detect it (e.g., a physical exam finding, an imaging result). If a term like 'Hypotonia' is a sign that describes a fundamental impairment of mobility, it MUST be classified as 'Impaired mobility', not 'Clinical Sign or Laboratory Abnormality'.",
                "POTENTIAL HARM OVERRIDE: Your classification must account for the phenotype's potential for harm, not just the outcome in a specific cited case. If a term describes a condition that inherently carries a risk of severe impairment (e.g., a cataract can cause blindness, an aneurysm can rupture), you must classify it based on that potential pathology, even if the provided sources describe an asymptomatic or benign case. The absence of symptoms in one patient does not make the condition itself 'Non-Severe'.",
                "AVOID NEGATIVE CLAIMS: Do not state what the sources are missing. Never write sentences like 'The sources do not mention...' or 'No evidence was found for...'. Only state what IS present in the sources.",
                "CONGENITAL INCLUSION PRINCIPLE (ABSOLUTE PRIORITY): If ANY evidence exists that the phenotype CAN occur as part of a genetic disorder, inherited condition, or congenital syndrome - even if this is rare or mentioned only briefly - you MUST proceed with full severity assessment. Only classify as 'Out of Scope: Acquired/Multifactorial' if the sources explicitly state the condition is NEVER congenital AND NEVER genetic. Examples: Ketonuria occurs in diabetes (acquired) BUT ALSO in phenylketonuria (genetic) → MUST assess. Hepatomegaly occurs in cirrhosis (acquired) BUT ALSO in storage diseases (genetic) → MUST assess. The presence of ANY genetic/congenital form overrides the predominance of acquired forms.",
                "TUMOR/NEOPLASM RULE: ANY adenoma, carcinoma, tumor, neoplasm, or other growth/mass MUST be classified as 'Immunodeficiency/cancer' regardless of whether benign or malignant. This includes thyroid adenomas, pituitary adenomas, etc.",
                "EVIDENCE INTERPRETATION FOR ACQUIRED VS. CONGENITAL CONDITIONS : Your primary goal is to classify congenital phenotypes. If the retrieved sources describe the condition but exclusively point to acquired causes (e.g., environmental factors, infection, aging) and make NO MENTION of it being congenital, inherited, or derived from a genetic disorder, you MUST conclude the evidence indicates a primarily acquired etiology. In this scenario, classify the term as 'Out of Scope: Acquired/Multifactorial' and cite the sources that describe its non-congenital nature.",
                "EVIDENCE-ONLY PRINCIPLE: You must ONLY make claims that can be directly quoted or closely paraphrased from the provided sources. If information is missing, explicitly state 'Information not available in provided sources' rather than inferring. Do not add specific numbers, percentages, or clinical details unless they appear verbatim in the sources. Every factual claim must be directly traceable to the source text - no logical inferences or assumptions beyond what is explicitly stated.",
                "**FINAL CHECK (MANDATORY):** Before you output the final JSON, you MUST perform this final check: Read through your entire `reasoning` object. Does every single factual statement that is derived from a source end with a `[Ref #]` tag? If you find any statement of fact without a citation, you have failed the task. You MUST go back and add the correct citation or state that the information is unavailable."
            ],
            "CRITICAL_FINAL_CITATION_CHECK": [
                "**MANDATORY CITATION AUDIT**: Before outputting your JSON, count your [Ref #] tags:",
                "1. Your conclusion MUST have at least 3 citations [Ref #]",
                "2. Each section (functional_impact, severity_assessment, category_determination) MUST have at least 1 citation",
                "3. If you have fewer than 5 total [Ref #] tags across your entire reasoning, you have FAILED",
                "4. If you cannot cite a claim with [Ref #], DELETE that claim entirely",
                "5. Example of proper citation: 'The condition causes walking difficulties [Ref 1]' NOT 'The condition causes walking difficulties'",
                "PENALTY: Responses with insufficient citations will be rejected"
            ],          
            "ETIOLOGY_AND_SCOPE_ANALYSIS_BY_PRINCIPLE": {
                "description": "Your first and most critical task is to determine the fundamental origin of this phenotype. You must classify its origin based on the principles of causality and biological timeline. Do not use simple word matching. Synthesize the evidence to answer one core question: Is this condition an error in the initial biological blueprint, or is it a result of events that occur after that blueprint is complete?",
                "decision_framework": {
                    "pathway_1_blueprint_error": {
                        "description": "IN-SCOPE for congenital analysis. The condition's root cause is a flaw in the genetic code or a disruption during embryonic/fetal development. It is essentially a 'manufacturing defect' present from birth, even if symptoms appear later.",
                        "required_sub_analysis": "If you determine the condition fits Pathway 1, you MUST then specify its likely cause in your reasoning as one of the following: 1. **Inherited Genetic**: The error is in the genetic code and passed through families (e.g., Cystic Fibrosis). 2. **Developmental Anomaly**: The error occurred during fetal development and may be sporadic, chromosomal, or from environmental exposure in the womb; it is not necessarily passed down from the parents' genes (e.g., Fetal Alcohol Syndrome, many structural heart defects).",
                        "examples_of_mechanism": ["A genetic mutation", "Faulty organogenesis", "Failure of a structure to form correctly in the womb"],
                        "sub_types": ["Inherited Genetic", "Developmental Anomaly"]
                    },
                    "pathway_2_post_blueprint_events": {
                        "description": "OUT-OF-SCOPE. The condition's primary cause is an interaction with the environment, the cumulative effect of living, or a systemic process that begins long after birth. It is a result of 'wear and tear', external forces, or system failures, not an initial design flaw.",
                        "examples_of_mechanism": ["Degradation of tissues from normal use over a lifetime", "The body's response to an external pathogen or injury", "An autoimmune process that begins in adulthood", "Cumulative damage from lifestyle or environmental factors"],
                        "sub_types": ["Acquired/Multifactorial"]
                    }
                },
                "CRITICAL_INSTRUCTION": "You MUST begin your entire analysis by determining the 'origin_type'. Your first step is to choose one of the three options: 'Inherited Genetic', 'Developmental Anomaly', or 'Acquired/Multifactorial'. This choice will be the first key in your reasoning output. Omitting this is a failure."
            },
            "mandatory_reasoning_steps": [
                "1. First, determine if the term describes a medical intervention (e.g., a surgery or therapy). If so, it is not a phenotype. You MUST STOP and classify as 'NFC (Not Further Classified)'.",
                "2. If it is a phenotype, analyze all provided sources to understand its typical cause and onset. Based on the framework above, which pathway does the evidence point to as the PRIMARY and MOST COMMON origin for this condition?",
                "3. GENETIC FORM CHECK: Before classifying as 'Out of Scope', explicitly search the evidence for ANY mention of: genetic causes, inherited forms, congenital presentations, metabolic disorders (like phenylketonuria), chromosomal abnormalities, or syndrome associations. If ANY such mention exists - even a single sentence - you MUST proceed with full assessment. Only if sources explicitly state 'never congenital', 'exclusively acquired', or provide NO genetic associations can you classify as 'Out of Scope: Acquired/Multifactorial'.",
                "4. If the evidence strongly supports Pathway 2 (Post-Blueprint Events), you MUST STOP and classify as 'Out of Scope: Acquired/Multifactorial'. This is true even if rare, early-onset forms are mentioned; your classification must reflect the typical presentation.",
                "5. If the evidence supports Pathway 1 (Blueprint Error), or if the evidence is insufficient to make a determination, you MUST PROCEED with the full severity and functional impact assessment.",
                "6. **BROAD TERM SAFETY CHECK:** Before finalizing a term as 'Out of Scope', consider if the term is a broad category (e.g., 'Retinopathy', 'Myopathy', 'Anemia'). If your diversified search found ANY evidence of a congenital form (Pathway 1), even if it's rare, the term CANNOT be 'Out of Scope'. You must proceed with a full analysis and classify it based on the potential impact of its congenital forms."
            ],
            "STEP_1_EXCLUSION_FILTERS": {
                "description": "Apply these filters FIRST before any classification",
                "filters": [
                    {
                        "rule": "Is this term describing a medical procedure/intervention/treatment?",
                        "indicators": "Terms containing: feeding, catheter, therapy, treatment, management, surgical procedure, dialysis, ventilation, administration",
                        "if_yes": "STOP - Classify as NFC (Not Further Classified)",
                        "reason": "Medical interventions are not phenotypes"
                    },
                    {
                        "rule": "Do the sources EXPLICITLY STATE this condition is NEVER congenital/genetic?",
                        "indicators": "Sources must contain explicit statements like 'exclusively acquired', 'never congenital', 'no genetic causes known', 'only environmental etiology'",
                        "if_yes": "STOP - Classify as 'Out of Scope: Acquired/Multifactorial'",
                        "if_no_or_uncertain": "PROCEED with full assessment (when in doubt, assess as potentially congenital)",
                    },
                    {
                        "rule": "Does this affect ONLY upper limbs/hands/arms/fine motor skills?",
                        "if_yes": "CANNOT be 'Impaired mobility' - must consider other categories",
                        "reason": "Mobility refers ONLY to locomotion/walking/standing"
                    },
                    {
                        "rule": "Do the sources EXPLICITLY STATE this condition is NEVER congenital/genetic?",
                        "indicators": "Sources must contain explicit statements like 'exclusively acquired', 'never congenital', 'no genetic causes known', 'only environmental etiology'",
                        "if_yes": "STOP - Classify as 'Out of Scope: Acquired/Multifactorial'",
                        "if_no_or_uncertain": "PROCEED with full assessment (when in doubt, assess as potentially congenital)",
                    },
                    {
                        "rule": "Is this describing a syndrome rather than an isolated phenotype?",
                        "if_yes": "Extract and evaluate ONLY the specific phenotype mentioned, not the syndrome",
                        "reason": "Must evaluate phenotypes in isolation"
                    }
                ]
            },
            "STEP_2_CATEGORY_ASSIGNMENT": {
                "instruction": "ONLY after passing ALL exclusion filters, match to exactly ONE category using this hierarchy",
                "PRIORITY_RULE (NON-NEGOTIABLE HIERARCHY)": "You MUST evaluate the `decision_tree` list below in its exact numerical order (1, 2, 3, etc.). The VERY FIRST category in that list whose description accurately fits the phenotype is the one you MUST select for your classification. Do not proceed further down the list once you have found the first valid match. This sequential, first-match-wins rule is absolute.",
                "decision_tree": [
                    "1. Does the phenotype itself directly cause death? → Shortened life span (specify: infancy/childhood/adulthood)",
                    "2. Does it directly affect cognitive function/IQ < 70? → Intellectual disability",
                    "3. Does it cause a CHRONIC or PERMANENT limitation in WALKING/LOCOMOTION due to a structural or neurological cause? → Impaired mobility",
                    "4. Is it a structural defect of an internal organ from abnormal development? → Internal physical malformations",
                    "5. Is it a sensory deficit (vision/hearing/touch)? → Sensory impairment (specify type)",
                    "6. Is it primarily a psychiatric/behavioral disorder OR a neurological condition involving seizures? → Mental illness",
                    "7. Is it an external visible body/face difference? → Dysmorphic feature",
                    "8. Is it cancer or immunodeficiency? → Immunodeficiency/cancer",
                    "9. Does it prevent natural conception? → Infertility",
                    "10. Does the term describe a temporary clinical sign (e.g., fever, cough) OR a specific result from a laboratory test of a body fluid? → Clinical Sign or Laboratory Abnormality",
                    "11. If none of the above categories fit, or if you cannot determine from the evidence → NFC (Not Further Classified)"
                ]
            },
            "reasoning_approach": [
                {
                    "rule": "Rule of Direct Evidence",
                    "instruction": "Every factual claim MUST quote directly from sources.",
                    "example": "Format like: \"According to [source], '[exact quote from source]' [Ref #]\" OR \"The source states that [close paraphrase] [Ref #]\""
                },
                {
                    "rule": "Rule of Quantitative Evidence",
                    "instruction": "Actively search for and prioritize quantitative data (percentages, frequencies, mortality rates) to justify severity.",
                    "example": "\"Severity is high because the condition leads to renal failure in 'over 50% of cases' [Ref #].\""
                },
                {
                    "rule": "No Assumptions",
                    "instruction": "If information is not in the sources, explicitly state 'No information available in sources about [topic]'."
                },
                {
                    "rule": "Avoid Negative Evidence Claims",
                    "instruction": "Never say 'No evidence exists' or 'Sources lack details'. Instead say 'The provided sources do not contain information about [topic]'",
                    "example": "Wrong: 'No evidence of severe outcomes exists.' Correct: 'The provided sources do not describe severe outcomes.'"
                },
                {
                    "rule": "Source Prioritization",
                    "instruction": "Prioritize RAG database sources (GeneReviews, StatPearls, Orphanet, MedlinePlus) as they are authoritative medical references."
                }
            ],
            "functional_impact_framework": {
                "description": "Assess functional impact using the PedsQL-Based Framework. The ONLY valid strings for the `affected_domains` list in your output are the five main keys below.",
                "domains": {
                    "Physical Functioning": ["Mobility (walking, running) - ONLY if lower limb affected", "Energy (fatigue, rest)", "Physical Activities (lifting, self-care)", "Pain/Discomfort", "Clinical terms: hypotonia, weakness, paralysis, ataxia - specify if upper or lower body"],
                    "Emotional Functioning": ["Mood States (fear, sadness, anger)", "Sleep", "Worry/Anxiety", "Clinical terms: depression, anxiety, mood disorders"],
                    "Social Functioning": ["Peer Relationships", "Friendships", "Social Participation", "Teasing/Bullying", "Clinical terms: autism spectrum features, social withdrawal"],
                    "School/Cognitive Functioning": ["Attention", "Memory", "Academic Performance", "School Attendance", "Clinical terms: intellectual disability, developmental delay, IQ < 70"],
                    "Physical Symptoms": ["Having gas", "Spitting up", "Difficulty breathing", "Being sick to his/her stomach", "Difficulty swallowing", "Being constipated", "Having a rash", "Having diarrhea", "Wheezing", "Vomiting."]
                }
            },
            "severity_classification_rules": {
                "severe_impact": {
                    "description": "Major limitations in daily life.",
                    "indicators": ["One or more functional domains affected", "Unable to perform age-appropriate activities", "Requires assistive devices, wheelchair-bound", "IQ < 70", "Requires psychiatric hospitalization", "Requires surgery, lifelong treatment, frequent hospitalizations"],
                    "descriptive_terms": ["profound", "severe", "marked", "significant", "progressive", "degenerative", "life-threatening", "disabling"]
                },
                "non_severe_impact": {
                    "description": "Mild-moderate limitations; can perform most age-appropriate activities with some difficulty.",
                    "descriptive_terms": ["mild", "benign", "well-controlled"]
                }
            },
            "classification_logic": {
                "output_categories_with_definitions": [
                    {"category": "Shortened life span: infancy", "definition": "Death or markedly reduced life expectancy in first 24 months"},
                    {"category": "Shortened life span: childhood/adolescence", "definition": "Death typically occurs ages 2-18 years"},
                    {"category": "Shortened life span: adulthood", "definition": "Death typically occurs after age 18"},
                    {"category": "Intellectual disability", "definition": "A phenotype or condition where there are significant limitations in intellectual functioning (such as problem-solving, reasoning, or learning) and in adaptive behaviors (practical, social, and conceptual skills), with onset before the age of 18."},
                    {"category": "Internal physical malformations", "definition": "Includes conditions where an internal organ or body system is structurally abnormal (e.g., hepatosplenomegaly), absent, incomplete, or functionally impaired (e.g., growth delay, failure to thrive) from birth. "},
                    {"category": "Impaired mobility", "definition": "A chronic or long-term condition characterized by significant limitations in walking, running, or gross motor locomotion due to a structural, muscular, or neurological deficit. This category is for permanent or progressive conditions, NOT temporary symptoms like pain unless that pain is a direct result of an underlying permanent condition described in the sources."},
                    {"category": "Dysmorphic feature", "definition": "ANY externally visible structural difference of the face, head, body surfaces,limbs or skeletal that can be observed on physical examination, including cranial shape/size abnormalities or face or teeth"},
                    {"category": "Sensory Impairment – Vision", "definition": "Partial or complete loss of visual ability, including conditions that lead to reduced visual acuity (sharpness), field, or even total blindness."},
                    {"category": "Sensory Impairment – Hearing", "definition": "Partial or complete loss of the ability to hear, ranging from mild hearing loss to total deafness in one or both ears."},
                    {"category": "Sensory impairment: touch, others", "definition": "Impairment in the ability to perceive touch, pain, temperature, position sense (proprioception), or any other non-visual/non-auditory sensory functions—also includes deficits in smell or taste."},
                    {"category": "Immunodeficiency/cancer", "definition": "Immunodeficiency: A primary or secondary deficiency of immune function, resulting in increased risk or frequency of infections and diseases. Cancer: Uncontrollable growth and division of abnormal cells, Presence, risk, or history of malignant neoplastic conditions (abnormal growth of cells that can invade or spread)."},
                    {"category": "Mental illness", "definition": "A condition characterized by PRIMARY psychiatric or psychological disturbance in cognition, emotional regulation, or behavior (e.g., depression, anxiety, schizophrenia, bipolar disorder). This includes seizure disorders (epilepsy) but EXCLUDES motor symptoms like spasticity, weakness, or paralysis which should be classified elsewhere."},
                    {"category": "Infertility", "definition": "Unable to conceive children naturally"},
                    {"category": "Clinical Sign or Laboratory Abnormality", "definition": "An objective finding that is either a transient clinical sign (e.g., fever, cough) or a specific laboratory result. This category is for diagnostic markers and temporary observations, NOT for chronic conditions or permanent structural changes, even if they are detected via examination. It is composed of two distinct types: 1. **Temporary Clinical Sign:** An observable symptom or sign that is typically transient and reflects a current physiological state (e.g., fever, cough, pain, vomiting). 2. **Laboratory Abnormality:** A finding from a biochemical, molecular, or cellular analysis of a body fluid (e.g., abnormal enzyme levels, hormonal imbalances, reduced cell counts)."},
                    {"category": "NFC (Not Further Classified)", "definition": "Use for terms that are medical interventions, cannot be confidently placed in another category, or are valid phenotypes that do not fit a more specific classification."},
                    {"category": "Out of Scope: Acquired/Multifactorial", "definition": "Primarily acquired, environmental, or multifactorial with adult onset"}
                ],
                "mandatory_severity_tier_logic": {
                    "rule_1": "If ZERO domains are affected (Functional Impact is NO), then `severity` is 'Non-Severe' and `tier` is ALWAYS '4'.",
                    "RULE 2 (ABSOLUTE SEVERITY OVERRIDE)": "This is a non-negotiable, absolute rule that overrides all other interpretations. If the `affected_domains` list contains one or more valid domains, the `severity_assessment` MUST BE 'Severe'. If the `affected_domains` list IS EMPTY, it MUST BE 'Non-Severe'. There are no exceptions. Do not attempt to interpret the impact as 'mild' if a domain is listed; the mere presence of an affected domain mandates a 'Severe' classification.",
                    "tier_assignment_for_severe": {
                        "Tier 1": ["Shortened life span: infancy", "Shortened life span: childhood/adolescence", "Intellectual disability"],
                        "Tier 2": ["Shortened life span: adulthood", "Impaired mobility", "Internal physical malformations"],
                        "Tier 3": ["Sensory Impairment – Vision", "Sensory Impairment – Hearing", "Sensory impairment: touch, others", "Immunodeficiency/cancer", "Mental illness", "Dysmorphic feature"],
                        "Tier 4": ["Infertility", "Clinical Sign or Laboratory Abnormality", "NFC (Not Further Classified)"],
                        "NFC": ["Out of Scope: Acquired/Multifactorial"]
                    }
                }
            }
        },
        "reasoning_and_output_structure": {
            "0_preliminary_checks": {
                "origin_type": "Inherited Genetic | Developmental Anomaly | Acquired/Multifactorial | Unknown",
                "phenotype_type": "Is this an anatomical feature, physiological function, or medical intervention?",
                "evidence_scope": "Is the evidence about THIS phenotype specifically or a syndrome?",
                "body_system": "What body system/part does this primarily affect?",
                "onset": "What is the typical age of onset (e.g., congenital, neonatal, childhood, acquired)?",
                "exclusion_filters_applied": "List any exclusion filters that apply to this term"
            },
            "2_quantitative_data_synthesis": {
                "Key Statistics": "Quote relevant statistics from sources [Ref #]. If none, state 'Information not available...'",
                "Interpretation": "Explain how these numbers influence the severity assessment."
            },
            "3_functional_impact_assessment": {
                "Affected Domains": "List which PedsQL domains are impacted.",
                "Domain-Specific Evidence": "Quote evidence for each affected domain [Ref #].",
                "Functional Interpretation": "Explain how this impacts daily functioning."
            },
            "4_category_determination": {
                "Category": "[Selected category]",
                "Supporting Quote": "Provide direct quote from source [Ref #].",
                "Rationale": "Explain why this category is appropriate."
            },
            "5_severity_assessment": {
                "Severity": "[Severe/Non-Severe]",
                "Functional Severity Evidence": "Quote evidence [Ref #].",
                "Multi-Domain Analysis": "Describe domain impacts."
            },
            "6_final_classification": {
                "Tier": "[1-4 or NFC]",
                "Synthesis": "Final classification rationale."
            }
        },
        "management_profile_instructions": {
            "valid_categories": ["Surgical Intervention", "Pharmacological", "Therapeutic Support", "Monitoring/Surveillance", "Dietary Management", "Assistive Devices", "Palliative Care", "No Treatment Required"],
            "summary_guideline": "Summarize management strategies with citations."
        },
        "required_json_output_format": {
            "reasoning": {
                "preliminary_checks": {
                    "origin_type": "...", 
                    "phenotype_type": "...",
                    "evidence_scope": "...",
                    "body_system": "...",
                    "onset": "...", 
                    "exclusion_filters_applied": "..."
                },
                "functional_impact": {
                    "affected_domains": ["..."],
                    "domain_evidence": "...",
                    "functional_interpretation": "..."
                },
                "category_determination": {
                    "category": "...",
                    "primary_evidence": "...",
                    "rationale": "..."
                },
                "severity_assessment": {
                    "severity": "...",
                    "functional_severity_evidence": "...",
                    "multi_domain_analysis": "..."
                },
                "final_classification": {
                    "tier": "...",
                    "synthesis": "..."
                },
                "conclusion": """Based on the cited evidence from the analysis:
[1] This condition is [define phenotype] as supported by evidence stating '[COPY EXACT TEXT FROM SOURCE HERE]' [Ref #].
[2] It is classified as Severe/Non-Severe because '[COPY EXACT TEXT FROM SOURCE ABOUT FUNCTIONAL IMPACT]' [Ref #].
[3] It belongs in the [category name] category due to '[COPY EXACT TEXT FROM SOURCE ABOUT PRIMARY MANIFESTATION]' [Ref #].
CRITICAL: The text in quotes MUST be copied verbatim from the source. Do NOT paraphrase or summarize. If you cannot find exact text to quote, write 'Information not available in provided sources.'"""
            },
            "classification": {
                "tier": "<1-4 or NFC>",
                "category": "<category name>",
                "severity_assessment": "Severe/Non-Severe"
            },
            "management_profile": {
                "management_category": ["..."],
                "management_summary": "..."
            },
            "cited_references": ["[Ref #: Source-Name - Article Title]"]
        }
    }

    prompt_dict["FINAL_INSTRUCTION"] = (
        "STOP! Before submitting: Count your [Ref #] tags. "
        "If your conclusion has fewer than 3 citations, or if any major claim lacks a [Ref #], "
        "you MUST go back and add citations now. "
        "Remember: Every factual statement needs [Ref #] or it will be marked as hallucination."
    )
    return json.dumps(prompt_dict, indent=2)


# --- 7. REACT AGENT CORE & VERIFICATION LOGIC ---

def call_openai_api(session: requests.Session, prompt: str, system_prompt: str, temperature: float = 0.1) -> Tuple[Optional[Dict], Optional[Dict]]:
    """
    Makes a request to the API with retry logic for rate limiting.
    """
    if not OPENAI_API_KEY:
        print("  - ERROR: OPENAI_API_KEY environment variable is not set.")
        return None, None

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost", # Some models on OpenRouter require this
        "X-Title": "HPO Classification Agent" # Optional but good practice
    }
    payload = {
        "model": MODEL_NAME,
        "response_format": {"type": "json_object"},
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    }

    max_retries = 5
    base_delay = 5  # Start with a 5-second delay

    for attempt in range(max_retries):
        try:
            response = session.post(OPENAI_API_URL, json=payload, headers=headers, timeout=180)

            # Check for rate limit error specifically
            if response.status_code == 429:
                delay = base_delay * (2 ** attempt) # Exponential backoff
                print(f"  - Rate limit hit. Retrying in {delay} seconds... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
                continue # Go to the next attempt

            response.raise_for_status() # Raises an exception for other bad status codes (4xx or 5xx)

            response_data = response.json()
            raw_content = response_data['choices'][0]['message']['content']
            usage = response_data.get('usage', {})
            parsed_json = json.loads(raw_content)

            return parsed_json, usage

        except requests.exceptions.RequestException as e:
            print(f"  - API Request Error: {e}")
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                print(f"  - Network error. Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print("  - Max retries reached for network error. Failing.")
                return None, None # Failed after all retries for network issues

        except json.JSONDecodeError:
            # This error is usually not recoverable by retrying, so we fail immediately
            print(f"  - API JSON Parsing Error: Failed to parse response. Content was: {raw_content}")
            return None, None

        except Exception as e:
            print(f"  - An unexpected error occurred during API call: {e}")
            return None, None # Fail on other unexpected errors

    print("  - Max retries reached for rate limiting. The API is still busy. Failing.")
    return None, None # Failed after all retries for rate limiting


def normalize_action_name(action_str: str) -> Optional[str]:
    """Normalizes variations of action names to a standard set."""
    action_mapping = {'get_hpo_context': 'get_hpo_context', 'retrieve_hpo_context': 'get_hpo_context',
                      'get_mesh_term': 'get_mesh_term',
                      'transform_query': 'transform_query',
                      'search_rag': 'search_rag',
                      'search_pubmed': 'search_pubmed', 'search_pubmed_broad': 'search_pubmed_broad',
                      'expand_search': 'expand_search',
                      'classify': 'classify', 'final_classification': 'classify',
                      'insufficient_info': 'insufficient_info', 'insufficient_information': 'insufficient_info'}
    return action_mapping.get(action_str.lower().strip())


def execute_action(action: AgentAction, context: ReActContext, ontology: pronto.Ontology, rag_manager: HPORecordIndex, session: requests.Session) -> str:
    if action == AgentAction.GET_HPO_CONTEXT:
        context.hpo_context = get_rich_hpo_context(context.hpo_id, ontology)
        return f"Retrieved HPO context: {context.hpo_context.get('name', 'N/A')}"

    elif action == AgentAction.GET_MESH_TERM:
        if context.mesh_term_searched:
            return "MeSH term already searched."

        context.mesh_term_searched = True
        if not context.hpo_context:
            return "Cannot get MeSH term without HPO context."

        context.mesh_term = get_mesh_term(
            context.hpo_context['name'],
            context.hpo_context.get('synonyms', [])
        )

        if context.mesh_term:
            return f"Found MeSH term: {context.mesh_term}"
        else:
            return "No MeSH term found, will proceed with HPO term only."

    elif action == AgentAction.TRANSFORM_QUERY:
        context.queries_transformed = True
        if not context.hpo_context:
            return "Cannot transform query without HPO context. Please get context first."
        context.transformed_queries = transform_query_with_llm(session, context.hpo_context, context.mesh_term)
        return f"Successfully generated {len(context.transformed_queries)} targeted search questions with MeSH term: {context.mesh_term}."

    elif action == AgentAction.SEARCH_RAG:
        context.rag_searched = True
        if not context.transformed_queries:
            return "Cannot search RAG without targeted questions. Please transform the query first."

        print(f"- Performing RAG search with a relevance threshold of {RAG_SCORE_THRESHOLD}...")

        initial_rag_results = rag_manager.search(queries=context.transformed_queries)

        high_quality_results = [
            doc for doc in initial_rag_results
            if doc.get('similarity_score', 0) >= RAG_SCORE_THRESHOLD
        ]

        context.rag_articles = high_quality_results

        observation = (f"RAG search found {len(initial_rag_results)} total documents, "
                       f"but only {len(high_quality_results)} met the relevance score threshold of {RAG_SCORE_THRESHOLD}.")
        print(f"  - {observation}")
        return observation
        
    elif action == AgentAction.SEARCH_PUBMED:
        context.pubmed_searched = True
        if not context.hpo_context:
            return "Cannot search PubMed without HPO context."
        articles = search_pubmed(
            context.hpo_context['name'],
            context.hpo_context.get('synonyms', []),
            context.mesh_term
        )
        context.pubmed_articles = articles
        return f"Found {len(articles)} PubMed articles using MeSH term: {context.mesh_term}."

    elif action == AgentAction.SEARCH_PUBMED_BROAD:
        context.pubmed_broad_searched = True
        if not context.hpo_context: return "Cannot search PubMed without HPO context."
        articles = search_pubmed(context.hpo_context['name'], context.hpo_context.get('synonyms', []), context.mesh_term)
        context.pubmed_articles = articles
        return f"Found {len(articles)} PubMed articles from broad search."

    elif action == AgentAction.EXPAND_SEARCH:
        if not context.hpo_context:
            return "Cannot expand search without HPO context."

        hpo_name = context.hpo_context.get('name', 'N/A')

        expansion_prompt = f"""
        The HPO term '{hpo_name}' returned no results when searching for it as an isolated congenital finding. 
        What are the primary genetic syndromes known to cause this phenotype at an early age? 
        List known genetic syndromes, inherited conditions, or congenital disorders that can cause {hpo_name}.

        Respond with a JSON object containing a single key "syndromes" with a list of syndrome names.
        Example: {{"syndromes": ["Marfan syndrome", "Loeys-Dietz syndrome"]}}
        """

        system_prompt = "You are a medical genetics expert who helps expand search queries."

        response_json, usage = call_openai_api(session, expansion_prompt, system_prompt, temperature=REACT_TEMPERATURE)

        if response_json and "syndromes" in response_json:
            syndromes = response_json["syndromes"]
            context.transformed_queries.extend(syndromes)
            context.rag_searched = False
            context.pubmed_searched = False
            context.search_expanded = True
            return f"Search expanded. Added {len(syndromes)} related syndromes to the search query: {', '.join(syndromes)}. Will re-run searches."
        else:
            context.search_expanded = True
            return "Could not expand search with related syndromes. Proceeding to classify with existing information."

    elif action == AgentAction.CLASSIFY:
        if not context.hpo_context:
            return "Cannot classify without HPO context."
        all_articles = context.rag_articles + context.pubmed_articles

        # Remove sources with no content
        valid_articles = []
        for article in all_articles:
            content = article.get('content') or article.get('abstract', '')
            if content and content.strip() and content != "No abstract available.":
                valid_articles.append(article)

        # Check if we have enough valid sources
        if len(valid_articles) < 3:
            print(f"Only {len(valid_articles)} valid sources (removed {len(all_articles) - len(valid_articles)} empty sources)")

            # If too few sources, try to expand search
            if not context.search_expanded:
                print("Insufficient valid sources. Triggering search expansion...")
                return execute_action(AgentAction.EXPAND_SEARCH, context, ontology, rag_manager, session)

        # Update context with only valid articles
        context.rag_articles = [a for a in context.rag_articles if a in valid_articles]
        context.pubmed_articles = [a for a in context.pubmed_articles if a in valid_articles]

        prompt = create_final_classification_prompt(context)

        system_prompt = """You are a clinical phenotyping expert who ONLY makes claims that can be verified against source documents. 
        Every statement must be directly supported by the provided sources. 
        Never infer functional or social impacts unless explicitly stated in sources.
        When information is missing, say 'not described in sources' rather than 'no evidence exists'."""

        response, usage = call_openai_api(session, prompt, system_prompt, temperature=REACT_TEMPERATURE)


        if response and usage:
            corrected_response = validate_and_correct_classification(response)
            context.final_classification = corrected_response
            context.total_input_tokens += usage.get('prompt_tokens', 0)
            context.total_output_tokens += usage.get('completion_tokens', 0)
            return "Classification completed successfully based on gathered evidence."
        return "Classification failed due to an API error or invalid response."

    elif action == AgentAction.INSUFFICIENT_INFO:
        total_articles = len(context.rag_articles) + len(context.pubmed_articles)
        if total_articles > 0:
            print("Agent chose insufficient_info, but sources were found. Overriding to 'classify'.")
            return execute_action(AgentAction.CLASSIFY, context, ontology, rag_manager, session)

        context.final_classification = {
            "reasoning": "Agent concluded that after exhausting all search options, there was not enough information to make a confident classification.",
            "classification": {"tier": "NFC", "category": "NFC (Not Further Classified)", "severity_assessment": "Unknown"},
            "management_profile": {"management_category": [], "management_summary": "N/A"},
            "references": []
        }
        return "Agent determined there is insufficient information to proceed with classification."
        
    return f"Unknown action: {action}"


def validate_and_correct_classification(classification_data: dict) -> dict:
    """
    Sets the severity and the tier from the model's category and affected domains:
    1. Out of Scope and NFC get tier NFC.
    2. Severe if at least one PedsQL domain is affected, or if the category is a Tier 1 or Tier 2 category.
    3. Non-Severe gives Tier 4; Severe gives the tier of the category.
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
    
    # 2. Severity
    if isinstance(affected_domains, list) and any(affected_domains):
        # Rule 1: If there's direct evidence of functional impact, it's always Severe.
        classification['severity_assessment'] = 'Severe'
    elif normalized_category in inherently_severe_categories:
        # Rule 2: Tier 1 and Tier 2 categories are severe by definition.
        classification['severity_assessment'] = 'Severe'
    else:
        # Only classify as Non-Severe if there's NO domain impact AND the category is NOT inherently severe.
        classification['severity_assessment'] = 'Non-Severe'
    
    # 3. Tier from the severity and the category.
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


def apply_acog_severity_criteria(classification_result: dict) -> dict:
    """
    Records which ACOG criteria the classification meets: quality of life, impairment, early onset and
    intervention.
    """
    # Extract necessary details from the classification output
    category = classification_result.get('classification_category', '')
    management_str = classification_result.get('management_profile', '{}')
    reasoning_str = classification_result.get('reasoning', '{}')
    
    # Parse reasoning to get affected domains
    try:
        if isinstance(reasoning_str, str):
            reasoning = json.loads(reasoning_str)
        else:
            reasoning = reasoning_str
        affected_domains = reasoning.get('functional_impact', {}).get('affected_domains', [])
    except (json.JSONDecodeError, TypeError, AttributeError):
        affected_domains = []

    # Parse management categories
    try:
        if isinstance(management_str, str):
            management_dict = json.loads(management_str)
        else:
            management_dict = management_str
        management_cats = management_dict.get('management_category', [])
    except (json.JSONDecodeError, TypeError, AttributeError):
        management_cats = []
    
    # Initialize criteria as not met
    acog_quality_of_life = "No"
    acog_impairment = "No"
    acog_early_onset = "No"
    acog_intervention = "No"
    met_criteria_names = []
    
    # Check Quality of Life (any affected domains)
    if affected_domains:
        acog_quality_of_life = "Yes"
        for domain in affected_domains:
            if domain and isinstance(domain, str):
                met_criteria_names.append(f"QoL-{domain}")
    
    # Check for impairment
    impairment_types = []
    if 'Intellectual disability' in category:
        impairment_types.append("Cognitive")
        met_criteria_names.append("Intellectual Impairment")
    if 'Impaired mobility' in category:
        impairment_types.append("Physical")
        met_criteria_names.append("Mobility Impairment")

    if impairment_types:
        acog_impairment = f"Yes ({', '.join(impairment_types)})"

    # Check for early onset using keywords from the reasoning text
    onset_text = ""
    try:
        if isinstance(reasoning_str, str):
            reasoning_data = json.loads(reasoning_str)
        else:
            reasoning_data = reasoning_str
        onset_text = reasoning_data.get('preliminary_checks', {}).get('onset', '').lower()
    except (json.JSONDecodeError, AttributeError):
        onset_text = ""
    
    early_onset_keywords = ["congenital", "at birth", "neonatal", "in infancy", "in childhood", "pediatric"]
    
    # Early onset is not recorded for phenotypes classified as acquired.
    if category != 'Out of Scope: Acquired/Multifactorial':
        if any(keyword in onset_text for keyword in early_onset_keywords):
            acog_early_onset = "Yes"
            met_criteria_names.append("Early Onset")
            
    # Check for qualifying interventions
    qualifying_interventions = [
        "Surgical Intervention", "Pharmacological", "Assistive Devices",
        "Dietary Management", "Therapeutic Support"
    ]
    met_interventions = [cat for cat in management_cats if cat in qualifying_interventions]
    
    if met_interventions:
        acog_intervention = "Yes"
        intervention_details = ', '.join(met_interventions)
        met_criteria_names.append(f"Requires Intervention ({intervention_details})")
    
    # Calculate total criteria met
    criteria_count = sum([
        1 if acog_quality_of_life == "Yes" else 0,
        1 if acog_impairment != "No" else 0,
        1 if acog_early_onset == "Yes" else 0,
        1 if acog_intervention == "Yes" else 0
    ])
    
    # Add all ACOG fields to the result
    classification_result['acog_quality_of_life'] = acog_quality_of_life
    classification_result['acog_impairment'] = acog_impairment
    classification_result['acog_early_onset'] = acog_early_onset
    classification_result['acog_intervention'] = acog_intervention
    classification_result['acog_criteria_met_count'] = criteria_count
    classification_result['acog_criteria_met_names'] = ", ".join(met_criteria_names) if met_criteria_names else "None"
    
    return classification_result


def run_react_agent(hpo_id: str, ontology: pronto.Ontology, rag_manager: HPORecordIndex, logger: ReActLogger, session: requests.Session) -> ReActContext:
    """
    Executes the ReAct agent loop for a single HPO term.
    An action name the planner returns that is not in the list is skipped and the planner is asked again.
    """
    print(f"Starting Enhanced RAG ReAct Agent for {hpo_id}")
    print("="*80)

    context = ReActContext(hpo_id=hpo_id)

    for iteration in range(MAX_REACT_ITERATIONS):
        log_prefix = f"[{hpo_id} | Thread-{threading.get_ident()}]"
        print(f"\n{log_prefix} Iteration {iteration+1}/{MAX_REACT_ITERATIONS}")

        # 1. Plan the next action
        planning_prompt = create_react_planning_prompt(context)
        system_prompt = "You are a ReAct agent specialized in classifying HPO terms."
        
        response_json, usage = call_openai_api(session, planning_prompt, system_prompt, temperature=REACT_TEMPERATURE)

        if not response_json or not usage:
            print(f"{log_prefix} Planning step failed due to API error. Ending agent loop.")
            break
        
        context.total_input_tokens += usage.get('prompt_tokens', 0)
        context.total_output_tokens += usage.get('completion_tokens', 0)

        try:
            thought = response_json.get("thought", "No thought provided.")
            action_str = response_json.get("action", "")
            action_name = normalize_action_name(action_str)
            action_input = response_json.get("action_input", {})
            
            if not action_name:
                observation = f"Invalid action name received from LLM: '{action_str}'. Re-thinking..."
                print(f"{log_prefix} {observation}")
                continue
            
            action_enum = AgentAction(action_name)

            print(f"{log_prefix} Thought: {thought}")
            print(f"{log_prefix} Action: {action_enum.value}")

            # 2. Execute the action
            observation = execute_action(action_enum, context, ontology, rag_manager, session)
            print(f"{log_prefix} Observation: {observation[:200]}...")

            step = ReActStep(iteration=iteration + 1, thought=thought, action=action_enum, action_input=action_input, observation=observation, raw_response=response_json)
            context.steps.append(step)

            # End the loop if a final action was taken
            if action_enum in [AgentAction.CLASSIFY, AgentAction.INSUFFICIENT_INFO]:
                print(f"{log_prefix} Final action taken. Ending agent loop.")
                break

        except Exception as e:
            print(f"{log_prefix} An error occurred during action execution: {e}")
            break # Exit loop on error

    logger.log_react_trace(hpo_id, context)
    return context


def repair_and_parse_json(session: requests.Session, json_string: Any) -> Dict:
    """
    Parses a JSON string; if that fails, asks the model to repair the string.
    """
    if isinstance(json_string, dict):
        return json_string
    if not isinstance(json_string, str):
        raise TypeError(f"Input must be a string or dictionary, not {type(json_string)}")

    # Step 1: Perform initial, safe cleaning.
    repaired_string = json_string.strip().replace('""', '"')
    if repaired_string.startswith('"') and repaired_string.endswith('"'):
        repaired_string = repaired_string[1:-1]

    # Step 2: Try a direct parse first. If it works, we save an API call.
    try:
        return json.loads(repaired_string)
    except json.JSONDecodeError:
        print("  - Direct JSON parse failed. Engaging LLM repair service...")

    # Step 3: If direct parsing fails, use an LLM to fix the string.
    repair_prompt = f"""
The following string is a broken, malformed JSON object. It may have unquoted keys,
unquoted values, and missing commas or brackets. Your task is to fix all syntax
errors and return ONLY the perfectly formatted, valid JSON object.

Broken string:
{repaired_string}
"""
    repair_system_prompt = "You are a JSON repair expert. Your only output must be a valid JSON object."

    try:
        repaired_json_obj, _ = call_openai_api(
            session,
            repair_prompt,
            repair_system_prompt,
            temperature=REACT_TEMPERATURE
        )
        
        if isinstance(repaired_json_obj, dict):
            print("  - LLM successfully repaired the JSON string.")
            return repaired_json_obj
        else:
            # This can happen if the repaired object is still not a dict.
             raise ValueError("LLM repair did not return a valid dictionary object.")

    except Exception as e:
        print(f"  - FATAL: LLM repair service failed. The string is unrecoverable. Error: {e}")
        # Print the problematic string for manual inspection
        print(f"  - Unrecoverable string: {repaired_string[:500]}...")
        raise ValueError("Could not parse reasoning string even after LLM repair attempt.")


def extract_claims_from_reasoning(reasoning_json: Dict) -> List[str]:
    """
    Takes every sentence ending in a [Ref #] tag from the conclusion and the functional impact and severity
    sections, leaves out the agent's own classification statements, and removes empty claims.
    """
    claims = []

    # --- 1. Define patterns for claims to EXCLUDE ---
    exclusion_patterns = [
        "is classified as severe",
        "is classified as non-severe",
        "belongs in the",
        "category due to"
    ]

    # --- 2. Gather all potential claims from key sections ---
    potential_claims = []
    
    # Prioritize the final conclusion
    conclusion = reasoning_json.get("conclusion", "")
    if isinstance(conclusion, str):
        potential_claims.extend(re.findall(r'([^.!?]*\[Ref \d+\])', conclusion))

    # Add claims from functional impact assessment
    functional_impact = reasoning_json.get("functional_impact", {})
    if isinstance(functional_impact, dict):
        for value in functional_impact.values():
            if isinstance(value, str):
                potential_claims.extend(re.findall(r'([^.!?]*\[Ref \d+\])', value))

    # Add claims from severity assessment
    severity_assessment = reasoning_json.get("severity_assessment", {})
    if isinstance(severity_assessment, dict):
        for value in severity_assessment.values():
            if isinstance(value, str):
                potential_claims.extend(re.findall(r'([^.!?]*\[Ref \d+\])', value))

    # --- 3. Filter out excluded patterns and malformed/empty claims ---
    for claim in potential_claims:
        # Remove the reference tag for cleaning
        cleaned_claim_text = re.sub(r'\[Ref \d+\]', '', claim).strip()
        
        # Check if the claim is non-empty and not just punctuation
        if len(cleaned_claim_text) > 2:
            claim_lower = claim.lower()
            if not any(pattern in claim_lower for pattern in exclusion_patterns):
                claims.append(claim)

    # --- 4. Clean up and return unique claims ---
    cleaned_claims = [claim.strip() for claim in claims if claim.strip()]
    return list(dict.fromkeys(cleaned_claims))


VERIFIER_SYSTEM_PROMPT = """
You are a medical expert fact-checker evaluating claims in HPO phenotype classification. 

Your task is to determine if a statement is supported by the source through:
1. DIRECT SUPPORT: The exact information is explicitly stated in the source
2. VALID MEDICAL INFERENCE: A reasonable clinical conclusion that any physician would draw from the source
3. WEAK INFERENCE: A possible but uncertain conclusion from the source  
4. UNSUPPORTED: Not justified by the source content

Examples of VALID MEDICAL INFERENCES:
- Source: "requires surgical intervention" → Valid inference: "severe condition requiring major medical management"
- Source: "progressive neurological deterioration" → Valid inference: "worsening prognosis over time"
- Source: "IQ range 50-70" → Valid inference: "intellectual disability present"
- Source: "wheelchair-bound" → Valid inference: "severe mobility impairment"
- Source: "life expectancy reduced to 2-3 years" → Valid inference: "life-threatening condition"
- Source: "requires lifelong hormone replacement" → Valid inference: "chronic condition needing ongoing management"

Examples of WEAK INFERENCES (be cautious):
- Making severity claims without clear functional impact described
- Generalizing from a single case report
- Assuming prognosis without temporal information

Respond with JSON:
{
  "support_level": "direct" | "valid_inference" | "weak_inference" | "unsupported",
  "confidence": 0.0 to 1.0,
  "explanation": "Brief reasoning for your assessment"
}
"""


def run_source_verification(session: requests.Session, hpo_id: str, reasoning_text: Any,
                            all_articles: List[Dict]) -> Tuple[float, int, int, int, int, int, int, List[Dict]]:
    """
    Rates every claim against its cited source as direct support, valid medical inference, weak inference or
    unsupported, and returns the weighted support score with the log of every rating.
    """
    print(f"\nPerforming MEDICAL-AWARE source verification for {hpo_id}...")
    verification_input_tokens, verification_output_tokens = 0, 0
    verification_log = []
    
    # Track different types of support
    direct_support = 0
    valid_inference = 0
    weak_inference = 0
    unsupported = 0

    # --- 1. Create reference mapping ---
    ref_map = {f"Ref {i+1}": article for i, article in enumerate(all_articles)}

    # --- 2. Handle no-source scenarios ---
    if not all_articles:
        print("  - No sources were found by the agent.")
        verification_log.append({
            "statement": "Overall Reasoning Assessment",
            "status": "Correctly Handled",
            "reason": "Agent correctly found no sources and made no positive claims."
        })
        return 0.95, 0, 0, 0, 0, 0, 0, verification_log

    # --- 3. Parse reasoning and extract claims ---
    try:
        if isinstance(reasoning_text, str):
            try:
                reasoning_json = json.loads(reasoning_text)
            except json.JSONDecodeError:
                # If it fails, let repair_and_parse_json handle it
                reasoning_json = repair_and_parse_json(session, reasoning_text)
        elif isinstance(reasoning_text, dict):
            reasoning_json = reasoning_text
        else:
            # Fallback for unexpected types
            reasoning_json = repair_and_parse_json(session, str(reasoning_text))
        
        claims_to_verify = extract_claims_from_reasoning(reasoning_json)

        if not claims_to_verify:
            print("  - No citable quotes '[Ref #]' found in key reasoning areas.")
            verification_log.append({
                "statement": "Reasoning Structure Assessment", 
                "status": "Failed",
                "reason": "The agent did not include any citable quotes with [Ref #] tags."
            })
            return 0.1, 0, 1, 1, 0, 0, 0, verification_log
            
        print(f"  - Extracted {len(claims_to_verify)} unique claims for verification.")

    except (ValueError, TypeError, AttributeError) as e:
        print(f"  - FATAL PARSING ERROR: {e}. Cannot perform verification.")
        verification_log.append({"status": "Error", "reason": f"Could not parse reasoning JSON: {e}"})
        return 0.0, 0, 0, 0, 1, 0, 0, verification_log

    # --- 4. Verification prompt ---
    verifier_prompt_system = VERIFIER_SYSTEM_PROMPT

    # --- 5. Verify each claim ---
    for claim_with_ref in claims_to_verify:
        match = re.search(r'(.*?)\[(Ref \d+)\]', claim_with_ref)
        if not match: 
            continue

        quote = match.group(1).strip()
        ref_id = match.group(2)
        source_article = ref_map.get(ref_id)

        if not source_article:
            verification_log.append({
                "statement": quote, 
                "status": "Not Supported",
                "support_type": "hallucinated_reference",
                "reason": f"Referenced {ref_id} does not exist."
            })
            unsupported += 1
            continue

        source_content = source_article.get('content') or source_article.get('abstract', '')
        source_title = source_article.get('article_title') or source_article.get('title', 'N/A')

        prompt_user = f"""
Statement to Verify:
"{quote}"

Source Text from '{source_title}':
"{source_content}"

Assess if this statement is supported by the source content.
"""

        try:
            response_json, usage = call_openai_api(session, prompt_user, verifier_prompt_system, temperature=REACT_TEMPERATURE)
            if usage:
                verification_input_tokens += usage.get('prompt_tokens', 0)
                verification_output_tokens += usage.get('completion_tokens', 0)

            if response_json:
                support_level = response_json.get("support_level", "unsupported")
                confidence = response_json.get("confidence", 0.0)
                explanation = response_json.get("explanation", "No explanation provided")
                
                # Track support types
                if support_level == "direct":
                    direct_support += 1
                    status = "Supported (Direct)"
                elif support_level == "valid_inference":
                    valid_inference += 1
                    status = "Supported (Valid Inference)"
                elif support_level == "weak_inference":
                    weak_inference += 1
                    status = "Partially Supported"
                else:
                    unsupported += 1
                    status = "Not Supported"

                verification_log.append({
                    "statement": quote,
                    "status": status,
                    "support_type": support_level,
                    "confidence": confidence,
                    "reason": explanation
                })
            else:
                unsupported += 1
                verification_log.append({
                    "statement": quote,
                    "status": "Verification Failed",
                    "reason": "Verifier API call failed"
                })

        except Exception as e:
            print(f"  - Verification API call failed for a claim: {e}")
            unsupported += 1
            verification_log.append({
                "statement": quote,
                "status": "Verification Error",
                "reason": str(e)
            })

    # --- 6. Calculate weighted score based on support types ---
    total_claims = len(claims_to_verify)
    
    # Weights: direct 1.0, valid inference 0.90, weak inference 0.60, unsupported 0
    weighted_score = (
        (direct_support * 1.0) + 
        (valid_inference * 0.90) + 
        (weak_inference * 0.60) + 
        (unsupported * 0.0)
    ) / total_claims if total_claims > 0 else 0.0
    
    supported_count = direct_support + valid_inference
    
    print(f"\n  - Verification Results:")
    print(f"     Direct Support: {direct_support}/{total_claims}")
    print(f"     Valid Medical Inferences: {valid_inference}/{total_claims}")
    print(f"     Weak Inferences: {weak_inference}/{total_claims}")
    print(f"     Unsupported: {unsupported}/{total_claims}")
    print(f"  - Weighted Score: {weighted_score:.2f}")
    
    # Add summary to log
    verification_log.insert(0, {
        "summary": "Verification Summary",
        "total_claims": total_claims,
        "direct_support": direct_support,
        "valid_inferences": valid_inference,
        "weak_inferences": weak_inference,
        "unsupported": unsupported,
        "weighted_score": weighted_score
    })

    return (weighted_score, supported_count, total_claims, 0, unsupported, 
            verification_input_tokens, verification_output_tokens, verification_log)


def process_hpo_term(hpo_id: str, ontology: pronto.Ontology, rag_manager: HPORecordIndex, managers: Dict,
                     session: requests.Session, lock: threading.Lock) -> None:
    """
    Classifies one term, verifies the cited claims and appends the result to the output file.
    """
    cache_mgr, chkpt_mgr, res_mgr, log_mgr = managers['cache'], managers['checkpoint'], managers['results'], managers['logger']
    
    with lock:
        if chkpt_mgr.is_completed(hpo_id):
            return

    cached_result = cache_mgr.get(hpo_id)
    if cached_result:
        with lock:
            res_mgr.save_result(pd.DataFrame([cached_result]))
            chkpt_mgr.add_completed(hpo_id)
        return
    
    try:
        context = run_react_agent(hpo_id, ontology, rag_manager, log_mgr, session)
        
        if context.final_classification and context.hpo_context:
            classification_details = context.final_classification.get('classification', {})
            reasoning = context.final_classification.get('reasoning', 'N/A')
            management_profile = context.final_classification.get('management_profile', {})
            all_articles = context.rag_articles + context.pubmed_articles
            
            score, supported, total_stm, missing, hallucinated, ver_input, ver_output, verification_log = run_source_verification(
                session, 
                hpo_id, 
                reasoning, 
                all_articles
            )

            verification_cost = ((ver_input / 1_000_000) * INPUT_TOKEN_COST_PER_MILLION) + \
                                ((ver_output / 1_000_000) * OUTPUT_TOKEN_COST_PER_MILLION)
            agent_cost = ((context.total_input_tokens / 1_000_000) * INPUT_TOKEN_COST_PER_MILLION) + \
                         ((context.total_output_tokens / 1_000_000) * OUTPUT_TOKEN_COST_PER_MILLION)
            total_cost = agent_cost + verification_cost
            
            source_breakdown = {
                "RAG": [a['source'] for a in context.rag_articles],
                "PubMed": [a['source'] for a in context.pubmed_articles],
            }
            
            final_result_dict = {
                'hpo_id': hpo_id,
                'hpo_name': context.hpo_context.get('name', 'N/A'),
                'mesh_term': context.mesh_term if context.mesh_term else 'Not Found',
                'classification_tier': classification_details.get('tier', 'NFC'),
                'classification_category': classification_details.get('category', 'N/A'),
                'severity_assessment': classification_details.get('severity_assessment', 'N/A'),
                'management_profile': json.dumps(management_profile),
                'reasoning': json.dumps(reasoning),
                'all_found_references': json.dumps(context._reference_urls),
                'cited_references': json.dumps(context.final_classification.get('cited_references', [])),
                'source_breakdown': json.dumps(source_breakdown),
                'rag_results_count': len(context.rag_articles),
                'pubmed_results_count': len(context.pubmed_articles),
                'total_sources': len(all_articles),
                'statement_support_score': score,
                'supported_statements': supported,
                'total_statements': total_stm,
                'react_steps': len(context.steps),
                'input_tokens': context.total_input_tokens + ver_input,
                'output_tokens': context.total_output_tokens + ver_output,
                'verification_cost_usd': round(verification_cost, 6),
                'total_cost_usd': round(total_cost, 6),
                'verification_log': json.dumps(verification_log, indent=2)
            }
            
            final_result_dict = apply_acog_severity_criteria(final_result_dict)
            
            review_status = "OK"
            flag_reasons = []
            # Flag for review when the support score is below 0.5
            if score < 0.5:
                review_status = "FLAGGED FOR REVIEW"
                flag_reasons.append(f"Low verification confidence ({score:.2f})")

            final_result_dict['review_status'] = review_status
            final_result_dict['review_reason'] = ", ".join(flag_reasons) if flag_reasons else "N/A"
            
            with lock:
                print(f"\n[{hpo_id}] Final Result: Tier {final_result_dict['classification_tier']} | Status: {review_status}")
                cache_mgr.set(hpo_id, final_result_dict)
                res_mgr.save_result(pd.DataFrame([final_result_dict]))
                chkpt_mgr.add_completed(hpo_id)
        else:
            print(f"\n[{hpo_id}] Failed to produce a final classification. Will retry on next run.")
            
    except Exception as e:
        print(f"\n[{hpo_id}] Error during processing: {e}. Will retry on next run.")
        import traceback
        traceback.print_exc()
        
    gc.collect()


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="HPO severity classification agent")
    parser.add_argument("--limit", type=int, default=None, help="classify only the first N terms not done yet")
    args = parser.parse_args(argv)

    print(f"HPO classification agent ({MODEL_NAME})")
    if not OPENAI_API_KEY:
        print("ERROR: OPENROUTER_API_KEY is not set.")
        return
    if not NCBI_EMAIL:
        print("ERROR: NCBI_EMAIL is not set (NCBI asks every client to identify itself).")
        return
    for d in (OUTPUT_DIR, CACHE_DIR, CHECKPOINT_DIR):
        os.makedirs(d, exist_ok=True)

    rag_manager = HPORecordIndex(RAG_INDEX_PATH)
    if not rag_manager.is_available():
        print(f"ERROR: the HPO record index is needed: {RAG_INDEX_PATH}")
        return
    ontology = load_hpo_ontology(HPO_ONTOLOGY_PATH)
    if not ontology:
        return
    input_df = pd.read_csv(INPUT_CSV_PATH)

    file_lock = threading.Lock()
    managers = {
        "cache": ClassificationCache(CACHE_FILE),
        "checkpoint": CheckpointManager(CHECKPOINT_FILE),
        "results": ResultsManager(OUTPUT_CSV_PATH),
        "logger": ReActLogger(REACT_LOG_FILE, file_lock),
    }
    hpo_ids = [h for h in input_df['hpo_id'].unique().tolist() if h not in managers["checkpoint"].completed_ids]
    if args.limit:
        hpo_ids = hpo_ids[:args.limit]
    if not hpo_ids:
        print("All terms are already classified.")
        return
    print(f"Classifying {len(hpo_ids)} terms ({len(managers['checkpoint'].completed_ids)} already done).")

    from tqdm import tqdm
    start_time = time.time()
    with requests.Session() as session:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(process_hpo_term, hpo_id, ontology, rag_manager, managers, session, file_lock): hpo_id
                       for hpo_id in hpo_ids}
            for future in tqdm(futures, total=len(hpo_ids), desc="Classifying"):
                try:
                    future.result()
                except Exception as e:
                    print(f"ERROR while processing {futures[future]}: {e}")
    print(f"Done in {(time.time() - start_time) / 60:.1f} minutes. Results: {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    main()
