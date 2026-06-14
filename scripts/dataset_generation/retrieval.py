import os
import ast
import asyncio
import json
import random
import torch
import csv
import httpx
from urllib.parse import urlparse
from typing import Dict, List
from dataclasses import dataclass, field
from tqdm import tqdm
from openai import AsyncOpenAI
import weaviate
import weaviate.classes as wvc
from schemas.data_generation import HypothesisResponse
from jinja2 import Environment, FileSystemLoader, PrefixLoader
from api_client.gemini_parser import GeminiResponseParser
from api_client.gemini_client import GeminiClient
from dotenv import load_dotenv
from transformers import AutoTokenizer
from utils.config import CONFIG

# Resolve env files from repository root (scripts/dataset_generation/../../)
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Load DB/API credentials first, then regular runtime environment variables.
load_dotenv(os.path.join(_PROJECT_ROOT, '.env_db'))
load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RAW_BASE_PATH = CONFIG['paths']['base_path']
ABS_BASE_PATH = os.path.expanduser(RAW_BASE_PATH)

PERSONAS_DIR  = os.path.join(ABS_BASE_PATH, CONFIG['paths']['personas_dir'])
PERSONAS_FILE = os.path.join(PERSONAS_DIR,  CONFIG['files']['personas'])

PROMPTS_DIR   = os.path.join(ABS_BASE_PATH, CONFIG['paths']['prompts_dir'])
CONCEPTS_FILE = os.path.join(PROMPTS_DIR,   CONFIG['files']['concepts'])

HYPOTHESES_DIR  = os.path.join(ABS_BASE_PATH, CONFIG['paths']['hypotheses_dir'])
CONTEXT_FILE = os.path.join(HYPOTHESES_DIR, CONFIG['files']['retrieved_context'])

ARTICLE_DIR                = os.path.join(ABS_BASE_PATH, CONFIG['paths']['articles_dir'])
EXCLUSION_REGISTRY_FILE = os.path.join(ARTICLE_DIR, 'exclusion_registry_synthetic.json')
CLASSIFICATIONS_FILE    = os.path.join(ARTICLE_DIR, 'classifications.csv')

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
GEN_MODEL       = CONFIG['models']['generation_model']
EMBEDDING_MODEL = CONFIG['models']['embedding_model']
RERANKER_MODEL  = CONFIG['models']['reranker_model']
VLLM_API_URL    = CONFIG['models']['vllm_api_url']
VLLM_API_KEY    = CONFIG['models']['vllm_api_key']

_EMBEDDER_PORT   = CONFIG['models']['embedder_port']   # 8001
_RERANKER_PORT   = CONFIG['models']['reranker_port']   # 8002


def _normalize_vllm_base_url(raw_url: str) -> str:
    """
    Normalise a vLLM API URL to bare scheme://host so that per-service ports
    can be appended cleanly.  Strips any existing port and path.

    Examples:
        "http://localhost"          → "http://localhost"
        "http://localhost:8000/v1"  → "http://localhost"
        "http://1.2.3.4:8001"       → "http://1.2.3.4"
    """
    parsed = urlparse(raw_url.strip())
    return f"{parsed.scheme}://{parsed.hostname}"


_VLLM_API_BASE   = _normalize_vllm_base_url(VLLM_API_URL)
EMBEDDER_API_URL = f"{_VLLM_API_BASE}:{_EMBEDDER_PORT}/v1"
RERANKER_API_URL = f"{_VLLM_API_BASE}:{_RERANKER_PORT}"

RERANKER_USE_INSTRUCTION: bool = CONFIG.get('retrieval', {}).get('reranker_use_instruction', False)

# ---------------------------------------------------------------------------
# Weaviate settings
# ---------------------------------------------------------------------------
# Add these keys to your CONFIG if you want to override the defaults:
#
#   weaviate:
#     host: "localhost"
#     port: 8080
#     collection: "ResearchPapers"
#     doc_types: ["CHUNK"]   # filter by type; use [] to search all types
#
# The defaults below match the docker-compose in this project.
WEAVIATE_COLLECTION = CONFIG.get('weaviate', {}).get('collection', 'ResearchPapers')
# Restrict ANN search to these `type` values.  Set to [] to search all types.
WEAVIATE_DOC_TYPES: List[str] = CONFIG.get('weaviate', {}).get('doc_types', ['CHUNK'])

# Connection mode:
#   - remote: connect to externally hosted Weaviate (VPS_IP/.env_db)
#   - local:  connect to local Weaviate started by retrieval.sh via Apptainer
WEAVIATE_MODE_RAW: str = str(CONFIG.get('weaviate', {}).get('mode', 'remote')).strip().lower()
if WEAVIATE_MODE_RAW not in {'remote', 'local'}:
    raise ValueError(
        f"Invalid weaviate.mode='{WEAVIATE_MODE_RAW}'. Expected one of: remote, local"
    )
WEAVIATE_MODE: str = WEAVIATE_MODE_RAW

if WEAVIATE_MODE == 'remote':
    WEAVIATE_HOST_RAW = os.getenv('VPS_IP', CONFIG.get('weaviate', {}).get('host', 'localhost'))
    WEAVIATE_PORT = CONFIG.get('weaviate', {}).get('port', 8080)
    WEAVIATE_GRPC_PORT = CONFIG.get('weaviate', {}).get('grpc_port', 50051)
    WEAVIATE_API_KEY = os.getenv('WEAVIATE_API_KEY', CONFIG.get('weaviate', {}).get('api_key'))
    WEAVIATE_SECURE = CONFIG.get('weaviate', {}).get('secure', False)
else:
    WEAVIATE_HOST_RAW = CONFIG.get('weaviate', {}).get('local_host', 'localhost')
    WEAVIATE_PORT = CONFIG.get('weaviate', {}).get('local_port', 8080)
    WEAVIATE_GRPC_PORT = CONFIG.get('weaviate', {}).get('local_grpc_port', 50051)
    WEAVIATE_API_KEY = CONFIG.get('weaviate', {}).get('local_api_key')
    WEAVIATE_SECURE = CONFIG.get('weaviate', {}).get('local_secure', False)


def _normalize_weaviate_host(raw_host: str, default_secure: bool) -> tuple[str, bool]:
    """
    Accept host values with or without scheme and return
    (host_without_scheme, http_secure_flag).
    """
    host = (raw_host or '').strip()
    if host.startswith('http://'):
        return host[len('http://'):], False
    if host.startswith('https://'):
        return host[len('https://'):], True
    return host, bool(default_secure)


WEAVIATE_HOST, WEAVIATE_HTTP_SECURE = _normalize_weaviate_host(
    WEAVIATE_HOST_RAW,
    WEAVIATE_SECURE,
)

# ---------------------------------------------------------------------------
# Inference settings
# ---------------------------------------------------------------------------
TEMPERATURE    = CONFIG['inference']['temperature']
TOP_K          = CONFIG['inference']['top_k']
TOP_P          = CONFIG['inference']['top_p']
MIN_P          = CONFIG['inference']['min_p']
MAX_TOKENS     = CONFIG['inference']['max_tokens']
SEMAPHORE_SIZE = CONFIG['inference']['semaphore_size']

# ---------------------------------------------------------------------------
# Retrieval settings
# ---------------------------------------------------------------------------
CONCEPTS_TOP_K = CONFIG['retrieval']['top_k']        # chunks retrieved per concept
RERANK_TOP_K   = CONFIG['retrieval']['rerank_top_k'] # final chunks kept after reranking

# Retrieval backend mode:
#   - "hybrid": BM25 + dense vector fusion (default)
#   - "dense":  dense vector only
RETRIEVAL_MODE_RAW: str = str(CONFIG.get('retrieval', {}).get('mode', 'hybrid')).strip().lower()
if RETRIEVAL_MODE_RAW not in {'dense', 'hybrid'}:
    raise ValueError(
        f"Invalid retrieval.mode='{RETRIEVAL_MODE_RAW}'. Expected one of: dense, hybrid"
    )
RETRIEVAL_MODE: str = RETRIEVAL_MODE_RAW

# Query source mode:
#   - "subqueries": use concept subqueries (fallback to original query if empty)
#   - "original":   ignore concepts and retrieve only from original prompt query
QUERY_SOURCE_RAW: str = str(CONFIG.get('retrieval', {}).get('query_source', 'subqueries')).strip().lower()
if QUERY_SOURCE_RAW not in {'subqueries', 'original'}:
    raise ValueError(
        f"Invalid retrieval.query_source='{QUERY_SOURCE_RAW}'. Expected one of: subqueries, original"
    )
QUERY_SOURCE: str = QUERY_SOURCE_RAW

# Hybrid search balance: 0.0 = pure BM25, 1.0 = pure dense vector, 0.5 = balanced.
# Values around 0.5–0.7 work well for scientific corpora where both exact keyword
# matching (rare technical terms, algorithm names) and semantic similarity matter.
HYBRID_ALPHA: float = CONFIG.get('retrieval', {}).get('hybrid_alpha', 0.5)

env = Environment(loader=PrefixLoader({
    "data_generation":       FileSystemLoader("prompt_templates/data_generation/"),
    "hypotheses_generation": FileSystemLoader("prompt_templates/data_generation/hypotheses_generation/"),
}))

system_template  = env.get_template("hypotheses_generation/system.j2")
user_template    = env.get_template("hypotheses_generation/user.j2")
persona_template = env.get_template("data_generation/persona_template.j2")


# ---------------------------------------------------------------------------
# vLLM Embedder  (query-side only — corpus vectors live in Weaviate)
# ---------------------------------------------------------------------------

class VLLMEmbedder:
    """
    Calls the vLLM /v1/embeddings endpoint (OpenAI-compatible) to embed
    concept strings at retrieval time.
    """

    def __init__(self, model: str, api_url: str, api_key: str, device: str = 'cuda:0'):
        self.model   = model
        self.device  = device
        self._client = AsyncOpenAI(base_url=api_url, api_key=api_key)

    async def embed_text(self, text: str) -> torch.Tensor:
        """Returns a float32 1-D torch.Tensor on self.device."""
        response = await self._client.embeddings.create(model=self.model, input=text)
        return torch.tensor(response.data[0].embedding, dtype=torch.float32, device=self.device)


# ---------------------------------------------------------------------------
# Weaviate retriever  (replaces DenseInMemoryIndex + EmbeddingRetriever)
# ---------------------------------------------------------------------------

@dataclass
class _Chunk:
    """
    Thin wrapper exposing the interface the rest of the pipeline expects:
      chunk.content             — raw text of the chunk
      chunk.chunk_id            — unique str key used for deduplication
      chunk.metadata['doc_id']  — parent paper / document identifier
    """
    chunk_id: str
    content:  str
    metadata: dict = field(default_factory=dict)


@dataclass
class RankedChunk:
    chunk:          _Chunk
    score:          float
    source_concept: str


class WeaviateRetriever:
    """
    Wraps a Weaviate collection and exposes a `retrieve()` interface that is
    a drop-in replacement for EmbeddingRetriever, so nothing downstream
    (dedup, reranking, context building) needs to change.

    Uses Weaviate's **hybrid search** (BM25 + dense vector + Reciprocal Rank
    Fusion) instead of pure ANN search.  This significantly improves recall
    for scientific queries that contain rare technical terms, algorithm names,
    acronyms, and author-specific notation — all cases where exact keyword
    matching (BM25) complements semantic similarity (dense retrieval).

    Parameters
    ----------
    collection_name : str
        Weaviate collection to query (default: "ResearchPapers").
    doc_types : list[str]
        When non-empty, restricts results to objects whose `type` property
        is in this list (e.g. ["CHUNK"]).  Pass [] to search across all types.
    host / port : str / int
        Must match your docker-compose weaviate service.
    hybrid_alpha : float
        Blending factor for BM25 vs. dense retrieval passed to Weaviate's
        hybrid() API.  0.0 = pure BM25, 1.0 = pure dense vector, 0.5 =
        balanced (default).  Values around 0.5–0.7 work well for scientific
        corpora where both exact keyword matching and semantic similarity
        matter.
    """

    def __init__(
        self,
        collection_name: str,
        doc_types: List[str],
        host: str = 'localhost',
        port: int = 8080,
        grpc_port: int = 50051,
        api_key: str | None = None,
        http_secure: bool = False,
        grpc_secure: bool = False,
        hybrid_alpha: float = 0.5,
        retrieval_mode: str = 'hybrid',
    ):
        self._doc_types    = doc_types
        self._hybrid_alpha = hybrid_alpha
        self._retrieval_mode = retrieval_mode

        if self._retrieval_mode not in {'dense', 'hybrid'}:
            raise ValueError(
                f"Invalid retrieval_mode='{self._retrieval_mode}'. Expected one of: dense, hybrid"
            )

        connect_kwargs = {
            'http_host': host,
            'http_port': port,
            'http_secure': http_secure,
            'grpc_host': host,
            'grpc_port': grpc_port,
            'grpc_secure': grpc_secure,
        }
        if api_key:
            connect_kwargs['auth_credentials'] = weaviate.auth.AuthApiKey(api_key)

        self._wv_client    = weaviate.connect_to_custom(**connect_kwargs)
        self._collection   = self._wv_client.collections.get(collection_name)
        print(
            f"WeaviateRetriever ready — collection='{collection_name}' "
            f"at {host}:{port} (grpc={grpc_port}, secure={http_secure})  "
            f"auth={'api-key' if api_key else 'none'}  type filter={doc_types or 'none'}  "
            f"retrieval_mode={self._retrieval_mode}  "
            f"hybrid_alpha={hybrid_alpha} (0=BM25, 1=dense)"
        )

    @property
    def retrieval_mode(self) -> str:
        return self._retrieval_mode

    def close(self):
        """Release the Weaviate connection."""
        self._wv_client.close()

    def retrieve(
        self,
        query_text: str,
        query_vector: torch.Tensor,
        top_k: int,
        excluded_paper_ids: List[str] | None = None,
    ) -> List[RankedChunk]:
        """
                Run retrieval in the configured mode and return up to top_k RankedChunks.

                Mode behavior:
                    - hybrid: BM25 + dense vector via Weaviate hybrid() fusion
                    - dense:  dense-only retrieval via near_vector()

        Hybrid retrieval improves over pure dense search in two important ways
        for this pipeline:
          • BM25 catches rare scientific terms (algorithm names, acronyms,
            author-specific notation) that dense embeddings may underweight.
          • RRF diversifies results across documents, reducing the tendency of
            dense ANN to surface multiple chunks from the same paper.

        Parameters
        ----------
        query_text : str
            Raw text of the concept/subquery — fed to the BM25 side of the
            hybrid search.
        query_vector : torch.Tensor
            Shape (D,) or (1, D) — the embedded concept from VLLMEmbedder,
            fed to the dense vector side of the hybrid search.
        top_k : int
            Maximum number of results to return.
        """
        vec: List[float] = query_vector.squeeze().cpu().tolist()

        # Optional property filter on `type`
        filters = None
        if self._doc_types:
            filters = wvc.query.Filter.by_property("type").contains_any(self._doc_types)

        # Exclude papers whose paperId is in the exclusion registry for this pair.
        if excluded_paper_ids:
            excl = ~wvc.query.Filter.by_property("paperId").contains_any(list(excluded_paper_ids))
            filters = excl if filters is None else (filters & excl)

        if self._retrieval_mode == 'dense':
            response = self._collection.query.near_vector(
                near_vector=vec,
                limit=top_k,
                filters=filters,
                return_metadata=wvc.query.MetadataQuery(score=True, distance=True),
            )
        else:
            # Hybrid search: BM25 + dense vector fused with Reciprocal Rank Fusion.
            # `alpha` controls the blend: 0.0 = pure BM25, 1.0 = pure dense.
            # The `vector` kwarg supplies our externally-computed embedding so
            # Weaviate does not need a vectorizer module configured.
            response = self._collection.query.hybrid(
                query=query_text,
                vector=vec,
                alpha=self._hybrid_alpha,
                limit=top_k,
                filters=filters,
                return_metadata=wvc.query.MetadataQuery(score=True, distance=True),
            )

        results: List[RankedChunk] = []
        for obj in response.objects:
            props     = obj.properties
            paper_id  = props.get('paperId', '')
            chunk_idx = props.get('chunkIndex', -1)

            # Stable unique key: "<paperId>_<chunkIndex>"
            chunk_id = f"{paper_id}_{chunk_idx}"

            chunk = _Chunk(
                chunk_id=chunk_id,
                content=props.get('content', ''),
                metadata={
                    'doc_id':      paper_id,
                    'chunk_index': chunk_idx,
                    'title':       props.get('title', ''),
                    'type':        props.get('type', ''),
                },
            )
            # Prefer score when available; for dense mode, fall back to
            # converting distance to a higher-is-better proxy.
            score = obj.metadata.score
            if score is None and obj.metadata.distance is not None:
                score = max(0.0, 1.0 - float(obj.metadata.distance))
            if score is None:
                score = 0.0
            results.append(RankedChunk(chunk=chunk, score=score, source_concept=''))

        return results


# ---------------------------------------------------------------------------
# Reranker
# ---------------------------------------------------------------------------

RERANKER_INSTRUCTION = (
    "Given a scientific query, retrieve relevant passages that can be used "
    "to generate scientific hypothesis for subjects mentioned in the query."
)


class VLLMReranker:
    """
    Calls vLLM's /score endpoint to score candidate chunks against the
    original query, then returns the top-N highest-scoring chunks.
    """

    def __init__(self, model: str, api_url: str, api_key: str):
        self.model = model
        base = api_url.rstrip('/')
        self.score_url = base if base.endswith('/score') else f"{base}/score"
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    async def rerank(
        self,
        query: str,
        candidates: List[RankedChunk],
        top_n: int,
        client: httpx.AsyncClient,
    ) -> List[RankedChunk]:
        documents = [rc.chunk.content for rc in candidates]
        print("Total character length of retrieved documents:", sum(len(d) for d in documents))

        for i, doc in enumerate(documents):
            tokens      = tokenizer(query, doc, truncation=False, return_tensors=None)
            token_count = len(tokens["input_ids"])
            print(f"Token count for doc {i}: {token_count}.")
            if token_count > 32768:
                print("  ↑ Token count exceeded 32 768!")

        payload = {
            "model": self.model,
            "queries": f"<Instruct>{RERANKER_INSTRUCTION}\n<Query>{query}" if RERANKER_USE_INSTRUCTION else query,
            "documents": documents,
        }
        response = await client.post(self.score_url, headers=self._headers, json=payload)
        response.raise_for_status()

        body = response.json()

        # /score response shape: {"data": [{"index": i, "score": s}, ...]}
        # Keep a fallback for /v1/rerank-like responses for compatibility.
        if isinstance(body, dict) and isinstance(body.get("data"), list):
            for item in body["data"]:
                idx = item.get("index")
                score = item.get("score")
                if isinstance(idx, int) and 0 <= idx < len(candidates) and score is not None:
                    candidates[idx].score = float(score)
        elif isinstance(body, dict) and isinstance(body.get("results"), list):
            for item in body["results"]:
                idx = item.get("index")
                score = item.get("relevance_score")
                if isinstance(idx, int) and 0 <= idx < len(candidates) and score is not None:
                    candidates[idx].score = float(score)

        reranked = sorted(candidates, key=lambda x: x.score, reverse=True)
        print(f"Reranked scores: {[r.score for r in reranked]}")
        return reranked[:top_n]


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------

gemini_parser = GeminiResponseParser()
tokenizer     = AutoTokenizer.from_pretrained(RERANKER_MODEL)

CSV_FIELDNAMES = [
    'pair_id',
    'initial_node',
    'neighbor_node',
    'initial_persona',
    'target_persona',
    'initial_persona_score',
    'target_persona_score',
    'relationship_type',
    'chosen_prompt',
    'chosen_concepts',
    'chosen_context',
    'chosen_context_meta',
    'chosen_n_chunks',
    'chosen_average_rerank_score',
    'chosen_average_rerank_score_per_concept',
    'n_excluded_nodes',
    'chosen_reasoning',
    'chosen_abstract',
    'chosen_summary',
    'chosen_paper_id',
    'initial_scores',
    'neighbor_scores',
    'synthetic',
    'error',
]


def _serialize_logprobs(logprobs) -> tuple[str, str, str]:
    if logprobs is None:
        return '', '', ''
    chosen = json.dumps(
        [{'token': t.token, 'log_probability': t.log_probability}
         for t in logprobs.chosen_candidates],
        ensure_ascii=False,
    )
    top = json.dumps(
        [[{'token': t.token, 'log_probability': t.log_probability} for t in step]
         for step in logprobs.top_candidates],
        ensure_ascii=False,
    )
    avg = '' if logprobs.avg_log_prob is None else str(logprobs.avg_log_prob)
    return chosen, top, avg


def _serialize_usage(usage) -> str:
    return json.dumps({
        'prompt':     usage.prompt_tokens,
        'candidates': usage.candidates_tokens,
        'thoughts':   usage.thoughts_tokens,
        'total':      usage.total_tokens,
    })


def load_concepts(concepts_path: str) -> list[dict]:
    if not os.path.exists(concepts_path):
        raise FileNotFoundError(f"Concepts file not found at: {concepts_path}")
    with open(concepts_path, 'r', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def load_completed_pairs(output_file: str) -> set[str]:
    done: set[str] = set()
    if not os.path.exists(output_file):
        return done
    with open(output_file, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row.get('pair_id'):
                done.add(row['pair_id'])
    return done


def _parse_concepts(raw) -> List[str]:
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        try:
            return ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            pass
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
    return []


def load_exclusion_registry(path: str) -> dict[str, list[str]]:
    """Load pair_id → [excluded_node_ids] mapping from JSON."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_node_to_paper_id(classifications_path: str) -> dict[str, str]:
    """Build node_id → paperId mapping from classifications CSV."""
    mapping: dict[str, str] = {}
    with open(classifications_path, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row.get('node_id') and row.get('paperId'):
                mapping[row['node_id']] = row['paperId']
    return mapping


def _resolve_queries(concepts: List[str], original_query: str, query_source: str) -> List[str]:
    """Resolve retrieval queries from config mode and available concepts."""
    if query_source == 'original':
        return [original_query]
    return concepts if concepts else [original_query]


# ---------------------------------------------------------------------------
# Core retrieval step
# ---------------------------------------------------------------------------

async def _retrieve_rerank_per_query(
    concepts: List[str],
    original_query: str,
    query_source: str,
    embedder: VLLMEmbedder,
    retriever: WeaviateRetriever,
    reranker: VLLMReranker,
    http_client: httpx.AsyncClient,
    excluded_paper_ids: List[str] | None = None,
) -> List[RankedChunk]:
    """
    Implements per-subquery hybrid retrieval + reranking:

    Given the decomposed query → {Q1, Q2, Q3, ..., Qn}:
      1. Hybrid retrieval for each Qi → CONCEPTS_TOP_K candidates (BM25 + vector).
      2. Dedup all candidates globally by chunk_id before reranking.
      3. Single rerank pass over all unique candidates (efficient: the same
         original_query is used for all Qi's, so reranking a chunk multiple
         times would produce the same score).
      4. Per-Qi: select RERANK_TOP_K chunks with unique doc_ids using the
         globally-reranked scores, preserving per-Qi diversity.
      5. Global merge deduplicated by (doc_id, chunk_id).

    Falls back to the original query when no subqueries / concepts are present.
    """
    queries = _resolve_queries(concepts, original_query, query_source)

    # ── 1. Retrieve candidates for every Qi ───────────────────────────────
    # chunk_id → RankedChunk (first-seen wins; source_concept set to first Qi)
    all_candidates: dict[str, RankedChunk] = {}
    # qi → ordered list of chunk_ids retrieved for that Qi
    qi_chunk_ids: dict[str, list[str]] = {}

    for qi in queries:
        embedding  = await embedder.embed_text(qi)
        candidates = retriever.retrieve(qi, embedding, top_k=CONCEPTS_TOP_K,
                                        excluded_paper_ids=excluded_paper_ids)
        if not candidates:
            qi_chunk_ids[qi] = []
            continue

        print(f"  [Qi='{qi[:60]}'] {retriever.retrieval_mode} candidates: {len(candidates)}")
        qi_chunk_ids[qi] = [rc.chunk.chunk_id for rc in candidates]
        for rc in candidates:
            if rc.chunk.chunk_id not in all_candidates:
                rc.source_concept = qi
                all_candidates[rc.chunk.chunk_id] = rc

    if not all_candidates:
        return []

    # ── 2. Single rerank pass over all unique candidates ──────────────────
    unique_candidates = list(all_candidates.values())
    print(f"  Unique candidates across all Qi's (before rerank): {len(unique_candidates)}")
    reranked = await reranker.rerank(
        query=original_query,
        candidates=unique_candidates,
        top_n=len(unique_candidates),
        client=http_client,
    )
    reranked_by_id: dict[str, RankedChunk] = {rc.chunk.chunk_id: rc for rc in reranked}

    # ── 3. Per-Qi selection with global dedup: up to RERANK_TOP_K unique chunks ──
    #
    # For each Qi, iterate its reranked candidates in score order.  A candidate
    # is skipped (but the loop continues) if its doc_id is already seen for this
    # Qi or its chunk is already in the global pool.  After all Qi's are
    # processed, any unfilled slots are gap-filled from the global reranked pool.
    seen_chunk_keys: set             = set()
    merged_chunks:   List[RankedChunk] = []
    total_deficit = 0  # unfilled slots across all Qi's (due to score threshold or empty retrieval)

    for qi in queries:
        chunk_ids = qi_chunk_ids.get(qi, [])
        if not chunk_ids:
            total_deficit += RERANK_TOP_K
            continue

        # Sort this Qi's candidates by globally-reranked score (desc).
        qi_reranked = sorted(
            [reranked_by_id[cid] for cid in chunk_ids if cid in reranked_by_id],
            key=lambda x: x.score,
            reverse=True,
        )

        seen_doc_ids: set = set()
        selected_count    = 0

        for rc in qi_reranked:
            if selected_count >= RERANK_TOP_K:
                break

            doc_id    = rc.chunk.metadata['doc_id']
            chunk_key = (doc_id, rc.chunk.chunk_id)

            # Skip duplicate doc_ids within this Qi (per-Qi diversity).
            if doc_id in seen_doc_ids:
                continue
            seen_doc_ids.add(doc_id)

            # Skip chunks already in the global pool from an earlier Qi;
            # the loop continues to look for the next eligible chunk.
            if chunk_key in seen_chunk_keys:
                continue

            rc.source_concept = qi
            seen_chunk_keys.add(chunk_key)
            merged_chunks.append(rc)
            selected_count += 1

        total_deficit += RERANK_TOP_K - selected_count

    # ── 4. Gap-fill from global reranked pool ────────────────────────────────
    # For every slot left empty (not enough unique candidates per Qi),
    # take the next best chunk from the globally-reranked pool that is not yet
    # in the pool.
    if total_deficit > 0:
        for rc in reranked:  # already sorted by score desc
            if total_deficit <= 0:
                break
            chunk_key = (rc.chunk.metadata['doc_id'], rc.chunk.chunk_id)
            if chunk_key in seen_chunk_keys:
                continue
            rc.source_concept = 'global_pool'
            seen_chunk_keys.add(chunk_key)
            merged_chunks.append(rc)
            total_deficit -= 1

    return merged_chunks

# seen_chunk_keys: set             = set()
    # merged_chunks:   List[RankedChunk] = []
    # total_deficit = 0  # unfilled slots across all Qi's (due to score threshold or empty retrieval)

    # for qi in queries:
    #     chunk_ids = qi_chunk_ids.get(qi, [])
    #     if not chunk_ids:
    #         total_deficit += RERANK_TOP_K
    #         continue

    #     # Sort this Qi's candidates by globally-reranked score (desc).
    #     qi_reranked = sorted(
    #         [reranked_by_id[cid] for cid in chunk_ids if cid in reranked_by_id],
    #         key=lambda x: x.score,
    #         reverse=True,
    #     )

    #     seen_doc_ids: set = set()
    #     selected_count    = 0

    #     for rc in qi_reranked:
    #         if selected_count >= RERANK_TOP_K:
    #             break

    #         # Stop adding for this Qi once quality falls below threshold.
    #         if rc.score < 0.15:
    #             break

    #         doc_id    = rc.chunk.metadata['doc_id']
    #         chunk_key = (doc_id, rc.chunk.chunk_id)

    #         # Skip duplicate doc_ids within this Qi (per-Qi diversity).
    #         if doc_id in seen_doc_ids:
    #             continue
    #         seen_doc_ids.add(doc_id)

    #         # Skip chunks already in the global pool from an earlier Qi;
    #         # the loop continues to look for the next eligible chunk.
    #         if chunk_key in seen_chunk_keys:
    #             continue

    #         rc.source_concept = qi
    #         seen_chunk_keys.add(chunk_key)
    #         merged_chunks.append(rc)
    #         selected_count += 1

    #     total_deficit += RERANK_TOP_K - selected_count

    # # ── 4. Gap-fill from global reranked pool ────────────────────────────────
    # # For every slot that was left empty (score threshold hit or no candidates),
    # # take the next best chunk from the globally-reranked pool that is not yet
    # # in the pool.  Stop as soon as the score drops to 0.2 or below — we never
    # # promote low-quality chunks regardless of how many slots remain.
    # if total_deficit > 0:
    #     for rc in reranked:  # already sorted by score desc
    #         if total_deficit <= 0:
    #             break
    #         if rc.score <= 0.15:
    #             break
    #         chunk_key = (rc.chunk.metadata['doc_id'], rc.chunk.chunk_id)
    #         if chunk_key in seen_chunk_keys:
    #             continue
    #         rc.source_concept = 'global_pool'
    #         seen_chunk_keys.add(chunk_key)
    #         merged_chunks.append(rc)
    #         total_deficit -= 1

    # return merged_chunks

# async def _retrieve_rerank_per_query(
#     concepts: List[str],
#     original_query: str,
#     query_source: str,
#     embedder: VLLMEmbedder,
#     retriever: WeaviateRetriever,
#     reranker: VLLMReranker,
#     http_client: httpx.AsyncClient,
#     excluded_paper_ids: List[str] | None = None,
# ) -> List[RankedChunk]:
#     """
#     Implements per-subquery hybrid retrieval + reranking:
 
#     Given the decomposed query → {Q1, Q2, Q3, ..., Qn}:
#       1. Hybrid retrieval for each Qi → CONCEPTS_TOP_K candidates (BM25 + vector).
#       2. Rerank candidates for each Qi separately (against Qi itself).
#       3. Select RERANK_TOP_K chunks with unique doc_ids for each Qi
#          (best-scoring chunk per paper, up to RERANK_TOP_K papers).
 
#     Results across all Qi are merged into a single pool, globally
#     deduplicated by (doc_id, chunk_id).  Falls back to the original query
#     when no subqueries / concepts are present.
#     """
#     queries = _resolve_queries(concepts, original_query, query_source)
 
#     seen_chunk_keys: set             = set()
#     merged_chunks:   List[RankedChunk] = []
 
#     for qi in queries:
#         # ── 1. Embed Qi and run hybrid retrieval ──────────────────────────
#         embedding  = await embedder.embed_text(qi)
#         candidates = retriever.retrieve(qi, embedding, top_k=CONCEPTS_TOP_K,
#                                         excluded_paper_ids=excluded_paper_ids)
 
#         if not candidates:
#             continue
 
#         print(f"  [Qi='{qi[:60]}'] {retriever.retrieval_mode} candidates: {len(candidates)}")
 
#         # ── 2. Rerank candidates against Qi ───────────────────────────────
#         reranked = await reranker.rerank(
#             query=qi,
#             candidates=candidates,
#             # Score all candidates; unique-doc_id selection is done below.
#             top_n=len(candidates),
#             client=http_client,
#         )
 
#         # ── 3. Select rerank_top_k unique doc_ids ─────────────────────────
#         seen_doc_ids: set           = set()
#         selected:     List[RankedChunk] = []
 
#         for rc in reranked:                          # already sorted by score desc
#             doc_id = rc.chunk.metadata['doc_id']
#             # if doc_id not in seen_doc_ids:
#             if doc_id in seen_doc_ids:
#                 continue
#             seen_doc_ids.add(doc_id)
#             rc.source_concept = qi
#             selected.append(rc)
#             if len(selected) >= RERANK_TOP_K:
#                 break
 
#         # print(f"  [Qi='{qi[:60]}'] selected {len(selected)} unique doc_ids")
 
#         # ── 4. Merge into global pool (dedup by chunk key) ────────────────
#         for rc in selected:
#             chunk_key = (rc.chunk.metadata['doc_id'], rc.chunk.chunk_id)
#             if chunk_key not in seen_chunk_keys:
#                 seen_chunk_keys.add(chunk_key)
#                 merged_chunks.append(rc)
 
#     return merged_chunks

async def _retrieve_and_dedup(
    concepts: List[str],
    original_query: str,
    query_source: str,
    embedder: VLLMEmbedder,
    retriever: WeaviateRetriever,
    excluded_paper_ids: List[str] | None = None,
) -> List[RankedChunk]:
    """
    For each concept:
      1. Embed it on-the-fly via vLLM.
      2. Run a **hybrid** BM25 + dense vector search in Weaviate.

    The concept text is passed as the BM25 keyword query alongside the dense
    vector, so rare technical terms in subqueries (algorithm names, acronyms,
    method-specific notation) are matched exactly by BM25 in addition to
    being matched semantically by the dense retriever.

    Falls back to the original query when no concepts are present.
    Deduplicates across concepts by (doc_id, chunk_id), keeping the first hit.
    """
    seen_ids: set            = set()
    unique_chunks: List[RankedChunk] = []

    queries = _resolve_queries(concepts, original_query, query_source)

    for concept in queries:
        embedding = await embedder.embed_text(concept)
        # Pass both the raw concept text (for BM25) and its embedding (for
        # dense ANN) so Weaviate can fuse them via Reciprocal Rank Fusion.
        results = retriever.retrieve(
            query_text=concept,
            query_vector=embedding,
            top_k=CONCEPTS_TOP_K,
            excluded_paper_ids=excluded_paper_ids,
        )

        for r in results:
            chunk_key = (r.chunk.metadata['doc_id'], r.chunk.chunk_id)
            if chunk_key not in seen_ids:
                seen_ids.add(chunk_key)
                unique_chunks.append(
                    RankedChunk(chunk=r.chunk, score=r.score, source_concept=concept)
                )

    return unique_chunks


# ---------------------------------------------------------------------------
# Per-item pipeline
# ---------------------------------------------------------------------------

async def process_item(
    item: dict,
    query_source: str,
    exclusion_registry: dict,
    node_to_paper_id: dict,
    embedder: VLLMEmbedder,
    retriever: WeaviateRetriever,
    reranker: VLLMReranker,
    http_client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
) -> dict:
    """
    Full RAG pipeline for one row of the concepts CSV.

    Retrieval is run for the chosen prompt/concepts with an exclusion filter
    derived from the global exclusion registry so that retrieval cannot surface
    the initial or neighbor node (or any node transitively related to this pair).
    """
    initial_node  = item['initial_node']
    neighbor_node = item['neighbor_node']
    key           = {'initial_node': initial_node, 'neighbor_node': neighbor_node}

    # --- Resolve pair_id and build per-pair excluded paper IDs ------------
    pair_id            = item.get('pair_id', '')
    excluded_node_ids  = exclusion_registry.get(pair_id, []) if pair_id else []
    excluded_paper_ids = [
        node_to_paper_id[nid]
        for nid in excluded_node_ids
        if nid in node_to_paper_id
    ]

    chosen_prompt   = item.get('chosen_prompt', '')
    chosen_concepts = _parse_concepts(item.get('chosen_concepts', []))

    max_retries = 3
    base_delay  = 2

    async with semaphore:
        # ------------------------------------------------------------------
        # Chosen retrieval (with exclusion filter)
        # ------------------------------------------------------------------
        for attempt in range(max_retries):
            try:
                chosen_chunks = await _retrieve_rerank_per_query(
                    chosen_concepts, chosen_prompt, query_source, embedder, retriever, reranker,
                    http_client, excluded_paper_ids=excluded_paper_ids,
                )
                print(f"[chosen] chunks after retrieval+rerank: {len(chosen_chunks)}")
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    wait = (base_delay ** attempt) + random.uniform(0, 1)
                    print(f"\nChosen retrieval error, retrying in {wait:.2f}s "
                          f"(attempt {attempt + 1}/{max_retries}): {repr(e)}")
                    await asyncio.sleep(wait)
                else:
                    return {**key, 'pair_id': pair_id or '', 'error': f'Chosen retrieval error: {repr(e)}'}

        # ------------------------------------------------------------------
        # Build context string and metadata
        # ------------------------------------------------------------------
        def _build_context(chunks: List[RankedChunk]):
            ctx_str = "\n\n".join(
                f"[Source: Document ID: {rc.chunk.metadata.get('doc_id', 'Unknown')}, "
                f"Chunk ID: {rc.chunk.chunk_id}]\n{rc.chunk.content}"
                for rc in chunks
            )
            ctx_meta = [
                {
                    'chunk_id':       rc.chunk.chunk_id,
                    'doc_id':         rc.chunk.metadata['doc_id'],
                    'rerank_score':   round(rc.score, 4),
                    'source_concept': rc.source_concept,
                    'chunk_length':   len(rc.chunk.content),
                }
                for rc in chunks
            ]
            return ctx_str, ctx_meta

        chosen_ctx, chosen_meta = _build_context(chosen_chunks)

        def _avg_score(chunks: List[RankedChunk]) -> float:
            return round(sum(rc.score for rc in chunks) / len(chunks), 4) if chunks else 0.0

        def _avg_score_per_concept(chunks: List[RankedChunk]) -> dict:
            per_concept: dict[str, list[float]] = {}
            for rc in chunks:
                per_concept.setdefault(rc.source_concept, []).append(rc.score)
            return {c: round(sum(s) / len(s), 4) for c, s in per_concept.items()}

        return {
            'pair_id':                                  pair_id or '',
            **key,
            'initial_persona':                          item.get('initial_persona', ''),
            'target_persona':                           item.get('target_persona', ''),
            'initial_persona_score':                    item.get('initial_persona_score', ''),
            'target_persona_score':                     item.get('target_persona_score', ''),
            'relationship_type':                        item.get('relationship_type', ''),
            'chosen_prompt':                            chosen_prompt,
            'chosen_concepts':                          json.dumps(chosen_concepts),
            'chosen_reasoning':                         item.get('chosen_reasoning', ''),
            'chosen_abstract':                          item.get('chosen_abstract', ''),
            'chosen_summary':                           item.get('chosen_summary', ''),
            'chosen_paper_id':                          item.get('chosen_paper_id', ''),
            'initial_scores':                           item.get('initial_scores', ''),
            'neighbor_scores':                          item.get('neighbor_scores', ''),
            'synthetic':                                item.get('synthetic', ''),
            'chosen_context':                           chosen_ctx,
            'chosen_context_meta':                      json.dumps(chosen_meta),
            'chosen_n_chunks':                          len(chosen_chunks),
            'chosen_average_rerank_score':              _avg_score(chosen_chunks),
            'chosen_average_rerank_score_per_concept':  json.dumps(_avg_score_per_concept(chosen_chunks)),
            'n_excluded_nodes':                         len(excluded_node_ids),
            'error':                                    '',
        }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main():
    print(f"Loading concepts from: {CONCEPTS_FILE}")
    all_rows = load_concepts(CONCEPTS_FILE)
    print(f"Total rows loaded    : {len(all_rows)}")

    done_pairs = load_completed_pairs(CONTEXT_FILE)
    print(f"Already completed    : {len(done_pairs)}")

    tasks_to_run = [
        row for row in all_rows
        if row.get('pair_id') not in done_pairs
    ]

    tasks_to_run = random.sample(tasks_to_run, min(1000, len(tasks_to_run))) 

    print(f"Remaining to run     : {len(tasks_to_run)}")

    if not tasks_to_run:
        print("All tasks completed. Nothing to do.")
        return

    # --- Load exclusion data ---------------------------------------------
    print(f"\nLoading exclusion registry from: {EXCLUSION_REGISTRY_FILE}")
    exclusion_registry = load_exclusion_registry(EXCLUSION_REGISTRY_FILE)
    print(f"  {len(exclusion_registry)} pairs in registry")

    print(f"Building node→paperId mapping from: {CLASSIFICATIONS_FILE}")
    node_to_paper_id = build_node_to_paper_id(CLASSIFICATIONS_FILE)
    print(f"  {len(node_to_paper_id)} node IDs mapped")

    print("\nRetrieval runtime configuration:")
    print(f"  weaviate_mode: {WEAVIATE_MODE}")
    print(f"  mode         : {RETRIEVAL_MODE}")
    print(f"  query_source : {QUERY_SOURCE}")

    # --- Connect to Weaviate ---------------------------------------------
    print(
        f"\nConnecting to Weaviate API at {WEAVIATE_HOST}:{WEAVIATE_PORT} "
        f"(grpc={WEAVIATE_GRPC_PORT}, secure={WEAVIATE_HTTP_SECURE}) ..."
    )
    retriever = WeaviateRetriever(
        collection_name=WEAVIATE_COLLECTION,
        doc_types=WEAVIATE_DOC_TYPES,
        host=WEAVIATE_HOST,
        port=WEAVIATE_PORT,
        grpc_port=WEAVIATE_GRPC_PORT,
        api_key=WEAVIATE_API_KEY,
        http_secure=WEAVIATE_HTTP_SECURE,
        grpc_secure=WEAVIATE_HTTP_SECURE,
        hybrid_alpha=HYBRID_ALPHA,
        retrieval_mode=RETRIEVAL_MODE,
    )

    resp = retriever._collection.query.fetch_objects(
        limit=1000,
        return_properties=["type"],
    )
    types = sorted({o.properties.get("type") for o in resp.objects if o.properties.get("type")})
    print("Available types:", types)

    # --- vLLM embedder (query concepts only — corpus lives in Weaviate) ---
    embedder = VLLMEmbedder(
        model=EMBEDDING_MODEL,
        api_url=EMBEDDER_API_URL,
        api_key=VLLM_API_KEY,
        device='cuda:0',
    )

    reranker = VLLMReranker(
        model=RERANKER_MODEL,
        api_url=RERANKER_API_URL,
        api_key=VLLM_API_KEY,
    )

    print(f"Processing {len(tasks_to_run)} items...")

    semaphore   = asyncio.Semaphore(SEMAPHORE_SIZE)
    file_exists = os.path.exists(CONTEXT_FILE) and os.path.getsize(CONTEXT_FILE) > 0

    success_count = 0
    fail_count    = 0

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(500.0, connect=10.0)) as http_client:
            tasks = [
                process_item(row, QUERY_SOURCE, exclusion_registry, node_to_paper_id,
                             embedder, retriever, reranker, http_client, semaphore)
                for row in tasks_to_run
            ]

            with open(CONTEXT_FILE, 'a', newline='', encoding='utf-8') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDNAMES, extrasaction='ignore')
                if not file_exists:
                    writer.writeheader()
                    csv_file.flush()

                for future in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
                    result = await future
                    writer.writerow(result)
                    csv_file.flush()

                    if result.get('error'):
                        print(f"\nFailed  initial_node={result.get('initial_node')}  "
                              f"neighbor_node={result.get('neighbor_node')} — {result.get('error')}")
                        fail_count += 1
                    else:
                        success_count += 1

    finally:
        retriever.close()   # always release the Weaviate connection

    print("-" * 30)
    print("Summary:")
    print(f"  Total Attempted : {len(tasks)}")
    print(f"  Successful      : {success_count}")
    print(f"  Failed          : {fail_count}")
    print("-" * 30)


if __name__ == "__main__":
    try:
        asyncio.run(main())
        print(f"Done! Hypotheses saved to {CONTEXT_FILE}")
    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"\nCritical error during execution: {e}")