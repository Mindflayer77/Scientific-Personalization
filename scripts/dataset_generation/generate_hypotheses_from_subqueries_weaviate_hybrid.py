import os
import ast
import asyncio
import json
import random
import torch
import csv
import httpx
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
HYPOTHESES_FILE = os.path.join(HYPOTHESES_DIR, CONFIG['files']['hypotheses'])

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

EMBEDDER_API_URL = f"{VLLM_API_URL}:{_EMBEDDER_PORT}/v1"
RERANKER_API_URL = f"{VLLM_API_URL}:{_RERANKER_PORT}/v1"

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
WEAVIATE_HOST       = CONFIG.get('weaviate', {}).get('host', 'localhost')
WEAVIATE_PORT       = CONFIG.get('weaviate', {}).get('port', 8080)
WEAVIATE_COLLECTION = CONFIG.get('weaviate', {}).get('collection', 'ResearchPapers')
# Restrict ANN search to these `type` values.  Set to [] to search all types.
WEAVIATE_DOC_TYPES: List[str] = CONFIG.get('weaviate', {}).get('doc_types', ['CHUNK'])

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

# Hybrid search balance: 0.0 = pure BM25, 1.0 = pure dense vector, 0.5 = balanced.
# Values around 0.5–0.7 work well for scientific corpora where both exact keyword
# matching (rare technical terms, algorithm names) and semantic similarity matter.
HYBRID_ALPHA: float = CONFIG.get('retrieval', {}).get('hybrid_alpha', 0.5)

load_dotenv()

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
        hybrid_alpha: float = 0.5,
    ):
        self._doc_types    = doc_types
        self._hybrid_alpha = hybrid_alpha
        self._wv_client    = weaviate.connect_to_local(host=host, port=port)
        self._collection   = self._wv_client.collections.get(collection_name)
        print(
            f"WeaviateRetriever ready — collection='{collection_name}' "
            f"at {host}:{port}  type filter={doc_types or 'none'}  "
            f"hybrid_alpha={hybrid_alpha} (0=BM25, 1=dense)"
        )

    def close(self):
        """Release the Weaviate connection."""
        self._wv_client.close()

    def retrieve(
        self,
        query_text: str,
        query_vector: torch.Tensor,
        top_k: int,
    ) -> List[RankedChunk]:
        """
        Run a **hybrid** BM25 + dense vector query via Weaviate's built-in
        Reciprocal Rank Fusion and return up to top_k RankedChunks.

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
            return_metadata=wvc.query.MetadataQuery(score=True),
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
            # hybrid() returns an RRF fusion score via metadata.score
            # (range 0–1, higher = better match).
            score = obj.metadata.score if obj.metadata.score is not None else 0.0
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
    Calls vLLM's /v1/rerank endpoint (Cohere-compatible) to score candidate
    chunks against the original query, then returns the top-N highest-scoring.
    """

    def __init__(self, model: str, api_url: str, api_key: str):
        self.model = model
        base = api_url.rstrip('/')
        self.rerank_url = base if base.endswith('/v1/rerank') else f"{base}/rerank"
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
            "model":     self.model,
            "query":     f"<Instruct>{RERANKER_INSTRUCTION}\n<Query>{query}",
            "documents": documents,
            "top_n":     top_n,
        }
        response = await client.post(self.rerank_url, headers=self._headers, json=payload)
        response.raise_for_status()

        for item in response.json()["results"]:
            candidates[item["index"]].score = item["relevance_score"]

        reranked = sorted(candidates, key=lambda x: x.score, reverse=True)
        print(f"Reranked scores: {[r.score for r in reranked]}")
        return reranked[:top_n]


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------

gemini_parser = GeminiResponseParser()
tokenizer     = AutoTokenizer.from_pretrained(RERANKER_MODEL)

CSV_FIELDNAMES = [
    'user_id', 'topic_id',
    'is_answerable', 'hypothesis', 'falsification_criteria',
    'prompt', 'concepts', 'retrieved_context', 'context_length', 'context_meta', 'n_chunks',
    'reasoning', 'answer', 'finish_reason', 'model_version',
    'usage',
    'logprobs_chosen', 'logprobs_top', 'avg_log_prob',
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


def load_completed_pairs(output_file: str) -> set[tuple[str, str]]:
    done: set[tuple[str, str]] = set()
    if not os.path.exists(output_file):
        return done
    with open(output_file, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row.get('user_id') and row.get('topic_id'):
                done.add((row['user_id'], row['topic_id']))
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


# ---------------------------------------------------------------------------
# Core retrieval step
# ---------------------------------------------------------------------------

async def _retrieve_rerank_per_query(
    concepts: List[str],
    original_query: str,
    embedder: VLLMEmbedder,
    retriever: WeaviateRetriever,
    reranker: VLLMReranker,
    http_client: httpx.AsyncClient,
) -> List[RankedChunk]:
    """
    Implements per-subquery hybrid retrieval + reranking:
 
    Given the decomposed query → {Q1, Q2, Q3, ..., Qn}:
      1. Hybrid retrieval for each Qi → CONCEPTS_TOP_K candidates (BM25 + vector).
      2. Rerank candidates for each Qi separately (against Qi itself).
      3. Select RERANK_TOP_K chunks with unique doc_ids for each Qi
         (best-scoring chunk per paper, up to RERANK_TOP_K papers).
 
    Results across all Qi are merged into a single pool, globally
    deduplicated by (doc_id, chunk_id).  Falls back to the original query
    when no subqueries / concepts are present.
    """
    queries = concepts if concepts else [original_query]
 
    seen_chunk_keys: set             = set()
    merged_chunks:   List[RankedChunk] = []
 
    for qi in queries:
        # ── 1. Embed Qi and run hybrid retrieval ──────────────────────────
        embedding  = await embedder.embed_text(qi)
        candidates = retriever.retrieve(qi, embedding, top_k=CONCEPTS_TOP_K)
 
        if not candidates:
            continue
 
        print(f"  [Qi='{qi[:60]}'] hybrid candidates: {len(candidates)}")
 
        # ── 2. Rerank candidates against Qi ───────────────────────────────
        reranked = await reranker.rerank(
            query=original_query,
            candidates=candidates,
            # Score all candidates; unique-doc_id selection is done below.
            top_n=len(candidates),
            client=http_client,
        )
 
        # ── 3. Select rerank_top_k unique doc_ids ─────────────────────────
        seen_doc_ids: set           = set()
        selected:     List[RankedChunk] = []
 
        for rc in reranked:                          # already sorted by score desc
            doc_id = rc.chunk.metadata['doc_id']
            # if doc_id not in seen_doc_ids:
            seen_doc_ids.add(doc_id)
            rc.source_concept = qi
            selected.append(rc)
            if len(selected) >= RERANK_TOP_K:
                break
 
        # print(f"  [Qi='{qi[:60]}'] selected {len(selected)} unique doc_ids")
 
        # ── 4. Merge into global pool (dedup by chunk key) ────────────────
        for rc in selected:
            chunk_key = (rc.chunk.metadata['doc_id'], rc.chunk.chunk_id)
            if chunk_key not in seen_chunk_keys:
                seen_chunk_keys.add(chunk_key)
                merged_chunks.append(rc)
 
    return merged_chunks

async def _retrieve_and_dedup(
    concepts: List[str],
    original_query: str,
    embedder: VLLMEmbedder,
    retriever: WeaviateRetriever,
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

    queries = concepts if concepts else [original_query]

    for concept in queries:
        embedding = await embedder.embed_text(concept)
        # Pass both the raw concept text (for BM25) and its embedding (for
        # dense ANN) so Weaviate can fuse them via Reciprocal Rank Fusion.
        results = retriever.retrieve(
            query_text=concept,
            query_vector=embedding,
            top_k=CONCEPTS_TOP_K,
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
    user_map: Dict,
    embedder: VLLMEmbedder,
    retriever: WeaviateRetriever,
    reranker: VLLMReranker,
    gemini_client: GeminiClient,
    http_client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
) -> dict:
    """
    Full RAG + hypothesis-generation pipeline for one concept-CSV row.

    Steps:
      1. Per-concept embedding (vLLM) + hybrid BM25+dense retrieval from Weaviate → deduplication
      2. Reranking against the original query string
      3. LLM hypothesis generation
    """
    user_id  = item['user_id']
    topic_id = item['topic_id']
    key      = {'user_id': user_id, 'topic_id': topic_id}

    persona = user_map.get(user_id)
    if not persona:
        return {**key, 'error': 'Persona not found'}

    original_query: str = item['generated_prompt']
    concepts: List[str] = _parse_concepts(item.get('core_concepts', []))

    max_retries = 3
    base_delay  = 2

    async with semaphore:
        # ------------------------------------------------------------------
        # Step 1: Per-concept retrieval from Weaviate → deduplication
        # ------------------------------------------------------------------
        for attempt in range(max_retries):
            try:
                reranked_chunks = await _retrieve_rerank_per_query(
                    concepts, original_query, embedder, retriever, reranker, http_client
                )
                print("Candidate chunks after per-Qi retrieval+rerank:", len(reranked_chunks))
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = (base_delay ** attempt) + random.uniform(0, 1)
                    print(f"\nRetrieval error, retrying in {wait_time:.2f}s "
                          f"(attempt {attempt + 1}/{max_retries}): {e}")
                    await asyncio.sleep(wait_time)
                else:
                    return {**key, 'error': f'Retrieval error: {e}'}
        # for attempt in range(max_retries):
        #     try:
        #         candidate_chunks = await _retrieve_and_dedup(
        #             concepts, original_query, embedder, retriever
        #         )
        #         print("Candidate chunks number:", len(candidate_chunks))
        #         break
        #     except Exception as e:
        #         if attempt < max_retries - 1:
        #             wait_time = (base_delay ** attempt) + random.uniform(0, 1)
        #             print(f"\nRetrieval error, retrying in {wait_time:.2f}s "
        #                   f"(attempt {attempt + 1}/{max_retries}): {e}")
        #             await asyncio.sleep(wait_time)
        #         else:
        #             return {**key, 'error': f'Retrieval error: {e}'}

        # ------------------------------------------------------------------
        # Step 2: Reranking against the original query string
        # ------------------------------------------------------------------
        # for attempt in range(max_retries):
        #     try:
        #         reranked_chunks = await reranker.rerank(
        #             query=original_query,
        #             candidates=candidate_chunks,
        #             top_n=RERANK_TOP_K,
        #             client=http_client,
        #         )
        #         print(len(reranked_chunks))
        #         print("Length of reranked content:", sum(len(rc.chunk.content) for rc in reranked_chunks))
        #         break
        #     except Exception as e:
        #         if attempt < max_retries - 1:
        #             wait_time = (base_delay ** attempt) + random.uniform(0, 1)
        #             print(f"\nReranking error, retrying in {wait_time:.2f}s "
        #                   f"(attempt {attempt + 1}/{max_retries}): {e}")
        #             await asyncio.sleep(wait_time)
        #         else:
        #             return {**key, 'error': f'Reranking error: {e}'}

        context_str = "\n\n".join(
            f"[Source: Document ID: {rc.chunk.metadata.get('doc_id', 'Unknown')}, "
            f"Chunk ID: {rc.chunk.chunk_id}]\n{rc.chunk.content}"
            for rc in reranked_chunks
        )
        context_meta = [
            {
                'chunk_id':       rc.chunk.chunk_id,
                'doc_id':         rc.chunk.metadata['doc_id'],
                'rerank_score':   round(rc.score, 4),
                'source_concept': rc.source_concept,
                'chunk_length':   len(rc.chunk.content)
            }
            for rc in reranked_chunks
        ]

        # ------------------------------------------------------------------
        # Step 3: Hypothesis generation
        # ------------------------------------------------------------------
        persona_desc = persona_template.render(
            display_name=persona['display_name'],
            core_philosophy=persona['core_philosophy'],
            areas_of_expertise=", ".join(persona['areas_of_expertise']),
            communication_style=persona['communication_style'],
            what_i_look_for=", ".join(persona['what_i_look_for']),
            what_i_reject=", ".join(persona['what_i_reject']),
        )

        system_prompt_str = system_template.render()
        user_prompt_str   = user_template.render(
            query=original_query,
            persona=persona_desc,
            context=context_str,
        )

        print(user_prompt_str)

        for attempt in range(max_retries):
            try:
                return {
                    **key,
                    'is_answerable':          "",
                    'hypothesis':             "",
                    'falsification_criteria': "",
                    'prompt':                 original_query,
                    'concepts':               concepts,
                    'retrieved_context':      context_str,
                    'context_length':         len(" ".join([rc.chunk.content for rc in reranked_chunks])),
                    'context_meta':           json.dumps(context_meta),
                    'n_chunks':               len(reranked_chunks),
                    'reasoning':              "",
                    'answer':                 "",
                    'finish_reason':          "",
                    'model_version':          "",
                    'usage':                  "",
                    'logprobs_chosen':        "",
                    'logprobs_top':           "",
                    'avg_log_prob':           "",
                }

            except Exception as e:
                if "429" in str(e) or "Resource exhausted" in str(e):
                    wait_time = (base_delay ** attempt) + random.uniform(0, 1)
                    print(f"\nRate limited. Retrying in {wait_time:.2f}s "
                          f"(attempt {attempt + 1}/{max_retries})...")
                    await asyncio.sleep(wait_time)
                else:
                    return {**key, 'error': f'Generation error: {e}'}

        return {**key, 'error': 'Max retries exceeded'}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main():
    print(f"Loading concepts from: {CONCEPTS_FILE}")
    all_rows = load_concepts(CONCEPTS_FILE)
    print(f"Total rows loaded    : {len(all_rows)}")

    done_pairs = load_completed_pairs(HYPOTHESES_FILE)
    print(f"Already completed    : {len(done_pairs)}")

    tasks_to_run = [
        row for row in all_rows
        if (row['user_id'], row['topic_id']) not in done_pairs
    ][:50]
    print(f"Remaining to run     : {len(tasks_to_run)}")

    if not tasks_to_run:
        print("All tasks completed. Nothing to do.")
        return

    # --- Connect to Weaviate (replaces index.load / load_index_from_parquet) ---
    print(f"\nConnecting to Weaviate at {WEAVIATE_HOST}:{WEAVIATE_PORT} ...")
    retriever = WeaviateRetriever(
        collection_name=WEAVIATE_COLLECTION,
        doc_types=WEAVIATE_DOC_TYPES,
        host=WEAVIATE_HOST,
        port=WEAVIATE_PORT,
        hybrid_alpha=HYBRID_ALPHA,
    )

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

    gemini_client = GeminiClient(project_id=os.environ['PROJECT_ID'])

    print("Loading personas...")
    try:
        with open(PERSONAS_FILE, 'r') as f:
            user_map = {u['persona_id']: u for u in json.load(f)}
    except FileNotFoundError as e:
        print(f"Error loading personas: {e}")
        retriever.close()
        raise

    print(f"Processing {len(tasks_to_run)} items...")

    semaphore   = asyncio.Semaphore(SEMAPHORE_SIZE)
    file_exists = os.path.exists(HYPOTHESES_FILE) and os.path.getsize(HYPOTHESES_FILE) > 0

    success_count = 0
    fail_count    = 0

    try:
        async with httpx.AsyncClient(timeout=60.0) as http_client:
            tasks = [
                process_item(row, user_map, embedder, retriever, reranker,
                             gemini_client, http_client, semaphore)
                for row in tasks_to_run
            ]

            with open(HYPOTHESES_FILE, 'a', newline='', encoding='utf-8') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDNAMES, extrasaction='ignore')
                if not file_exists:
                    writer.writeheader()
                    csv_file.flush()

                for future in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
                    result = await future
                    writer.writerow(result)
                    csv_file.flush()

                    if result.get('error'):
                        print(f"\nFailed  user={result.get('user_id')}  "
                              f"topic={result.get('topic_id')} — {result.get('error')}")
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
        print(f"Done! Hypotheses saved to {HYPOTHESES_FILE}")
    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"\nCritical error during execution: {e}")