#!/usr/bin/env python3
"""Build a neighbourhood + DPO-pair dataset from Neo4j for sampled initial nodes.

For each accepted initial node:
  - Retrieves up to --n-neighbours nodes from its neighbourhood in Neo4j.
  - Guarantees that no initial node ever appears in another initial node's neighbourhood.
  - Selects DPO personas from the node's scores:
      chosen  : persona with score == 5
      rejected: persona with score == 3  → row 1
      rejected: persona with score == 2  → row 2 (omitted if no score-2 persona exists)
  - Node is skipped (and the next pool candidate is tried) if:
      * no persona has score 5
      * no persona has score 3
      * neighbourhood cannot be satisfied up to --max-depth

Output CSV columns (up to 2 rows per node):
  node_id, abstract, summary, paper_id, scores, neighbours,
  chosen_persona, rejected_persona, rejected_score

  `neighbours` is a JSON-encoded list of neighbour node_ids.

Usage example:
  python scripts_2/build_neighbourhood_dataset.py \\
      --input data/classifications.csv \\
      --output data/dpo_neighbourhood_dataset.csv \\
      --n-initial 10000 \\
      --n-neighbours 10 \\
      --depth 1 \\
      --max-depth 3
"""

import argparse
import csv
import json
import os
import random
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Environment / Neo4j helpers
# ---------------------------------------------------------------------------


def load_environment(env_file: str) -> None:
    if not os.path.exists(env_file):
        raise FileNotFoundError(f"Environment file not found: {env_file}")
    load_dotenv(env_file)


def get_driver():
    vps_ip = os.getenv("VPS_IP")
    user = os.getenv("NEO4J_USER")
    password = os.getenv("NEO4J_PASSWORD")

    if not all([vps_ip, user, password]):
        raise EnvironmentError(
            "Missing Neo4j connection variables. "
            "Ensure VPS_IP, NEO4J_USER, and NEO4J_PASSWORD are set."
        )

    uri = f"bolt://{vps_ip}:7687"
    return GraphDatabase.driver(uri, auth=(user, password))


# ---------------------------------------------------------------------------
# CSV loading helpers
# ---------------------------------------------------------------------------


def load_node_metadata(input_csv: Path) -> dict[str, dict]:
    """Return {node_id: {abstract, summary, paper_id, scores}} from the CSV."""
    metadata: dict[str, dict] = {}
    with input_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            node_id = row.get("node_id", "").strip()
            abstract = row.get("abstract", "").strip()
            summary = row.get("summary", "").strip()
            if not node_id or not abstract or not summary:
                continue

            metadata[node_id] = {
                "abstract": abstract,
                "summary": summary,
                "paper_id": row.get("paperId", ""),
                "scores": row.get("scores", ""),
            }
    return metadata


def load_already_processed(output_csv: Path) -> set[str]:
    """Return the set of node_ids already written to the output file."""
    if not output_csv.exists():
        return set()
    processed: set[str] = set()
    with output_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            node_id = row.get("node_id", "").strip()
            if node_id:
                processed.add(node_id)
    return processed


# ---------------------------------------------------------------------------
# DPO persona selection
# ---------------------------------------------------------------------------


def select_dpo_personas(
    scores_raw: str,
) -> tuple[str | None, str | None, str | None]:
    """Parse scores JSON and return (chosen, rejected_score3, rejected_score2).

    chosen          : alphabetically first persona with score == 5, or None
    rejected_score3 : alphabetically first persona with score == 3, or None
    rejected_score2 : alphabetically first persona with score == 2, or None
    """
    try:
        scores: dict = json.loads(scores_raw)
    except Exception:
        return None, None, None

    if not isinstance(scores, dict):
        return None, None, None

    def _first(target_score: int) -> str | None:
        candidates = sorted(k for k, v in scores.items() if v == target_score)
        return candidates[0] if candidates else None

    return _first(5), _first(3), _first(2)


# ---------------------------------------------------------------------------
# Neo4j neighbourhood query
# ---------------------------------------------------------------------------


def fetch_neighbourhood(
    driver,
    node_id: str,
    depth: int,
    n_neighbours: int,
    initial_node_ids: set[str],
    max_depth: int,
) -> list[str] | None:
    """Return exactly n_neighbours neighbour node_ids (excluding initial nodes).

    Starts at `depth` hops and increases up to `max_depth` if fewer than
    n_neighbours non-initial neighbours are found.  Returns None when even
    at max_depth the node cannot supply enough neighbours (caller should
    discard the node).

    Variable-length path bounds are embedded directly in the Cypher string
    (after validation as positive integers) because Neo4j does not support
    Cypher parameters for path-length bounds.
    """
    depth = max(1, int(depth))
    max_depth = max(depth, int(max_depth))

    # Fetch more than needed so we can filter out initial nodes and still
    # have enough neighbours.  Cap the over-fetch to avoid huge result sets.
    fetch_limit = min(n_neighbours * 4 + len(initial_node_ids), 50_000)

    for current_depth in range(depth, max_depth + 1):
        cypher = f"""
            MATCH (n)-[*1..{current_depth}]-(m)
            WHERE elementId(n) = $node_id
              AND elementId(m) <> $node_id
            RETURN DISTINCT elementId(m) AS neighbour_id
            LIMIT $limit
        """
        with driver.session() as session:
            result = session.run(cypher, node_id=node_id, limit=fetch_limit)
            raw = [record["neighbour_id"] for record in result]

        # Exclude all other initial nodes from this neighbourhood.
        filtered = [nid for nid in raw if nid not in initial_node_ids]

        if len(filtered) >= n_neighbours:
            return filtered[:n_neighbours]

    # Could not satisfy n_neighbours even at max_depth.
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build neighbourhood + DPO-pair dataset from Neo4j for sampled initial nodes."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/classifications.csv",
        help="Input CSV path (must have a node_id column). Default: data/classifications.csv",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/dpo_neighbourhood_dataset.csv",
        help="Output CSV path. Default: data/dpo_neighbourhood_dataset.csv",
    )
    parser.add_argument(
        "--n-initial",
        type=int,
        default=10_000,
        help="Number of accepted initial nodes to collect. Default: 10000",
    )
    parser.add_argument(
        "--n-neighbours",
        type=int,
        default=10,
        help="Exact number of neighbours required per initial node. Default: 10",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=1,
        help="Starting hop depth for neighbourhood expansion (>= 1). Default: 1",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=3,
        help=(
            "Maximum hop depth tried before discarding a node that cannot "
            "supply enough neighbours. Must be >= --depth. Default: 3"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling. Default: 42",
    )
    parser.add_argument(
        "--env-file",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / ".env_db"),
        help="Path to .env file with Neo4j credentials.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    load_environment(args.env_file)

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    # ------------------------------------------------------------------
    # Load metadata
    # ------------------------------------------------------------------
    print(f"Loading node metadata from {input_path} ...")
    metadata = load_node_metadata(input_path)
    print(f"  {len(metadata):,} unique nodes found in input CSV.")

    # ------------------------------------------------------------------
    # Pre-filter: keep only nodes that have score=5 AND score=3 personas
    # ------------------------------------------------------------------
    dpo_valid: list[str] = []
    for node_id, meta in metadata.items():
        chosen, rej3, _ = select_dpo_personas(meta["scores"])
        if chosen is not None and rej3 is not None:
            dpo_valid.append(node_id)

    print(f"  {len(dpo_valid):,} nodes pass DPO score filter (have score=5 and score=3 personas).")

    if not dpo_valid:
        print("No valid nodes found. Exiting.")
        return

    random.seed(args.seed)
    random.shuffle(dpo_valid)

    # ------------------------------------------------------------------
    # Resume: skip node_ids already written to output
    # ------------------------------------------------------------------
    already_done = load_already_processed(output_path)
    if already_done:
        print(f"  Resuming: {len(already_done):,} node_ids already in output, skipping them.")

    # Pool of candidates not yet processed (preserve shuffled order)
    candidate_pool = [nid for nid in dpo_valid if nid not in already_done]
    n_still_needed = args.n_initial - len(already_done)

    if n_still_needed <= 0:
        print("Target already reached. Nothing to do.")
        return

    print(f"  Need {n_still_needed:,} more nodes. Pool size: {len(candidate_pool):,}.")

    # ------------------------------------------------------------------
    # Open output CSV (append mode for resumability)
    # ------------------------------------------------------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_path.exists() or output_path.stat().st_size == 0

    output_fields = [
        "node_id", "abstract", "summary", "paper_id", "scores", "neighbours",
        "chosen_persona", "rejected_persona", "rejected_score",
    ]

    # initial_node_ids grows as we accept nodes; used to exclude them from
    # each other's neighbourhoods.
    initial_node_ids: set[str] = set(already_done)

    accepted = 0
    discarded_neighbourhood = 0
    discarded_dpo = 0  # shouldn't happen after pre-filter, but safety net

    driver = get_driver()
    try:
        with output_path.open("a", encoding="utf-8", newline="") as out_f:
            writer = csv.DictWriter(out_f, fieldnames=output_fields)
            if write_header:
                writer.writeheader()

            with tqdm(total=n_still_needed, desc="Accepted nodes", unit="node") as pbar:
                for node_id in candidate_pool:
                    if accepted >= n_still_needed:
                        break

                    meta = metadata[node_id]
                    chosen, rej3, rej2 = select_dpo_personas(meta["scores"])

                    # Safety net (should have been filtered already)
                    if chosen is None or rej3 is None:
                        discarded_dpo += 1
                        continue

                    # Tentatively add to initial set before fetching
                    # so the neighbourhood query excludes it too.
                    initial_node_ids.add(node_id)

                    neighbours = fetch_neighbourhood(
                        driver,
                        node_id=node_id,
                        depth=args.depth,
                        n_neighbours=args.n_neighbours,
                        initial_node_ids=initial_node_ids,
                        max_depth=args.max_depth,
                    )

                    if neighbours is None:
                        # Cannot satisfy neighbourhood requirement; remove from set.
                        initial_node_ids.discard(node_id)
                        discarded_neighbourhood += 1
                        continue

                    # Write row 1: chosen (score=5) vs rejected (score=3)
                    base = {
                        "node_id": node_id,
                        "abstract": meta["abstract"],
                        "summary": meta["summary"],
                        "paper_id": meta["paper_id"],
                        "scores": meta["scores"],
                        "neighbours": json.dumps(neighbours),
                        "chosen_persona": chosen,
                    }
                    writer.writerow({**base, "rejected_persona": rej3, "rejected_score": 3})

                    # Write row 2: chosen (score=5) vs rejected (score=2), if available
                    if rej2 is not None:
                        writer.writerow({**base, "rejected_persona": rej2, "rejected_score": 2})

                    out_f.flush()
                    accepted += 1
                    pbar.update(1)

        print(f"\n  Accepted : {accepted:,} nodes ({accepted + len(already_done):,} total in output)")
        if discarded_neighbourhood:
            print(f"  Discarded (neighbourhood): {discarded_neighbourhood:,}")
        if discarded_dpo:
            print(f"  Discarded (DPO scores):    {discarded_dpo:,}")
        if accepted < n_still_needed:
            print(
                f"  Warning: pool exhausted after {accepted:,} new nodes "
                f"(target was {n_still_needed:,} more)."
            )

    finally:
        driver.close()
        print("Neo4j connection closed.")

    print(f"\nDone. Output written to: {output_path}")


if __name__ == "__main__":
    main()
