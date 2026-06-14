#!/usr/bin/env python3
"""Sample Neo4j nodes and references into lightweight CSV files.

This script connects to Neo4j, samples a given number of nodes, and saves:
- sampled node ids only
- references with source/target node ids and relationship type

It also supports streaming verification of output CSV files without loading them fully.
"""

import argparse
import csv
import os
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase


def load_environment(env_file: str) -> None:
    if not os.path.exists(env_file):
        raise FileNotFoundError(f"Environment file not found: {env_file}")
    load_dotenv(env_file)


def get_driver() -> GraphDatabase:
    vps_ip = os.getenv("VPS_IP")
    user = os.getenv("NEO4J_USER")
    password = os.getenv("NEO4J_PASSWORD")

    if not all([vps_ip, user, password]):
        raise EnvironmentError(
            "Missing Neo4j connection variables. Ensure VPS_IP, NEO4J_USER, and NEO4J_PASSWORD are set."
        )

    uri = f"bolt://{vps_ip}:7687"
    return GraphDatabase.driver(uri, auth=(user, password))


def load_existing_node_ids(file_path: Path) -> list[str]:
    if not file_path.exists():
        return []

    with file_path.open("r", encoding="utf-8", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        node_ids: list[str] = []
        seen: set[str] = set()

        for row in reader:
            node_id = row.get("node_id")
            if node_id and node_id not in seen:
                seen.add(node_id)
                node_ids.append(node_id)

    return node_ids


def sample_node_ids(
    driver: GraphDatabase,
    sample_size: int,
    excluded_node_ids: list[str],
) -> list[str]:
    """
    Sample random nodes that have at least one relationship.

    Uses MATCH (n) instead of MATCH (n)-[r]-(m) to avoid expanding
    all edges before deduplication — the relationship expansion multiplies
    matched rows by node degree, making ORDER BY rand() on large graphs
    extremely expensive. Connectivity is checked with an existence predicate.
    """
    with driver.session() as session:
        result = session.run(
            """
            MATCH (n)
            WHERE NOT elementId(n) IN $excluded_ids
              AND (n)--()
            RETURN elementId(n) AS node_id
            ORDER BY rand()
            LIMIT $limit
            """,
            limit=sample_size,
            excluded_ids=excluded_node_ids,
        )
        return [record["node_id"] for record in result]


def collect_references(
    driver: GraphDatabase,
    node_ids: list[str],
) -> list[dict[str, str]]:
    """
    Collect all relationships touching the sampled nodes.

    Returns each physical relationship exactly once using startNode/endNode
    with DISTINCT. The original undirected MATCH (n)-[r]-(m) WHERE n IN
    node_ids returns the same edge twice when both endpoints are sampled,
    which causes duplicate entries in relationships_dict and downstream
    duplicate DPO pairs.

    source_node_id is always the relationship's startNode (outgoing end),
    target_node_id is always the endNode (incoming end), consistent with
    how form_dpo_pairs.py interprets these columns.
    """
    with driver.session() as session:
        result = session.run(
            """
            MATCH (n)-[r]-(m)
            WHERE elementId(n) IN $node_ids
            RETURN DISTINCT
                elementId(startNode(r)) AS source_node_id,
                type(r)                 AS relationship_type,
                elementId(endNode(r))   AS target_node_id
            """,
            node_ids=node_ids,
        )
        return [
            {
                "source_node_id": record["source_node_id"],
                "relationship_type": record["relationship_type"],
                "target_node_id": record["target_node_id"],
            }
            for record in result
        ]


def append_nodes_csv(node_ids: list[str], file_path: Path) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    has_content = file_path.exists() and file_path.stat().st_size > 0
    with file_path.open("a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        if not has_content:
            writer.writerow(["node_id"])
        for node_id in node_ids:
            writer.writerow([node_id])


def append_references_csv(references: list[dict[str, str]], file_path: Path) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    has_content = file_path.exists() and file_path.stat().st_size > 0
    with file_path.open("a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=["source_node_id", "relationship_type", "target_node_id", "direction"],
        )
        if not has_content:
            writer.writeheader()
        writer.writerows(references)


def verify_csv_file(file_path: Path) -> int:
    if not file_path.exists():
        raise FileNotFoundError(f"CSV file not found: {file_path}")

    with file_path.open("r", encoding="utf-8", newline="") as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)
        return sum(1 for _ in reader)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample Neo4j nodes and references to CSV.")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=3000,
        help="Number of nodes to sample from Neo4j.",
    )
    parser.add_argument(
        "--nodes-output",
        type=str,
        default="data/neo4j_sample_nodes.csv",
        help="Output CSV path for sampled node ids.",
    )
    parser.add_argument(
        "--references-output",
        type=str,
        default="data/neo4j_sample_references.csv",
        help="Output CSV path for sampled references.",
    )
    parser.add_argument(
        "--env-file",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / ".env_db"),
        help="Path to the .env file containing Neo4j credentials.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify written CSV files by scanning them without loading full content.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_environment(args.env_file)

    nodes_output_path = Path(args.nodes_output)
    references_output_path = Path(args.references_output)

    existing_node_ids = load_existing_node_ids(nodes_output_path)
    print(f"Found {len(existing_node_ids)} existing sampled node ids in {nodes_output_path}.")

    driver = get_driver()
    try:
        print(
            f"Sampling up to {args.sample_size} new node ids from Neo4j (excluding existing ids)..."
        )
        new_node_ids = sample_node_ids(driver, args.sample_size, existing_node_ids)
        total_node_count = len(existing_node_ids) + len(new_node_ids)

        print(f"Collected {len(new_node_ids)} new sampled node ids.")
        print(f"Total unique sampled node ids now: {total_node_count}")
        references = collect_references(driver, new_node_ids)
        print(f"Collected {len(references)} references for newly sampled nodes.")

        append_nodes_csv(new_node_ids, nodes_output_path)
        append_references_csv(references, references_output_path)

        print(f"Appended sampled node ids to: {nodes_output_path}")
        print(f"Appended sampled references to: {references_output_path}")

        if args.verify:
            node_count = verify_csv_file(nodes_output_path)
            ref_count = verify_csv_file(references_output_path)
            print(f"Verified nodes CSV row count: {node_count}")
            print(f"Verified references CSV row count: {ref_count}")
    finally:
        driver.close()
        print("Neo4j connection closed.")


if __name__ == "__main__":
    main()
