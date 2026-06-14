#!/usr/bin/env python3
"""Stratify a DPO neighbourhood dataset by chosen_persona and (chosen, rejected) pair.

Reads the CSV produced by build_neighbourhood_dataset.py and outputs a balanced
subset where:
  1. Every (chosen_persona, rejected_persona) pair has the same number of rows.
  2. As a consequence, every chosen_persona also has the same total row count
     (since each chosen persona is paired with the same number of rejected partners,
     each at equal count).

Strategy
--------
  - Group rows by (chosen_persona, rejected_persona).
  - Find min_count = minimum group size across all observed pairs.
  - Optionally cap min_count with --max-per-pair to limit output size.
  - Sample exactly min_count rows from each group (reproducibly via --seed).
  - Warn about any (chosen, rejected) pairs that are absent from the data.

Usage example
-------------
  python scripts_2/stratify_dpo_dataset.py \\
      --input  data/dpo_neighbourhood_dataset.csv \\
      --output data/dpo_neighbourhood_balanced.csv \\
      --max-per-pair 1000 \\
      --seed 42
"""

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_rows(input_csv: Path) -> list[dict]:
    with input_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows


def write_rows(rows: list[dict], output_csv: Path, fieldnames: list[str]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Stratification
# ---------------------------------------------------------------------------


def stratify(
    rows: list[dict],
    max_per_pair: int | None,
    seed: int,
) -> tuple[list[dict], dict]:
    """Return (stratified_rows, stats_dict)."""
    rng = random.Random(seed)

    # Group by (chosen_persona, rejected_persona)
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    skipped_no_persona = 0
    for row in rows:
        chosen = row.get("chosen_persona", "").strip()
        rejected = row.get("rejected_persona", "").strip()
        if not chosen or not rejected:
            skipped_no_persona += 1
            continue
        groups[(chosen, rejected)].append(row)

    if not groups:
        return [], {"skipped_no_persona": skipped_no_persona}

    # Unique chosen / rejected personas
    all_chosen = sorted({k[0] for k in groups})
    all_rejected = sorted({k[1] for k in groups})

    # Warn about missing pairs
    missing_pairs = []
    for c in all_chosen:
        for r in all_rejected:
            if c != r and (c, r) not in groups:
                missing_pairs.append((c, r))

    # Determine per-pair sample count
    group_sizes = {pair: len(g) for pair, g in groups.items()}
    min_count = min(group_sizes.values())
    per_pair = min_count if max_per_pair is None else min(min_count, max_per_pair)

    # Sample
    result: list[dict] = []
    for pair in sorted(groups):
        sample = rng.sample(groups[pair], per_pair)
        result.append(sample)

    # Shuffle final list so rows from same pair are not contiguous
    flat: list[dict] = [row for group in result for row in group]
    rng.shuffle(flat)

    stats = {
        "total_input_rows": len(rows),
        "skipped_no_persona": skipped_no_persona,
        "unique_chosen_personas": len(all_chosen),
        "unique_rejected_personas": len(all_rejected),
        "observed_pairs": len(groups),
        "missing_pairs": missing_pairs,
        "min_group_size": min_count,
        "per_pair_count": per_pair,
        "total_output_rows": len(flat),
        "group_sizes_before": group_sizes,
    }
    return flat, stats


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_stats(stats: dict) -> None:
    print(f"\n{'='*60}")
    print(f"  Input rows          : {stats['total_input_rows']:,}")
    print(f"  Skipped (no persona): {stats['skipped_no_persona']:,}")
    print(f"  Chosen personas     : {stats['unique_chosen_personas']}")
    print(f"  Rejected personas   : {stats['unique_rejected_personas']}")
    print(f"  Observed pairs      : {stats['observed_pairs']}")
    if stats["missing_pairs"]:
        print(f"  Missing pairs ({len(stats['missing_pairs'])}):")
        for c, r in stats["missing_pairs"]:
            print(f"    {c}  ->  {r}")
    print(f"\n  Min group size      : {stats['min_group_size']:,}")
    print(f"  Samples per pair    : {stats['per_pair_count']:,}")
    print(f"  Total output rows   : {stats['total_output_rows']:,}")

    print(f"\n  Group sizes (before stratification):")
    for pair, size in sorted(stats["group_sizes_before"].items()):
        chosen, rejected = pair
        print(f"    {chosen:30s} -> {rejected:30s} : {size:,}")
    print("="*60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stratify DPO dataset to balance chosen and (chosen, rejected) pairs."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/dpo_neighbourhood_dataset.csv",
        help="Input CSV (output of build_neighbourhood_dataset.py). "
             "Default: data/dpo_neighbourhood_dataset.csv",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/dpo_neighbourhood_balanced.csv",
        help="Output balanced CSV path. Default: data/dpo_neighbourhood_balanced.csv",
    )
    parser.add_argument(
        "--max-per-pair",
        type=int,
        default=None,
        help="Optional hard cap on samples per (chosen, rejected) pair. "
             "If omitted, uses the natural minimum group size.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling. Default: 42",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    print(f"Loading {input_path} ...")
    rows = load_rows(input_path)
    print(f"  Loaded {len(rows):,} rows.")

    if not rows:
        print("Input file is empty. Nothing to do.")
        return

    fieldnames = list(rows[0].keys())

    balanced, stats = stratify(rows, args.max_per_pair, args.seed)
    print_stats(stats)

    if not balanced:
        print("\nNo rows to write after stratification.")
        return

    write_rows(balanced, output_path, fieldnames)
    print(f"\nDone. Balanced dataset written to: {output_path}")


if __name__ == "__main__":
    main()
