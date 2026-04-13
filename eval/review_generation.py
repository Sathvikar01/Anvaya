"""Interactively score blinded generation answers and resume safely."""

from __future__ import annotations

import argparse
from pathlib import Path

from hh_goa_rag.generation.review import review_judgments


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--judgments",
        type=Path,
        default=Path("results/generation_blinded_judgments.csv"),
    )
    args = parser.parse_args()
    try:
        completed, total = review_judgments(args.judgments)
    except (EOFError, KeyboardInterrupt):
        print("\nStopped. Every entered score was already saved.")
        return
    print(f"Review complete: {completed}/{total} judgments scored.")


if __name__ == "__main__":
    main()
