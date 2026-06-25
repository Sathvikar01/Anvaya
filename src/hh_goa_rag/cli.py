"""Command line interface for phase-wise retrieval experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

from hh_goa_rag.chunking_ablation import run_chunking_ablation
from hh_goa_rag.config import load_config
from hh_goa_rag.dataset import download_full_dataset, prepare_dataset, read_manifest
from hh_goa_rag.embedding_ablation import run_embedding_ablation
from hh_goa_rag.finalization import run_finalization
from hh_goa_rag.index_ablation import run_index_ablation
from hh_goa_rag.small_ablation import run_small_ablation


def _dataset_prepare(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    output_dir = prepare_dataset(config, force=args.force)
    manifest = read_manifest(output_dir / "manifest.json")
    table = Table(title="MSMARCO-XI prepared evaluation artifacts")
    table.add_column("Split")
    table.add_column("Source")
    table.add_column("Queries", justify="right")
    table.add_column("Parent passages", justify="right")
    table.add_column("Qrels", justify="right")
    for split, stats in manifest["splits"].items():
        table.add_row(
            split,
            stats["source_split"],
            str(stats["queries"]),
            str(stats["unique_parent_passages"]),
            str(stats["qrels"]),
        )
    console = Console()
    console.print(table)
    console.print(f"Artifacts: [bold]{output_dir}[/bold]")
    if args.json:
        print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


def _dataset_download_full(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    dataset_config = config["dataset"]
    manifest = download_full_dataset(
        dataset_config["repository"],
        config["cache"]["huggingface"],
        revision=dataset_config.get("revision"),
        max_workers=int(dataset_config.get("download_workers", 8)),
        force=args.force,
    )
    print(f"Downloaded and verified {len(manifest.parquet_files)} parquet files")
    print(f"Revision: {manifest.revision}")
    print(f"Bytes: {manifest.total_bytes}")
    print(f"Root: {manifest.root}")
    return 0


def _embedding_run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    _, winner = run_embedding_ablation(config, data_dir=args.data_dir)
    print(f"\nEmbedding winner: **{winner['winner']}**")
    return 0


def _chunking_run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    _, winner = run_chunking_ablation(config, data_dir=args.data_dir)
    print(f"\nChunking winner: **{winner['winner']}**")
    return 0


def _index_run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    _, winner = run_index_ablation(config, data_dir=args.data_dir)
    print(f"\nIndex/storage winner: **{winner['winner']}**")
    return 0


def _small_ablation_run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    summary = run_small_ablation(config, data_dir=args.data_dir, index_only=args.index_only)
    embedding = summary.get("embedding_winner", summary.get("embedding_model", "n/a"))
    print(f"\nSmall embedding: **{embedding}**")
    print(f"Index winner: **{summary['index_winner']}**")
    return 0


def _finalize_run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    result = run_finalization(config, data_dir=args.data_dir)
    print(f"\nSealed test nDCG@10: **{result['ndcg_at_10']:.4f}**")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.yaml", type=Path)
    subparsers = parser.add_subparsers(dest="command", required=True)
    dataset_parser = subparsers.add_parser("dataset", help="Dataset operations")
    dataset_subparsers = dataset_parser.add_subparsers(dest="dataset_command", required=True)
    prepare_parser = dataset_subparsers.add_parser("prepare", help="Prepare fixed eval artifacts")
    prepare_parser.add_argument("--force", action="store_true")
    prepare_parser.add_argument("--json", action="store_true")
    prepare_parser.set_defaults(handler=_dataset_prepare)
    download_parser = dataset_subparsers.add_parser(
        "download-full", help="Download and verify every dataset parquet"
    )
    download_parser.add_argument("--force", action="store_true")
    download_parser.set_defaults(handler=_dataset_download_full)
    embedding_parser = subparsers.add_parser("embedding", help="Embedding-model ablation")
    embedding_subparsers = embedding_parser.add_subparsers(
        dest="embedding_command", required=True
    )
    embedding_run = embedding_subparsers.add_parser("run", help="Run/resume all model candidates")
    embedding_run.add_argument("--data-dir", type=Path)
    embedding_run.set_defaults(handler=_embedding_run)
    chunking_parser = subparsers.add_parser("chunking", help="Chunking-strategy ablation")
    chunking_subparsers = chunking_parser.add_subparsers(
        dest="chunking_command", required=True
    )
    chunking_run = chunking_subparsers.add_parser("run", help="Run/resume all chunk candidates")
    chunking_run.add_argument("--data-dir", type=Path)
    chunking_run.set_defaults(handler=_chunking_run)
    index_parser = subparsers.add_parser("index", help="Index and local-storage ablation")
    index_subparsers = index_parser.add_subparsers(dest="index_command", required=True)
    index_run = index_subparsers.add_parser("run", help="Run/resume all index backends")
    index_run.add_argument("--data-dir", type=Path)
    index_run.set_defaults(handler=_index_run)
    small_parser = subparsers.add_parser(
        "small-ablation", help="Small embedding-model and index ablation"
    )
    small_subparsers = small_parser.add_subparsers(dest="small_command", required=True)
    small_run = small_subparsers.add_parser("run", help="Run/resume the small-model study")
    small_run.add_argument("--data-dir", type=Path)
    small_run.add_argument(
        "--index-only",
        action="store_true",
        help="Reuse embedding vectors and run only the index comparison",
    )
    small_run.set_defaults(handler=_small_ablation_run)
    finalize_parser = subparsers.add_parser("finalize", help="Sealed test and final handoff")
    finalize_subparsers = finalize_parser.add_subparsers(
        dest="finalize_command", required=True
    )
    finalize_run = finalize_subparsers.add_parser(
        "run", help="Evaluate once, recommend, and clean losing project models"
    )
    finalize_run.add_argument("--data-dir", type=Path)
    finalize_run.set_defaults(handler=_finalize_run)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
