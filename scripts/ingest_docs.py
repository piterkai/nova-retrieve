"""Ingest one or more files/directories into the vector store.

Usage:
    python -m scripts.ingest_docs ./data/docs
    python -m scripts.ingest_docs ./a.pdf ./b/ --collection mykb
"""
import argparse
import sys

from app.core.logging import setup_logging
from app.ingest.pipeline import ingest_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help="Files or directories to ingest")
    parser.add_argument("--collection", default=None)
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--chunk-overlap", type=int, default=120)
    args = parser.parse_args()

    setup_logging()
    result = ingest_paths(
        args.paths,
        collection=args.collection,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    print(f"Ingest finished: {result}")


if __name__ == "__main__":
    sys.exit(main())
