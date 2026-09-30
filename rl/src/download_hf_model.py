#!/usr/bin/env python3
"""Download a complete Hugging Face model repository into this project."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download a Hugging Face model repository to a local directory."
    )
    parser.add_argument("repo_id", help="Hugging Face repository name, e.g. Qwen/Qwen3-8B")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Optional Hugging Face cache root; default follows HF_HOME/HF_HUB_CACHE",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Branch, tag, or commit to download (default: main)",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Optional Hugging Face token",
    )
    args = parser.parse_args()

    token = args.token
    cache_dir = args.cache_dir.expanduser().resolve() if args.cache_dir else None

    print(f"Repository:  {args.repo_id}")
    print(f"Revision:    {args.revision}")
    print(f"Cache root:  {cache_dir or 'Hugging Face default cache'}")
    snapshot_path = snapshot_download(
        repo_id=args.repo_id,
        repo_type="model",
        revision=args.revision,
        cache_dir=cache_dir,
        token=token,
        resume_download=True,
    )
    print(f"Model snapshot: {snapshot_path}")


if __name__ == "__main__":
    main()
