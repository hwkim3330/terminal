#!/usr/bin/env python3
"""Download Cosmos models to local storage."""

from huggingface_hub import snapshot_download
import os

DOWNLOAD_DIR = "/mnt/data/lfm_agi/models"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

models = [
    "nvidia/Cosmos-Reason2-8B",  # Physical AI reasoning
    "nvidia/Cosmos-Transfer1-7B",  # Video transfer
    "nvidia/Cosmos-1.0-Tokenizer-DV8x16x16",  # Video tokenizer
]

for model_id in models:
    print(f"\n{'='*50}")
    print(f"Downloading: {model_id}")
    print(f"{'='*50}")
    try:
        local_dir = os.path.join(DOWNLOAD_DIR, model_id.replace("/", "--"))
        snapshot_download(
            repo_id=model_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
        )
        print(f"✓ Saved to: {local_dir}")
    except Exception as e:
        print(f"✗ Error: {e}")

print("\nDone!")
