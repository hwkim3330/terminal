#!/usr/bin/env python3
"""
Download essential PhysicalAI-AV dataset components.
Focus on metadata, calibration, and sample camera data.
"""

from huggingface_hub import snapshot_download, hf_hub_download, HfApi
import os

REPO_ID = "nvidia/PhysicalAI-Autonomous-Vehicles"
LOCAL_DIR = "/mnt/data/lfm_agi/datasets/PhysicalAI-AV"
os.makedirs(LOCAL_DIR, exist_ok=True)

api = HfApi()

print("=" * 60)
print("PhysicalAI-Autonomous-Vehicles Dataset Download")
print("=" * 60)

# 1. Download metadata files (small, essential)
print("\n[1/3] Downloading metadata...")
metadata_patterns = [
    "*.parquet",  # Clip index, sensor presence
    "*.json",     # Config files
    "*.md",       # Documentation
    "README*",
]

try:
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=LOCAL_DIR,
        allow_patterns=metadata_patterns,
        ignore_patterns=["cameras/*", "lidar/*", "radar/*"],
    )
    print("  ✓ Metadata downloaded")
except Exception as e:
    print(f"  ✗ Error: {e}")

# 2. Download calibration data
print("\n[2/3] Downloading calibration data...")
try:
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=LOCAL_DIR,
        allow_patterns=["calibration/*", "calib/*"],
    )
    print("  ✓ Calibration downloaded")
except Exception as e:
    print(f"  ✗ Error: {e}")

# 3. Download sample camera chunks (first 5 chunks only)
print("\n[3/3] Downloading sample camera data (5 chunks)...")
try:
    # List available camera files
    files = api.list_repo_files(REPO_ID, repo_type="dataset")
    camera_files = sorted([f for f in files if f.startswith("cameras/") and f.endswith(".zip")])[:5]

    for f in camera_files:
        print(f"  Downloading {f}...")
        hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=f,
            local_dir=LOCAL_DIR,
        )
    print(f"  ✓ Downloaded {len(camera_files)} camera chunks")
except Exception as e:
    print(f"  ✗ Error: {e}")

# Summary
print("\n" + "=" * 60)
print("Download complete!")
print(f"Location: {LOCAL_DIR}")

import subprocess
result = subprocess.run(["du", "-sh", LOCAL_DIR], capture_output=True, text=True)
print(f"Size: {result.stdout.strip()}")
print("=" * 60)
