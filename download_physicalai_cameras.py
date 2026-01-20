#!/usr/bin/env python3
"""
Download PhysicalAI-AV camera data for Alpamayo.
Focus on front cameras which are most useful for driving demos.
Target: ~800GB of camera data.
"""

from huggingface_hub import hf_hub_download, HfApi
import os
from pathlib import Path
import sys

# Unbuffered output
sys.stdout.reconfigure(line_buffering=True)

REPO_ID = "nvidia/PhysicalAI-Autonomous-Vehicles"
LOCAL_DIR = Path("/mnt/data/lfm_agi/datasets/nurec")

# Camera types to download (prioritized)
CAMERA_TYPES = [
    "camera_front_wide_120fov",   # Main camera for Alpamayo
    "camera_cross_left_120fov",   # Side view
    "camera_cross_right_120fov",  # Side view
    "camera_front_tele_30fov",    # Telephoto
]

# Download chunks 0-130 for each camera (~130 chunks * 4 cameras * ~1.8GB = ~936GB)
START_CHUNK = 0
END_CHUNK = 130

def main():
    api = HfApi()

    print("=" * 60)
    print("PhysicalAI-AV Camera Download")
    print(f"Chunks {START_CHUNK}-{END_CHUNK} for {len(CAMERA_TYPES)} cameras")
    print("=" * 60)

    # List all files
    print("\nListing repository files...")
    files = api.list_repo_files(REPO_ID, repo_type="dataset")

    for cam_type in CAMERA_TYPES:
        print(f"\n{'='*50}")
        print(f"Downloading: {cam_type}")
        print(f"{'='*50}")

        # Find matching chunks
        pattern = f"camera/{cam_type}/{cam_type}.chunk_"
        matching = sorted([f for f in files if f.startswith(pattern) and f.endswith(".zip")])

        # Select range
        selected = []
        for f in matching:
            try:
                chunk_num = int(f.split("chunk_")[1].split(".")[0])
                if START_CHUNK <= chunk_num <= END_CHUNK:
                    selected.append((chunk_num, f))
            except:
                continue

        selected.sort()
        print(f"Found {len(selected)} chunks to download")

        for i, (chunk_num, filename) in enumerate(selected):
            local_path = LOCAL_DIR / filename

            # Skip if exists
            if local_path.exists():
                print(f"  [{i+1}/{len(selected)}] Chunk {chunk_num} exists, skipping")
                continue

            print(f"  [{i+1}/{len(selected)}] Downloading chunk {chunk_num}...")
            try:
                hf_hub_download(
                    repo_id=REPO_ID,
                    repo_type="dataset",
                    filename=filename,
                    local_dir=str(LOCAL_DIR),
                )
                print(f"    ✓ Done")
            except KeyboardInterrupt:
                print("\n\nDownload interrupted by user")
                sys.exit(0)
            except Exception as e:
                print(f"    ✗ Error: {e}")
                continue

    print("\n" + "=" * 60)
    print("Download complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
