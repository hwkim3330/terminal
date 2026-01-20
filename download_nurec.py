#!/usr/bin/env python3
"""Download NuRec dataset chunks to local storage."""

import physical_ai_av
import os

# Set local cache directory
LOCAL_DIR = "/mnt/data/lfm_agi/datasets/nurec"
os.makedirs(LOCAL_DIR, exist_ok=True)

print("Initializing dataset interface...")
avdi = physical_ai_av.PhysicalAIAVDatasetInterface(
    local_dir=LOCAL_DIR,
    confirm_download_threshold_gb=1000,  # Auto-confirm up to 1TB
)

print(f"Local dir: {LOCAL_DIR}")
print(f"Total chunks: {avdi.chunk_sensor_presence.shape[0]}")

# Download first 100 chunks (for now)
NUM_CHUNKS = 100
print(f"\nDownloading first {NUM_CHUNKS} chunks...")

for chunk_idx in range(NUM_CHUNKS):
    try:
        print(f"[{chunk_idx+1}/{NUM_CHUNKS}] Downloading chunk {chunk_idx}...")
        avdi.download_chunk_features(chunk_idx)
    except Exception as e:
        print(f"  Error: {e}")
        continue

print("\nDone!")
print(f"Check {LOCAL_DIR} for downloaded data")
