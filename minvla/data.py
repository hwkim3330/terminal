"""
Data Collection and Loading for MiniVLA
========================================

This module handles:
1. Recording human demonstrations (screenshot + action pairs)
2. Loading and preprocessing training data
3. Data augmentation

The key insight: you need REAL data of humans using computers.
No amount of clever architecture can substitute for good data.

Data format (JSON Lines):
    {"screenshot": "path/to/img.png", "instruction": "click search", "action": {...}}

Usage:
    # Record demonstrations
    python data.py --record --output demos/

    # Create dataset
    python data.py --create-dataset --input demos/ --output dataset/

Author: Karpathy-style refactor
"""

import os
import json
import time
import threading
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple
import numpy as np

try:
    import mss
    import pyautogui
    from pynput import mouse, keyboard
    HAS_RECORDING_DEPS = True
except ImportError:
    HAS_RECORDING_DEPS = False

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image


# -----------------------------------------------------------------------------
# Data Structures
# -----------------------------------------------------------------------------

@dataclass
class Action:
    """Represents a computer action."""
    action_type: str  # click, double_click, right_click, drag, scroll, type, hotkey
    x: int
    y: int
    end_x: Optional[int] = None  # for drag
    end_y: Optional[int] = None
    text: Optional[str] = None   # for type action
    key: Optional[str] = None    # for hotkey


@dataclass
class Sample:
    """A single training sample."""
    screenshot_path: str
    instruction: str
    action: Action
    timestamp: float
    screen_size: Tuple[int, int]


# -----------------------------------------------------------------------------
# Data Recorder
# -----------------------------------------------------------------------------

class DemoRecorder:
    """
    Records human demonstrations.

    Captures:
    - Screenshots at each action
    - Mouse clicks and positions
    - Keyboard input
    - User-provided instructions

    Usage:
        recorder = DemoRecorder("demos/")
        recorder.start()
        # ... human performs actions ...
        recorder.stop()
    """

    def __init__(self, output_dir: str, capture_interval: float = 0.1):
        if not HAS_RECORDING_DEPS:
            raise ImportError("Recording requires: pip install mss pyautogui pynput")

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.capture_interval = capture_interval
        self.samples: List[Sample] = []
        self.current_instruction: str = ""
        self.recording = False
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Screen capture
        self.sct = mss.mss()
        self.screen_size = (
            self.sct.monitors[1]["width"],
            self.sct.monitors[1]["height"]
        )

        # Listeners
        self.mouse_listener = None
        self.key_listener = None

        # State
        self.last_screenshot = None
        self.pending_action = None

    def set_instruction(self, instruction: str):
        """Set the current instruction being demonstrated."""
        self.current_instruction = instruction
        print(f"[Recorder] Instruction: {instruction}")

    def capture_screenshot(self) -> str:
        """Capture and save screenshot."""
        timestamp = time.time()
        filename = f"{self.session_id}_{timestamp:.3f}.png"
        filepath = self.output_dir / "screenshots" / filename
        filepath.parent.mkdir(exist_ok=True)

        # Capture
        monitor = self.sct.monitors[1]
        screenshot = self.sct.grab(monitor)

        # Save
        img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
        img.save(filepath)

        return str(filepath)

    def _on_click(self, x, y, button, pressed):
        """Handle mouse click."""
        if not self.recording or not pressed:
            return

        if not self.current_instruction:
            print("[Recorder] Warning: No instruction set!")
            return

        # Capture screenshot before action
        screenshot_path = self.capture_screenshot()

        # Determine action type
        if button == mouse.Button.left:
            action_type = "click"
        elif button == mouse.Button.right:
            action_type = "right_click"
        else:
            action_type = "click"

        action = Action(
            action_type=action_type,
            x=x,
            y=y
        )

        sample = Sample(
            screenshot_path=screenshot_path,
            instruction=self.current_instruction,
            action=action,
            timestamp=time.time(),
            screen_size=self.screen_size
        )

        self.samples.append(sample)
        print(f"[Recorder] Recorded: {action_type} at ({x}, {y})")

    def _on_scroll(self, x, y, dx, dy):
        """Handle scroll."""
        if not self.recording:
            return

        if not self.current_instruction:
            return

        screenshot_path = self.capture_screenshot()

        action = Action(
            action_type="scroll",
            x=x,
            y=y,
            text=f"{dx},{dy}"
        )

        sample = Sample(
            screenshot_path=screenshot_path,
            instruction=self.current_instruction,
            action=action,
            timestamp=time.time(),
            screen_size=self.screen_size
        )

        self.samples.append(sample)
        print(f"[Recorder] Recorded: scroll at ({x}, {y})")

    def _on_key(self, key):
        """Handle key press."""
        if not self.recording:
            return

        # Could record hotkeys here
        pass

    def start(self):
        """Start recording."""
        self.recording = True

        # Start listeners
        self.mouse_listener = mouse.Listener(
            on_click=self._on_click,
            on_scroll=self._on_scroll
        )
        self.key_listener = keyboard.Listener(on_press=self._on_key)

        self.mouse_listener.start()
        self.key_listener.start()

        print(f"[Recorder] Started recording to {self.output_dir}")
        print("[Recorder] Set instruction with: recorder.set_instruction('...')")
        print("[Recorder] Stop with: recorder.stop()")

    def stop(self):
        """Stop recording and save."""
        self.recording = False

        if self.mouse_listener:
            self.mouse_listener.stop()
        if self.key_listener:
            self.key_listener.stop()

        # Save samples
        self._save_samples()
        print(f"[Recorder] Stopped. Recorded {len(self.samples)} samples.")

    def _save_samples(self):
        """Save samples to JSONL file."""
        output_file = self.output_dir / f"samples_{self.session_id}.jsonl"

        with open(output_file, 'w') as f:
            for sample in self.samples:
                data = {
                    'screenshot_path': sample.screenshot_path,
                    'instruction': sample.instruction,
                    'action': asdict(sample.action),
                    'timestamp': sample.timestamp,
                    'screen_size': sample.screen_size
                }
                f.write(json.dumps(data) + '\n')

        print(f"[Recorder] Saved to {output_file}")


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------

class VLADataset(Dataset):
    """
    PyTorch Dataset for VLA training.

    Loads screenshot-instruction-action triplets.
    """

    def __init__(
        self,
        data_dir: str,
        tokenizer,
        image_size: int = 224,
        max_seq_len: int = 128,
        augment: bool = True
    ):
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.max_seq_len = max_seq_len
        self.augment = augment

        # Load all samples
        self.samples = []
        for jsonl_file in self.data_dir.glob("*.jsonl"):
            with open(jsonl_file) as f:
                for line in f:
                    self.samples.append(json.loads(line))

        print(f"[Dataset] Loaded {len(self.samples)} samples from {data_dir}")

        # Action type mapping
        self.action_types = ['click', 'double_click', 'right_click', 'drag', 'scroll', 'type', 'hotkey']
        self.action_to_idx = {a: i for i, a in enumerate(self.action_types)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load and preprocess image
        img = Image.open(sample['screenshot_path']).convert('RGB')
        orig_size = img.size
        img = img.resize((self.image_size, self.image_size))
        img = np.array(img) / 255.0

        # Augmentation
        if self.augment:
            img = self._augment_image(img)

        img = torch.from_numpy(img).float().permute(2, 0, 1)  # HWC -> CHW

        # Tokenize instruction
        tokens = self.tokenizer.encode(sample['instruction'])
        tokens = tokens[:self.max_seq_len]
        tokens = tokens + [0] * (self.max_seq_len - len(tokens))  # pad
        tokens = torch.tensor(tokens, dtype=torch.long)

        # Parse action
        action = sample['action']
        screen_w, screen_h = sample['screen_size']

        # Normalize coordinates to [0, 1]
        x = action['x'] / screen_w
        y = action['y'] / screen_h
        end_x = (action.get('end_x') or action['x']) / screen_w
        end_y = (action.get('end_y') or action['y']) / screen_h

        coords = torch.tensor([x, y, end_x, end_y], dtype=torch.float)
        action_type = self.action_to_idx.get(action['action_type'], 0)

        return {
            'image': img,
            'tokens': tokens,
            'coords': coords,
            'action_type': action_type
        }

    def _augment_image(self, img: np.ndarray) -> np.ndarray:
        """Simple augmentation: brightness/contrast."""
        if np.random.random() < 0.5:
            # Brightness
            factor = np.random.uniform(0.8, 1.2)
            img = np.clip(img * factor, 0, 1)

        if np.random.random() < 0.3:
            # Add noise
            noise = np.random.normal(0, 0.02, img.shape)
            img = np.clip(img + noise, 0, 1)

        return img


def create_dataloader(
    data_dir: str,
    tokenizer,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    **kwargs
) -> DataLoader:
    """Create a DataLoader for training."""
    dataset = VLADataset(data_dir, tokenizer, **kwargs)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )


# -----------------------------------------------------------------------------
# Synthetic Data (for testing without real demos)
# -----------------------------------------------------------------------------

class SyntheticDataset(Dataset):
    """
    Generate synthetic training data for testing.

    Creates random screenshots with colored rectangles as "UI elements"
    and generates click actions on them.
    """

    def __init__(
        self,
        num_samples: int = 10000,
        image_size: int = 224,
        max_seq_len: int = 128,
        vocab_size: int = 50257
    ):
        self.num_samples = num_samples
        self.image_size = image_size
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size

        # Pre-generate all samples for consistency
        self.data = []
        for i in range(num_samples):
            self.data.append(self._generate_sample())

        print(f"[SyntheticDataset] Generated {num_samples} synthetic samples")

    def _generate_sample(self):
        """Generate a single synthetic sample."""
        # Create blank image
        img = np.ones((self.image_size, self.image_size, 3)) * 0.9  # light gray

        # Add random "UI elements" (colored rectangles)
        num_elements = np.random.randint(3, 8)
        elements = []

        for _ in range(num_elements):
            x1 = np.random.randint(0, self.image_size - 30)
            y1 = np.random.randint(0, self.image_size - 20)
            w = np.random.randint(20, 60)
            h = np.random.randint(15, 30)
            color = np.random.rand(3) * 0.5 + 0.3

            x2 = min(x1 + w, self.image_size)
            y2 = min(y1 + h, self.image_size)

            img[y1:y2, x1:x2] = color
            elements.append((x1, y1, x2, y2))

        # Pick one element as target
        target_idx = np.random.randint(len(elements))
        x1, y1, x2, y2 = elements[target_idx]

        # Target click is center of element
        target_x = (x1 + x2) / 2 / self.image_size
        target_y = (y1 + y2) / 2 / self.image_size

        # Random instruction tokens
        tokens = np.random.randint(0, self.vocab_size, size=(self.max_seq_len,))

        # Random action type (mostly clicks)
        action_type = np.random.choice([0, 0, 0, 1, 2], p=[0.6, 0.15, 0.15, 0.05, 0.05])

        return {
            'image': torch.from_numpy(img).float().permute(2, 0, 1),
            'tokens': torch.from_numpy(tokens).long(),
            'coords': torch.tensor([target_x, target_y, target_x, target_y]),
            'action_type': action_type
        }

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx]


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='VLA Data Collection')
    parser.add_argument('--record', action='store_true', help='Start recording')
    parser.add_argument('--output', type=str, default='demos/', help='Output directory')
    parser.add_argument('--test-synthetic', action='store_true', help='Test synthetic data')
    args = parser.parse_args()

    if args.record:
        if not HAS_RECORDING_DEPS:
            print("Install recording dependencies: pip install mss pyautogui pynput")
            exit(1)

        recorder = DemoRecorder(args.output)
        recorder.start()

        print("\n" + "="*50)
        print("Recording started!")
        print("="*50)
        print("\nCommands:")
        print("  i <text>  - Set instruction")
        print("  q         - Quit and save")
        print("\nExample:")
        print("  i click the search button")
        print("  <perform the action>")
        print("  q")
        print("="*50 + "\n")

        while True:
            try:
                cmd = input("> ").strip()
                if cmd == 'q':
                    break
                elif cmd.startswith('i '):
                    recorder.set_instruction(cmd[2:])
            except KeyboardInterrupt:
                break

        recorder.stop()

    elif args.test_synthetic:
        print("Testing synthetic dataset...")
        dataset = SyntheticDataset(num_samples=100)

        sample = dataset[0]
        print(f"Image shape: {sample['image'].shape}")
        print(f"Tokens shape: {sample['tokens'].shape}")
        print(f"Coords: {sample['coords']}")
        print(f"Action type: {sample['action_type']}")

        # Test dataloader
        loader = DataLoader(dataset, batch_size=8, shuffle=True)
        batch = next(iter(loader))
        print(f"\nBatch shapes:")
        print(f"  Image: {batch['image'].shape}")
        print(f"  Tokens: {batch['tokens'].shape}")
        print(f"  Coords: {batch['coords'].shape}")
        print(f"  Action type: {batch['action_type'].shape}")

    else:
        parser.print_help()
