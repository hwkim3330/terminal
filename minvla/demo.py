"""
MiniVLA Demo / Inference
========================

Run a trained MiniVLA model to control the computer.

Usage:
    # Interactive demo
    python demo.py --model out/best.pt

    # Single inference
    python demo.py --model out/best.pt --screenshot test.png --instruction "click the button"

    # Autonomous mode (careful!)
    python demo.py --model out/best.pt --auto

Author: Karpathy-style refactor
"""

import os
import sys
import time
import argparse
from pathlib import Path

import torch
import numpy as np
from PIL import Image

from minvla import MiniVLA, VLAConfig


# -----------------------------------------------------------------------------
# Screen Capture & Control
# -----------------------------------------------------------------------------

try:
    import mss
    import pyautogui
    pyautogui.FAILSAFE = True  # Move mouse to corner to abort
    HAS_CONTROL = True
except ImportError:
    HAS_CONTROL = False


def capture_screen(image_size=224):
    """Capture current screen."""
    if not HAS_CONTROL:
        raise ImportError("Need mss: pip install mss")

    with mss.mss() as sct:
        monitor = sct.monitors[1]
        screenshot = sct.grab(monitor)
        img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
        screen_size = (monitor["width"], monitor["height"])

    # Resize for model
    img_small = img.resize((image_size, image_size))
    return img, img_small, screen_size


def execute_action(action: dict, dry_run=True):
    """Execute predicted action on screen."""
    if not HAS_CONTROL:
        raise ImportError("Need pyautogui: pip install pyautogui")

    x, y = action['x'], action['y']
    action_type = action['action']

    print(f"  Action: {action_type} at ({x}, {y})")
    print(f"  Confidence: {action['confidence']:.2%}")

    if dry_run:
        print("  [DRY RUN - not executing]")
        return

    # Safety check
    if action['confidence'] < 0.5:
        print("  [LOW CONFIDENCE - skipping]")
        return

    # Execute
    if action_type == 'click':
        pyautogui.click(x, y)
    elif action_type == 'double_click':
        pyautogui.doubleClick(x, y)
    elif action_type == 'right_click':
        pyautogui.rightClick(x, y)
    elif action_type == 'drag':
        pyautogui.moveTo(x, y)
        pyautogui.drag(action['end_x'] - x, action['end_y'] - y)
    elif action_type == 'scroll':
        pyautogui.scroll(3, x, y)  # scroll up
    else:
        print(f"  [UNKNOWN ACTION: {action_type}]")


# -----------------------------------------------------------------------------
# Simple Tokenizer
# -----------------------------------------------------------------------------

class SimpleTokenizer:
    """Character-level tokenizer for demo."""

    def __init__(self, vocab_size=50257):
        self.vocab_size = vocab_size

    def encode(self, text):
        return [ord(c) % self.vocab_size for c in text[:128]]

    def decode(self, tokens):
        return ''.join(chr(t % 128) for t in tokens)


# -----------------------------------------------------------------------------
# Demo
# -----------------------------------------------------------------------------

def load_model(checkpoint_path, device='cuda'):
    """Load trained model."""
    print(f"Loading model from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Infer config from checkpoint if available
    config = VLAConfig()
    model = MiniVLA(config).to(device)

    # Handle compiled models
    state_dict = checkpoint['model']
    model.load_state_dict(state_dict)

    model.eval()
    return model


def run_inference(model, img_small, instruction, tokenizer, device):
    """Run single inference."""
    # Preprocess image
    img_np = np.array(img_small) / 255.0
    img_tensor = torch.from_numpy(img_np).float().permute(2, 0, 1).unsqueeze(0).to(device)

    # Tokenize
    tokens = tokenizer.encode(instruction)
    tokens = torch.tensor(tokens).unsqueeze(0).to(device)

    # Forward
    with torch.no_grad():
        out = model(img_tensor, tokens)

    # Decode
    coords = out['coords'][0].cpu().numpy()
    action_idx = out['action_type'][0].argmax().item()
    conf = out['confidence'][0].item()

    action_types = ['click', 'double_click', 'right_click', 'drag', 'scroll', 'type', 'hotkey']

    return {
        'x': int(coords[0] * 1920),  # Will be adjusted
        'y': int(coords[1] * 1080),
        'end_x': int(coords[2] * 1920),
        'end_y': int(coords[3] * 1080),
        'action': action_types[action_idx],
        'confidence': conf
    }


def interactive_demo(model, device):
    """Interactive demo mode."""
    print("\n" + "="*60)
    print("MiniVLA Interactive Demo")
    print("="*60)
    print("\nCommands:")
    print("  <instruction>  - Execute instruction")
    print("  q              - Quit")
    print("  exec           - Toggle execution (currently: dry run)")
    print("\nExamples:")
    print("  click the search button")
    print("  scroll down")
    print("="*60 + "\n")

    tokenizer = SimpleTokenizer()
    dry_run = True

    while True:
        try:
            user_input = input("\n> ").strip()

            if user_input == 'q':
                break
            elif user_input == 'exec':
                dry_run = not dry_run
                print(f"Execution: {'DRY RUN' if dry_run else 'LIVE'}")
                continue
            elif not user_input:
                continue

            instruction = user_input

            # Capture screen
            print("Capturing screen...")
            img_full, img_small, screen_size = capture_screen()

            # Run inference
            print(f"Instruction: {instruction}")
            action = run_inference(model, img_small, instruction, tokenizer, device)

            # Scale to actual screen
            action['x'] = int(action['x'] * screen_size[0] / 1920)
            action['y'] = int(action['y'] * screen_size[1] / 1080)
            action['end_x'] = int(action['end_x'] * screen_size[0] / 1920)
            action['end_y'] = int(action['end_y'] * screen_size[1] / 1080)

            # Execute
            execute_action(action, dry_run=dry_run)

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

    print("\nBye!")


def single_inference(model, screenshot_path, instruction, device):
    """Single inference from image file."""
    tokenizer = SimpleTokenizer()

    # Load image
    img = Image.open(screenshot_path).convert('RGB')
    img_small = img.resize((224, 224))
    screen_size = img.size

    # Run inference
    action = run_inference(model, img_small, instruction, tokenizer, device)

    # Scale to image size
    action['x'] = int(action['x'] * screen_size[0] / 1920)
    action['y'] = int(action['y'] * screen_size[1] / 1080)

    print(f"\nInstruction: {instruction}")
    print(f"Predicted action: {action['action']} at ({action['x']}, {action['y']})")
    print(f"Confidence: {action['confidence']:.2%}")

    return action


def autonomous_demo(model, device, max_steps=10, delay=2.0):
    """
    Autonomous mode - model controls computer.
    USE WITH CAUTION!
    """
    print("\n" + "!"*60)
    print("AUTONOMOUS MODE")
    print("Move mouse to corner to abort!")
    print("!"*60)

    tokenizer = SimpleTokenizer()

    # Simple task
    instruction = input("\nEnter task instruction: ").strip()

    for step in range(max_steps):
        print(f"\n--- Step {step+1}/{max_steps} ---")

        # Capture screen
        img_full, img_small, screen_size = capture_screen()

        # Run inference
        action = run_inference(model, img_small, instruction, tokenizer, device)

        # Scale
        action['x'] = int(action['x'] * screen_size[0] / 1920)
        action['y'] = int(action['y'] * screen_size[1] / 1080)

        # Execute
        execute_action(action, dry_run=False)

        time.sleep(delay)

    print("\nAutonomous demo complete.")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Model checkpoint')
    parser.add_argument('--screenshot', type=str, help='Single screenshot inference')
    parser.add_argument('--instruction', type=str, help='Instruction for single inference')
    parser.add_argument('--auto', action='store_true', help='Autonomous mode (careful!)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    # Load model
    model = load_model(args.model, args.device)

    if args.screenshot and args.instruction:
        # Single inference
        single_inference(model, args.screenshot, args.instruction, args.device)
    elif args.auto:
        # Autonomous mode
        if not HAS_CONTROL:
            print("Autonomous mode requires: pip install mss pyautogui")
            return
        autonomous_demo(model, args.device)
    else:
        # Interactive demo
        if not HAS_CONTROL:
            print("Interactive mode requires: pip install mss pyautogui")
            return
        interactive_demo(model, args.device)


if __name__ == '__main__':
    main()
