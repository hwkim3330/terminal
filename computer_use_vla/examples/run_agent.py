#!/usr/bin/env python3
"""
Computer Use Agent - Example Script

This example demonstrates how to use the Computer Use VLA agent
to automate computer tasks.

Usage:
    python run_agent.py --task "Open Chrome and search for hello world"
    python run_agent.py --task "Click on the Settings icon" --dry-run
    python run_agent.py --interactive --task "Navigate to folder"
"""

import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent import create_agent, AgentConfig, ComputerUseAgent


def setup_logging(verbose: bool = True):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
        ],
    )


def print_banner():
    """Print application banner."""
    print("""
╔═══════════════════════════════════════════════════════════════╗
║              Computer Use VLA Agent                           ║
║     Vision-Language-Action Model for Computer Control         ║
║                                                               ║
║  Inspired by NVIDIA Alpamayo + Liquid AI LFM 2.5 VL          ║
╚═══════════════════════════════════════════════════════════════╝
""")


def run_demo():
    """Run a simple demo."""
    print("\n=== Demo Mode ===")
    print("This demo will use dummy mode (no actual computer control)")

    # Create agent in dummy mode
    agent = create_agent(
        model="LiquidAI/LFM2.5-VL-1.6B",
        backend="dummy",  # Use dummy backend for demo
        safe_mode=True,
        dry_run=True,
    )

    task = "Move the mouse to the center of the screen and click"

    print(f"\nTask: {task}")
    print("Starting agent...")

    result = agent.run(task)

    print(f"\nResult:")
    print(f"  Success: {result['success']}")
    print(f"  Steps: {result['steps']}")

    if result['history']:
        print(f"  Actions taken:")
        for h in result['history']:
            print(f"    Step {h['step']}: {h['action']['action_type']}")


def main():
    parser = argparse.ArgumentParser(
        description="Computer Use VLA Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with a specific task
  python run_agent.py --task "Open the browser"

  # Interactive mode with confirmation
  python run_agent.py --interactive --task "Navigate to Documents"

  # Dry run mode (no actual execution)
  python run_agent.py --task "Click the button" --dry-run

  # Use dummy backend for testing
  python run_agent.py --task "Test action" --backend dummy

  # Run demo
  python run_agent.py --demo
        """,
    )

    parser.add_argument(
        "--task",
        type=str,
        help="Task to perform",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive mode with confirmation",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't actually execute actions",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="pyautogui",
        choices=["pyautogui", "xdotool", "dummy"],
        help="Controller backend",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="LiquidAI/LFM2.5-VL-1.6B",
        help="VLM model name",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=50,
        help="Maximum steps",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demo mode",
    )

    args = parser.parse_args()

    # Setup
    print_banner()
    setup_logging(args.verbose)

    # Demo mode
    if args.demo:
        run_demo()
        return

    # Check task
    if not args.task:
        parser.print_help()
        print("\nError: --task is required (or use --demo for demo mode)")
        sys.exit(1)

    # Create agent
    print(f"Creating agent...")
    print(f"  Model: {args.model}")
    print(f"  Backend: {args.backend}")
    print(f"  Dry run: {args.dry_run}")
    print(f"  Interactive: {args.interactive}")

    agent = create_agent(
        model=args.model,
        backend=args.backend,
        safe_mode=True,
        dry_run=args.dry_run,
        interactive=args.interactive,
    )

    # Run task
    print(f"\nTask: {args.task}")
    print("Starting agent... (Press Ctrl+C to stop)")
    print("-" * 50)

    try:
        result = agent.run(args.task)

        print("-" * 50)
        print(f"\nResult:")
        print(f"  Success: {result['success']}")
        print(f"  Steps: {result['steps']}")

        if result['error']:
            print(f"  Error: {result['error']}")

        if result['history']:
            print(f"\nAction history:")
            for h in result['history']:
                status = "OK" if h.get('success', True) else "FAIL"
                print(f"  [{status}] Step {h['step']}: {h['action']['action_type']}")

    except KeyboardInterrupt:
        print("\n\nAgent interrupted by user")
        agent.stop()


if __name__ == "__main__":
    main()
