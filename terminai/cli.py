#!/usr/bin/env python3
"""
TerminaI - Command Line Interface

Sovereign System Operator with native PTY and OODA reasoning.

Usage:
    terminai "Audit /var/log for errors"
    terminai --interactive
    terminai --fleet --discover
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src import TerminaIAgent
from src.llm import LLMConfig


def setup_logging(verbose: bool = False, debug: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if debug else (logging.INFO if verbose else logging.WARNING)

    logging.basicConfig(
        level=level,
        format="%(asctime)s │ %(levelname)s │ %(message)s",
        datefmt="%H:%M:%S",
    )


def print_banner():
    """Print TerminaI banner."""
    banner = """
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║   ████████╗███████╗██████╗ ███╗   ███╗██╗███╗   ██╗ █████╗ ██╗   ║
║   ╚══██╔══╝██╔════╝██╔══██╗████╗ ████║██║████╗  ██║██╔══██╗██║   ║
║      ██║   █████╗  ██████╔╝██╔████╔██║██║██╔██╗ ██║███████║██║   ║
║      ██║   ██╔══╝  ██╔══██╗██║╚██╔╝██║██║██║╚██╗██║██╔══██║██║   ║
║      ██║   ███████╗██║  ██║██║ ╚═╝ ██║██║██║ ╚████║██║  ██║██║   ║
║      ╚═╝   ╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝   ║
║                                                                   ║
║              Sovereign System Operator                            ║
║              Powered by Gemini 3 • Native PTY • OODA Loop        ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
"""
    print(banner)


def print_status(key: str, value: str, color: str = ""):
    """Print status line."""
    colors = {
        "green": "\033[92m",
        "yellow": "\033[93m",
        "red": "\033[91m",
        "blue": "\033[94m",
        "reset": "\033[0m",
    }

    c = colors.get(color, "")
    r = colors["reset"]
    print(f"  {c}●{r} {key}: {c}{value}{r}")


async def run_interactive():
    """Run interactive REPL mode."""
    print("\n📍 Interactive Mode")
    print("   Type your commands. Use 'exit' or Ctrl+D to quit.\n")

    agent = TerminaIAgent(
        llm_provider="gemini",
        model="gemini-2.0-flash",
        verbose=True,
    )

    while True:
        try:
            # Get input
            task = input("\033[94m❯\033[0m ").strip()

            if not task:
                continue

            if task.lower() in ["exit", "quit", "q"]:
                print("\n👋 Goodbye!")
                break

            # Execute task
            print()
            result = await agent.run(task)

            # Show result
            if result["success"]:
                print(f"\n\033[92m✓\033[0m Task completed in {result['iterations']} iterations")
            else:
                print(f"\n\033[91m✗\033[0m Task failed: {result.get('error', 'Unknown error')}")

            print()

        except EOFError:
            print("\n\n👋 Goodbye!")
            break
        except KeyboardInterrupt:
            print("\n\n⚠ Interrupted")
            continue


async def run_task(task: str, model: str = "gemini-2.0-flash", provider: str = "gemini"):
    """Run single task."""
    print(f"\n📋 Task: {task}\n")

    agent = TerminaIAgent(
        llm_provider=provider,
        model=model,
        verbose=True,
    )

    # Setup output callback
    def on_output(output: str):
        if output.strip():
            for line in output.split('\n')[:10]:  # Limit output
                print(f"   {line}")

    agent.on_output = on_output

    # Run
    result = await agent.run(task)

    # Summary
    print("\n" + "─" * 60)
    if result["success"]:
        print(f"\033[92m✓ SUCCESS\033[0m in {result['iterations']} iterations")
    else:
        print(f"\033[91m✗ FAILED\033[0m: {result.get('error', 'Unknown')}")

    return result


async def run_fleet_mode(endpoints: list = None):
    """Run in fleet commander mode."""
    from src.fleet import FleetCommander

    print("\n🚀 Fleet Commander Mode")

    commander = FleetCommander()

    if endpoints:
        print(f"   Discovering agents at {len(endpoints)} endpoints...")
        count = await commander.discover(endpoints)
        print(f"   Found {count} agents")

    # Show fleet status
    status = commander.get_fleet_status()
    print(f"\n   Fleet Status:")
    print(f"   • Total Agents: {status['total_agents']}")
    print(f"   • Local: {status['local_agents']}")
    print(f"   • Remote: {status['remote_agents']}")

    return status


def main():
    parser = argparse.ArgumentParser(
        description="TerminaI - Sovereign System Operator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  terminai "Audit /var/log for errors"
  terminai --model gemini-3-pro "Complex architecture analysis"
  terminai --interactive
  terminai --fleet --discover http://agent1:8080 http://agent2:8080

Environment Variables:
  GEMINI_API_KEY    Google Gemini API key
  GOOGLE_API_KEY    Alternative API key variable
  OPENAI_API_KEY    For OpenAI-compatible backends
        """,
    )

    parser.add_argument(
        "task",
        nargs="?",
        help="Task to execute",
    )

    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Interactive REPL mode",
    )

    parser.add_argument(
        "-m", "--model",
        default="gemini-2.0-flash",
        choices=[
            "gemini-2.0-flash",
            "gemini-2.0-pro",
            "gemini-3-flash",
            "gemini-3-pro",
        ],
        help="Model to use (default: gemini-2.0-flash)",
    )

    parser.add_argument(
        "-p", "--provider",
        default="gemini",
        choices=["gemini", "ollama", "openai"],
        help="LLM provider (default: gemini)",
    )

    parser.add_argument(
        "--fleet",
        action="store_true",
        help="Enable Fleet Commander mode",
    )

    parser.add_argument(
        "--discover",
        nargs="*",
        metavar="ENDPOINT",
        help="Discover agents at endpoints",
    )

    parser.add_argument(
        "--mcp",
        nargs="*",
        metavar="SERVER",
        help="Enable MCP servers (github, postgres, slack, etc.)",
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode",
    )

    parser.add_argument(
        "--no-banner",
        action="store_true",
        help="Skip banner",
    )

    args = parser.parse_args()

    # Setup
    setup_logging(args.verbose, args.debug)

    if not args.no_banner:
        print_banner()

    # Check API key
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if args.provider == "gemini" and not api_key:
        print("\033[93m⚠ Warning: GEMINI_API_KEY not set\033[0m")
        print("  Set it with: export GEMINI_API_KEY='your-key'")
        print("  Get a key at: https://aistudio.google.com/app/apikey\n")

    # Show configuration
    if not args.no_banner:
        print_status("Provider", args.provider, "blue")
        print_status("Model", args.model, "blue")
        if args.fleet:
            print_status("Mode", "Fleet Commander", "green")
        elif args.interactive:
            print_status("Mode", "Interactive", "green")
        else:
            print_status("Mode", "Single Task", "green")

    # Run
    try:
        if args.fleet:
            asyncio.run(run_fleet_mode(args.discover))
        elif args.interactive:
            asyncio.run(run_interactive())
        elif args.task:
            asyncio.run(run_task(args.task, args.model, args.provider))
        else:
            parser.print_help()
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\033[91m✗ Error: {e}\033[0m")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
