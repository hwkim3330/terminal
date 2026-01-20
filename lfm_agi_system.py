#!/usr/bin/env python3
"""
LFM AGI System - 통합 시스템

Liquid AI의 LFM 2.5 모델을 기반으로 한 AGI 시스템:
- TerminaI: 시스템 운영 에이전트 (LFM 2.5 Instruct)
- Computer Use VLA: 컴퓨터 제어 에이전트 (LFM 2.5 VL)

특징:
- OODA Loop 추론
- Native PTY 터미널 제어
- 비전-언어 이해
- MCP/A2A 프로토콜 지원
- Fleet Commander 멀티 에이전트
"""

import asyncio
import logging
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, Optional, List

# 경로 설정
sys.path.insert(0, str(Path(__file__).parent / "terminai"))
sys.path.insert(0, str(Path(__file__).parent / "computer_use_vla"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class LFMAGIConfig:
    """LFM AGI 시스템 설정."""
    # TerminaI 설정
    terminai_model: str = "LiquidAI/LFM2.5-1.2B-Instruct"
    terminai_device: str = "auto"

    # Computer Use 설정
    computer_use_model: str = "LiquidAI/LFM2.5-VL-1.6B"
    computer_use_device: str = "auto"

    # 공통 설정
    dtype: str = "bfloat16"
    max_iterations: int = 50
    verification_enabled: bool = True
    language: str = "auto"


class LFMAGISystem:
    """
    LFM AGI 통합 시스템.

    두 가지 모드 지원:
    1. Terminal Mode: 시스템 운영 (TerminaI)
    2. GUI Mode: 컴퓨터 제어 (Computer Use VLA)
    """

    def __init__(self, config: Optional[LFMAGIConfig] = None):
        self.config = config or LFMAGIConfig()
        self._terminai = None
        self._computer_use = None

    @property
    def terminai(self):
        """TerminaI 에이전트 (lazy load)."""
        if self._terminai is None:
            from terminai.src.core.lfm_agent import LFMAgent, LFMAgentConfig
            self._terminai = LFMAgent(LFMAgentConfig(
                model_id=self.config.terminai_model,
                device=self.config.terminai_device,
                dtype=self.config.dtype,
                max_iterations=self.config.max_iterations,
                verification_enabled=self.config.verification_enabled,
                language=self.config.language,
            ))
        return self._terminai

    @property
    def computer_use(self):
        """Computer Use 에이전트 (lazy load)."""
        if self._computer_use is None:
            from computer_use_vla.src.models.lfm_vl_model import LFMVLAgent, LFMVLConfig
            self._computer_use = LFMVLAgent(LFMVLConfig(
                model_id=self.config.computer_use_model,
                device=self.config.computer_use_device,
                dtype=self.config.dtype,
            ))
        return self._computer_use

    async def run_terminal(self, task: str, context: Optional[str] = None) -> Dict[str, Any]:
        """
        터미널 모드로 태스크 실행.

        Args:
            task: 실행할 태스크
            context: 추가 컨텍스트

        Returns:
            실행 결과
        """
        logger.info(f"[Terminal Mode] Task: {task}")
        return await self.terminai.run(task, context)

    async def run_gui(self, task: str, context: Optional[str] = None) -> Dict[str, Any]:
        """
        GUI 모드로 태스크 실행.

        Args:
            task: 실행할 태스크
            context: 추가 컨텍스트

        Returns:
            실행 결과
        """
        logger.info(f"[GUI Mode] Task: {task}")
        return await self.computer_use.run(task, context=context)

    async def run_auto(self, task: str, context: Optional[str] = None) -> Dict[str, Any]:
        """
        자동 모드 선택으로 태스크 실행.

        GUI 관련 키워드가 있으면 GUI 모드, 아니면 터미널 모드.
        """
        gui_keywords = [
            "click", "button", "browser", "chrome", "firefox",
            "window", "gui", "screen", "mouse", "open app",
            "클릭", "버튼", "브라우저", "화면", "마우스", "앱 열기",
        ]

        task_lower = task.lower()
        use_gui = any(kw in task_lower for kw in gui_keywords)

        if use_gui:
            return await self.run_gui(task, context)
        else:
            return await self.run_terminal(task, context)


def print_banner():
    """배너 출력."""
    print("""
╔═══════════════════════════════════════════════════════════════════════╗
║                                                                       ║
║   ██╗     ███████╗███╗   ███╗     █████╗  ██████╗ ██╗                ║
║   ██║     ██╔════╝████╗ ████║    ██╔══██╗██╔════╝ ██║                ║
║   ██║     █████╗  ██╔████╔██║    ███████║██║  ███╗██║                ║
║   ██║     ██╔══╝  ██║╚██╔╝██║    ██╔══██║██║   ██║██║                ║
║   ███████╗██║     ██║ ╚═╝ ██║    ██║  ██║╚██████╔╝██║                ║
║   ╚══════╝╚═╝     ╚═╝     ╚═╝    ╚═╝  ╚═╝ ╚═════╝ ╚═╝                ║
║                                                                       ║
║              Liquid Foundation Model AGI System                       ║
║                                                                       ║
║   • LFM 2.5 Instruct (1.2B) - Terminal Operations                    ║
║   • LFM 2.5 VL (1.6B) - Vision-Language-Action                       ║
║   • OODA Loop Reasoning                                               ║
║   • Native PTY Terminal                                               ║
║   • MCP + A2A Protocol Support                                        ║
║                                                                       ║
╚═══════════════════════════════════════════════════════════════════════╝
    """)


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="LFM AGI System")
    parser.add_argument("task", nargs="?", help="Task to execute")
    parser.add_argument("-m", "--mode", choices=["terminal", "gui", "auto"],
                        default="auto", help="Execution mode")
    parser.add_argument("-i", "--interactive", action="store_true",
                        help="Interactive mode")
    parser.add_argument("--no-banner", action="store_true",
                        help="Skip banner")

    args = parser.parse_args()

    if not args.no_banner:
        print_banner()

    system = LFMAGISystem()

    if args.interactive:
        print("\n📍 Interactive Mode (type 'exit' to quit)\n")
        while True:
            try:
                task = input("\033[94m❯\033[0m ").strip()
                if task.lower() in ["exit", "quit", "q"]:
                    break
                if not task:
                    continue

                result = await system.run_auto(task)
                print(f"\n{'✓' if result['success'] else '✗'} "
                      f"Completed in {result.get('iterations', result.get('steps', 0))} steps\n")

            except KeyboardInterrupt:
                print("\n")
                continue
            except EOFError:
                break

    elif args.task:
        if args.mode == "terminal":
            result = await system.run_terminal(args.task)
        elif args.mode == "gui":
            result = await system.run_gui(args.task)
        else:
            result = await system.run_auto(args.task)

        print(f"\n{'✓' if result['success'] else '✗'} "
              f"Completed in {result.get('iterations', result.get('steps', 0))} steps")

    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())
